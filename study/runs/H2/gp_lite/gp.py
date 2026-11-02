"""Regression par processus gaussien : ajustement, prediction, leave-one-out."""
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize

# valeurs relatives ajoutees a la diagonale si la factorisation de Cholesky echoue
JITTERS = (0.0, 1e-10, 1e-8, 1e-6, 1e-4)
# bornes (en espace log) des hyperparametres, valables pour un signal normalise
_BOUNDS = {
    "lengthscales": (np.log(1e-2), np.log(1e3)),
    "variance": (np.log(1e-4), np.log(1e4)),
    "noise": (np.log(1e-12), np.log(1e-1)),
}
NLL_FAILURE = 1e10


def _safe_cho_factor(M, max_jitter=1e-4):
    """Cholesky de M avec jitters croissants si elle echoue.

    Returns
    -------
    c_factor : tuple
        Facteurisee de Cholesky de ``M + jitter * I``.
    jitter : float
        Jitter effectivement utilise (0.0 si la factorisation de base a reussi).
    """
    n = M.shape[0]
    last_err = None
    for jitter in (0.0, 1e-10, 1e-8, 1e-6, max_jitter):
        try:
            c, low = cho_factor(M + jitter * np.eye(n))
            return c, low
        except np.linalg.LinAlgError as err:
            last_err = err
    raise np.linalg.LinAlgError(
        "Cholesky failed even with jitter %g" % max_jitter) from last_err


class GPRegressor:
    """Regression par processus gaussien a moyenne nulle, bruit blanc gaussien.

    Parameters
    ----------
    kernel : object
        Noyau de ``gp_lite.kernels`` (ex. ``SquaredExponentialARD``).
    noise : float, optional
        Variance du bruit d'observation. Par defaut 1e-6.
    normalize_y : bool, optional
        Si True, ``y`` est standardise (moyenne 0, variance 1) pendant
        l'ajustement ; les predictions sont renvoyees dans les unites de ``y``.
    """

    def __init__(self, kernel, noise=1e-6, normalize_y=True):
        self.kernel = kernel
        self.noise = float(noise)
        self.normalize_y = bool(normalize_y)
        self.X_train = None
        self.y_mean = 0.0
        self.y_std = 1.0
        self._alpha = None      # M^{-1} y_n
        self._S = None          # M^{-1}
        self._log_lik = None

    # ------------------------------------------------------------------ #
    # conditionnement / vraisemblance
    # ------------------------------------------------------------------ #
    def log_marginal_likelihood(self):
        """Log-vraisemblance marginale aux hyperparametres courants.

        Returns
        -------
        float
            log p(y_n | hyperparametres) = -0.5 y^T M^{-1} y
            - 0.5 log|K + noise I| - 0.5 n log(2 pi)
            (sur ``y_n`` si ``normalize_y=True``).
        """
        if self.X_train is None:
            raise RuntimeError("fit() must be called first")
        nll = self._nll_from(self.kernel.K(self.X_train, self.X_train)
                             + self.noise * np.eye(self.X_train.shape[0]))
        return -nll

    def _nll_from(self, M):
        """Negatif de la log-vraisemblance pour une matrice M donnee."""
        factor, low = _safe_cho_factor(M)
        alpha = cho_solve((factor, low), self._y_n)
        # log|det(M)| = 2 * somme des logs des diagonales du facteur
        logdet = 2.0 * np.sum(np.log(np.abs(np.diag(factor))))
        nll = 0.5 * self._y_n @ alpha + 0.5 * logdet \
            + 0.5 * self.X_train.shape[0] * np.log(2.0 * np.pi)
        return nll

    # ------------------------------------------------------------------ #
    # ajustement
    # ------------------------------------------------------------------ #
    def _nll(self, log_params):
        """Negatif de la log-vraisemblance en fonction de log(l), log(sig), log(noise).

        Parameters
        ----------
        log_params : ndarray, shape (d + 2,)
            Log des lengthscales, de la variance et de la variance du bruit.

        Returns
        -------
        float
            NLL (1e10 si la factorisation echoue).
        """
        d = self.kernel.lengthscales.shape[0]
        # bornes sur les log-parametres : evite les depassements en exp
        params = np.exp(np.clip(log_params, -30.0, 30.0))
        self.kernel.lengthscales = params[:d].copy()
        self.kernel.variance = float(params[d])
        self.noise = float(params[d + 1])
        try:
            M = self.kernel.K(self.X_train, self.X_train) \
                + self.noise * np.eye(self.X_train.shape[0])
            nll = self._nll_from(M)
            return nll if np.isfinite(nll) else NLL_FAILURE
        except (np.linalg.LinAlgError, ValueError, OverflowError):
            return NLL_FAILURE

    def fit(self, X, y, optimize=True):
        """Ajuste le processus gaussien aux donnees ``(X, y)``.

        Parameters
        ----------
        X : array_like, shape (n, d)
            Points d'entree.
        y : array_like, shape (n,)
            Valeurs observees.
        optimize : bool, optional
            Si True, optimise log(lengthscales), log(variance), log(noise)
            par L-BFGS-B sur la log-vraisemblance negative (multiples
            demarrages, graine fixee, meilleur retenu). Sinon les
            hyperparametres courants sont conserves.

        Returns
        -------
        self : GPRegressor
        """
        self.X_train = np.atleast_2d(np.asarray(X, dtype=float))
        y = np.asarray(y, dtype=float).reshape(-1)
        n = self.X_train.shape[0]
        if self.X_train.shape[0] != y.shape[0]:
            raise ValueError("X and y have different lengths")

        if self.normalize_y:
            self.y_mean = float(np.mean(y))
            self.y_std = float(np.std(y, ddof=0))
            if self.y_std == 0.0:
                self.y_std = 1.0
            self._y_n = (y - self.y_mean) / self.y_std
        else:
            self.y_mean = 0.0
            self.y_std = 1.0
            self._y_n = y.copy()

        if optimize:
            self._optimize(n)
        self._condition(n)
        return self

    def _optimize(self, n):
        """Optimise les hyperparametres par L-BFGS-B multi-demarrages.

        Strategie (robuste sur de petits jeux de donnees) :
        1. depart : hyperparametres courants + redemarrages log-uniformes
           (graine fixee) + grille grossiere quand d <= 3 ;
        2. affinement L-BFGS-B de chaque point ;
        3. voisinage des meilleurs points, puis un dernier affinement.
        Le meilleur minimum (meilleure log-vraisemblance) est retenu.
        """
        d = self.kernel.lengthscales.shape[0]
        rng = np.random.RandomState(42)
        saved = (self.kernel.lengthscales.copy(), self.kernel.variance, self.noise)
        lo = np.concatenate([np.full(d, _BOUNDS["lengthscales"][0]),
                             [_BOUNDS["variance"][0], _BOUNDS["noise"][0]]])
        hi = np.concatenate([np.full(d, _BOUNDS["lengthscales"][1]),
                             [_BOUNDS["variance"][1], _BOUNDS["noise"][1]]])
        bounds = list(zip(lo, hi))
        n_par = len(lo)

        def start_from_current():
            return np.clip(np.concatenate([np.log(np.maximum(self.kernel.lengthscales, 1e-300)),
                                           [np.log(max(self.kernel.variance, 1e-300)),
                                            np.log(max(self.noise, 1e-300))]]),
                           lo, hi)

        starts = [start_from_current()]
        # redemarrages aleatoires log-uniformes
        for _ in range(12):
            starts.append(lo + (hi - lo) * rng.rand(n_par))
        # grille grossiere (utile quand d est petit, ex. d = 2)
        if d <= 2:
            g_ls = np.linspace(lo[0], hi[0], 4)
            g_var = np.linspace(lo[d], hi[d], 4)
            g_nz = np.linspace(lo[d + 1], hi[d + 1], 2)
            for i1 in range(4):
                for iv in range(4):
                    for ino in range(2):
                        x = np.empty(n_par)
                        x[:d] = g_ls[i1]
                        x[d] = g_var[iv]
                        x[d + 1] = g_nz[ino]
                        starts.append(x)

        results = []
        for x0 in starts:
            res = minimize(self._nll, x0, method="L-BFGS-B",
                           bounds=bounds, options={"maxiter": 1000})
            if res.success:
                results.append(res)
        if not results:
            self.kernel.lengthscales, self.kernel.variance, self.noise = saved
            return
        results.sort(key=lambda r: r.fun)
        # affinement local : voisinage des 3 meilleurs points
        for base in results[:3]:
            for _ in range(4):
                pert = base.x + 0.5 * rng.randn(n_par)
                pert = np.clip(pert, lo + 1e-9, hi - 1e-9)
                res = minimize(self._nll, pert, method="L-BFGS-B",
                               bounds=bounds, options={"maxiter": 1000})
                if res.success:
                    results.append(res)
        results.sort(key=lambda r: r.fun)
        best = results[0]
        # derniere passe d'affinement a partir du meilleur point
        res = minimize(self._nll, best.x, method="L-BFGS-B",
                       bounds=bounds, options={"gtol": 1e-8, "maxiter": 2000})
        if res.success and res.fun < best.fun:
            best = res
        params = np.exp(np.clip(best.x, -30.0, 30.0))
        self.kernel.lengthscales = params[:d].copy()
        self.kernel.variance = float(params[d])
        self.noise = float(params[d + 1])

    def _condition(self, n):
        """Conditionne le GP sur les donnees d'entrainement (alpha, S)."""
        M = self.kernel.K(self.X_train, self.X_train) + self.noise * np.eye(n)
        factor, low = _safe_cho_factor(M)
        self._alpha = cho_solve((factor, low), self._y_n)
        self._S = cho_solve((factor, low), np.eye(n))
        # memoise la log-vraisemblance aux hyperparametres finaux
        logdet = 2.0 * np.sum(np.log(np.abs(np.diag(factor))))
        self._log_lik = (-0.5 * self._y_n @ self._alpha - 0.5 * logdet
                         - 0.5 * n * np.log(2.0 * np.pi))

    # ------------------------------------------------------------------ #
    # prediction
    # ------------------------------------------------------------------ #
    def predict(self, X, return_std=False):
        """Prediction a posteriori moyenne (et ecart-type) aux points X.

        Parameters
        ----------
        X : array_like, shape (m, d)
            Points de prediction.
        return_std : bool, optional
            Si True, renvoie aussi l'ecart-type de ``f`` (sans le bruit).

        Returns
        -------
        mean : ndarray, shape (m,)
            Moyenne predictive dans les unites de ``y``.
        std : ndarray, shape (m,)
            Ecart-type predictive (uniquement si ``return_std=True``).
        """
        if self.X_train is None:
            raise RuntimeError("fit() must be called first")
        X = np.atleast_2d(np.asarray(X, dtype=float))
        k = self.kernel.K(X, self.X_train)
        f_mean_n = k @ self._alpha
        mean = self.y_mean + self.y_std * f_mean_n
        if not return_std:
            return mean
        cov = self.kernel.K(X, X) - k @ (self._S @ k.T)
        var = np.maximum(np.diag(cov), 0.0)
        return mean, np.sqrt(var) * self.y_std

    def loo(self):
        """Residus leave-one-out en forme fermee (Rasmussen & Williams 5.4.2).

        Pour chaque point d'entrainement i, moyenne et variance predictive
        de l'observation y_i (bruit inclus) en ne conditionnant que sur
        y_{-i} : derive par complement de Schur sans refaire n ajustements.

        Returns
        -------
        mean : ndarray, shape (n,)
            Moyennes leave-one-out dans les unites de ``y``.
        var : ndarray, shape (n,)
            Variances leave-one-out dans les unites de ``y``.
        """
        if self.X_train is None:
            raise RuntimeError("fit() must be called first")
        # complement de Schur :
        #   mean_i = y_i - (M^{-1} y)_i / (M^{-1})_ii
        #   var_i  = 1 / (M^{-1})_ii
        d = np.diag(self._S)                     # (M^{-1})_ii
        beta = self._S @ self._y_n
        mean_n = self._y_n - beta / d
        var_n = 1.0 / d
        mean = self.y_mean + self.y_std * mean_n
        var = (self.y_std ** 2) * var_n
        return mean, var
