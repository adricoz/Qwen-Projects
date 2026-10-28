"""Regressiveur a processus gaussien (moyenne nulle, bruit d'observation)."""
import numpy as np
from scipy.linalg import cho_factor, cho_solve, cholesky, solve
from scipy.optimize import minimize


class GPRegressor:
    """Regressiveur par processus gaussien a noyau stationnaire ARD.

    Parameters
    ----------
    kernel : object
        Noyau ayant les methodes ``K(X1, X2)`` et ``diag(X)`` (voir
        :mod:`gp_lite.kernels`). Ses attributs ``lengthscales`` (d,) et
        ``variance`` sont modifies par :meth:`fit`.
    noise : float, optional
        Bruit d'observation initial (defaut 1e-6) ; il est optimise en
        espace logarithmique lors de :meth:`fit`.
    normalize_y : bool, optional
        Si True (defaut), la cible est standardisee avant l'optimisation
        (moyenne 0, variance 1) ; les predictions sont renvoyees dans
        l'unite d'origine.
    """

    # bornes de recherche (espace lineaire, valeurs positives)
    LENGTHSCALE_BOUNDS = (1e-2, 1e1)
    VARIANCE_BOUNDS = (1e-4, 1e2)
    NOISE_BOUNDS = (1e-10, 1e-2)
    # jitters relatifs tries quand le Cholesky echoue
    JITTERS = (0.0, 1e-10, 2e-8, 2e-6)
    NLL_FAILURE = 1e10

    def __init__(self, kernel, noise=1e-6, normalize_y=True):
        self.kernel = kernel
        self.noise = float(noise)
        self.normalize_y = bool(normalize_y)
        self.X_train = None
        self.y_train = None
        self.y_mean = 0.0
        self.y_std = 1.0
        self.lengthscales = np.array(self.kernel.lengthscales, dtype=float)
        self.variance = float(self.kernel.variance)
        self._L = None
        self._alpha = None
        self._beta_diag = None
        self.log_marginal_likelihood_ = None

    # ------------------------------------------------------------------ #
    # likelihood                                                         #
    # ------------------------------------------------------------------ #
    def _nll(self, log_params, y_n):
        """Negative log-likelihood marginale, parametree en espace log.

        Parameters
        ----------
        log_params : ndarray, shape (d + 2,)
            Log de [lengthscales (d,), variance, noise].
        y_n : ndarray, shape (n,)
            Cibles (standardisees si ``normalize_y``).

        Returns
        -------
        float
            Valeur de la NLL, ou ``NLL_FAILURE`` si le Cholesky echoue.
        """
        params = np.exp(log_params)
        self.kernel.lengthscales = params[:-2].copy()
        self.kernel.variance = float(params[-2])
        noise = float(params[-1])
        K = self.kernel.K(self.X_train, self.X_train)
        n = y_n.shape[0]
        L = None
        for jitter in self.JITTERS:
            C = K + (noise + jitter * np.trace(K) / n) * np.eye(n)
            try:
                L, _ = cho_factor(C, lower=True, check_finite=False)
                break
            except Exception:
                continue
        if L is None:
            return self.NLL_FAILURE
        alpha = cho_solve((L, _), y_n, check_finite=False)[0]
        nll = (0.5 * float(y_n @ alpha) + np.sum(np.log(np.diag(L)))
               + 0.5 * n * np.log(2.0 * np.pi))
        return float(nll)

    # ------------------------------------------------------------------ #
    # conditioning and prediction                                        #
    # ------------------------------------------------------------------ #
    def condition(self, X, y_n):
        """Conditionner le GP aux donnees avec les hyperparametres courants.

        Parameters
        ----------
        X : ndarray, shape (n, d)
        y_n : ndarray, shape (n,)
            Cibles dans l'espace de la likelihood (standardisees le cas
            echeant, deja preparees par l'appelant).

        Returns
        -------
        self
        """
        self.X_train = np.atleast_2d(np.asarray(X, dtype=float))
        self.y_train = np.asarray(y_n, dtype=float)
        n = self.X_train.shape[0]
        K = self.kernel.K(self.X_train, self.X_train)
        L = None
        for jitter in self.JITTERS:
            C = K + (self.noise + jitter * np.trace(K) / n) * np.eye(n)
            try:
                L, _ = cho_factor(C, lower=True, check_finite=False)
                break
            except Exception:
                continue
        if L is None:
            raise np.linalg.LinAlgError("Cholesky echoue meme avec jitter")
        self._L = L
        self._alpha = cho_solve((L, _), self.y_train, check_finite=False)[0]
        # diagonale de C^-1 : C = L L^T -> C^-1 = L^-T L^-1, diagonale =
        # somme sur les colonnes de (L^-1)^2
        L_inv = cholesky(L, lower=True, check_finite=False)
        M = solve(L_inv, np.eye(n), lower=False, check_finite=False)
        self._beta_diag = np.einsum("ij,ij->i", M, M)
        self.log_marginal_likelihood_ = float(
            0.5 * self.y_train @ self._alpha
            + np.sum(np.log(np.diag(L))) + 0.5 * n * np.log(2.0 * np.pi))
        return self

    def predict(self, X, return_std=False):
        """Moyenne (et ecart-type) a posteriori sur de nouveaux points.

        Parameters
        ----------
        X : ndarray, shape (m, d)
        return_std : bool, optional
            Si True, renvoyer aussi l'ecart-type a posteriori.

        Returns
        -------
        mean : ndarray, shape (m,)
            Moyenne predite dans l'unite d'origine de y.
        std : ndarray, shape (m,), optional
            Ecart-type a posteriori (si ``return_std=True``).
        """
        if self._alpha is None:
            raise RuntimeError("le modele n'est pas conditionne (appeler fit)")
        X = np.atleast_2d(np.asarray(X, dtype=float))
        K_s = self.kernel.K(X, self.X_train)          # (m, n)
        mean = self.y_mean + self.y_std * (K_s @ self._alpha)
        if return_std:
            v = self.kernel.diag(X) - np.einsum(
                "ij,ij->i", K_s, K_s * self._alpha)
            std = np.sqrt(np.clip(v, 0.0, None)) * self.y_std
            return mean, std
        return mean

    def log_marginal_likelihood(self):
        """Log-vraisemblance marginale aux hyperparametres courants.

        Returns
        -------
        float
        """
        if self.log_marginal_likelihood_ is None:
            raise RuntimeError("le modele n'est pas conditionne (appeler fit)")
        return float(self.log_marginal_likelihood_)

    def loo(self):
        """Residus leave-one-out en forme fermee (Rasmussen & Williams 5.4.2).

        Avec C = K + s^2 I et alpha = C^-1 y, la prediction leave-one-out est
        y_{-i} = -alpha_i / beta_ii avec beta_ii = (C^-1)_ii ; le residu
        renvoye est y_i - y_{-i}, dans l'unite d'origine de y.

        Returns
        -------
        ndarray, shape (n,)
        """
        if self._alpha is None:
            raise RuntimeError("le modele n'est pas conditionne (appeler fit)")
        y_loo_n = -self._alpha / self._beta_diag
        return self.y_std * (self.y_train - y_loo_n)

    # ------------------------------------------------------------------ #
    # hyperparameter optimization                                        #
    # ------------------------------------------------------------------ #
    def fit(self, X, y, n_restarts=3, seed=0):
        """Optimiser les hyperparametres puis conditionner le modele.

        L'espace optimise est le log de [lengthscales, variance, noise]
        (L-BFGS-B). ``n_restarts`` demarrages aleatoires (graine ``seed``),
        le meilleur minimum est conserve.

        Parameters
        ----------
        X : ndarray, shape (n, d)
        y : ndarray, shape (n,)
        n_restarts : int, optional
            Nombre de redemarrages aleatoires (defaut 3).
        seed : int, optional
            Graine de l'alea pour les demarrages (defaut 0).

        Returns
        -------
        self
        """
        X = np.atleast_2d(np.asarray(X, dtype=float))
        y = np.asarray(y, dtype=float)
        self.X_train = X
        d = np.atleast_1d(self.kernel.lengthscales).shape[0]
        lb = np.concatenate([
            np.full(d, float(np.log(self.LENGTHSCALE_BOUNDS[0]))),
            [float(np.log(self.VARIANCE_BOUNDS[0])),
             float(np.log(self.NOISE_BOUNDS[0]))]])
        ub = np.concatenate([
            np.full(d, float(np.log(self.LENGTHSCALE_BOUNDS[1]))),
            [float(np.log(self.VARIANCE_BOUNDS[1])),
             float(np.log(self.NOISE_BOUNDS[1]))]])
        # cible pour la likelihood (standardisee le cas echeant)
        if self.normalize_y:
            self.y_mean = float(np.mean(y))
            self.y_std = max(float(np.std(y)), 1e-15)
            y_n = (y - self.y_mean) / self.y_std
        else:
            self.y_mean, self.y_std = 0.0, 1.0
            y_n = y.copy()
        # demarrage : hyperparametres unitaires, bruit initial preserve
        self.kernel.lengthscales = np.ones(d)
        self.kernel.variance = 1.0
        self.noise = max(float(self.noise), self.NOISE_BOUNDS[0])
        rng = np.random.default_rng(seed)
        best = None
        for _ in range(max(1, n_restarts)):
            x0 = rng.uniform(lb, ub)
            res = minimize(self._nll, x0, args=(y_n,), method="L-BFGS-B",
                           bounds=list(zip(lb, ub)),
                           options={"maxiter": 300, "ftol": 1e-10, "gtol": 1e-8})
            if best is None or res.fun < best.fun:
                best = res
        params = np.exp(best.x)
        self.kernel.lengthscales = params[:d].copy()
        self.lengthscales = params[:d].copy()
        self.kernel.variance = float(params[d])
        self.variance = float(params[d])
        self.noise = float(params[-1])
        # conditionnement final sur la cible raw (condition attend les cibles
        # standardisees ; y_n est renvoye tel quel, y raw reconstruit ensuite)
        self.condition(X, y_n)
        return self
