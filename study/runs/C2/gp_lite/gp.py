"""Régresseur par processus gaussien (GP) de gp_lite.

Conventions numériques (simples et directes) :
    K    = k(X, X) + noise I
    Kinv = K^{-1}          (calculé par inversion, n modeste)
    alpha = Kinv @ y_n
    L    = cholesky(K)     (uniquement pour le log-déterminant)
"""
import numpy as np
from scipy.optimize import minimize

from .kernels import Kernel


class GPRegressor:
    """Régression par processus gaussien à moyenne nulle (Rasmussen & Williams 5.4).

    Parameters
    ----------
    kernel : gp_lite.kernels.Kernel
        Noyau dont les hyperparamètres (``lengthscales``, ``variance``) sont ajustés.
    noise : float, default=1e-6
        Variance du bruit d'observation (sur y normalisé si ``normalize_y=True``).
    normalize_y : bool, default=True
        Si True, y est centré/réduit (``np.std``, ddof=0) avant l'apprentissage ;
        les prédictions sont renvoyées dans les unités de y.

    Notes
    -----
    Formules (n points, K = k(X, X) + noise I, alpha = K^{-1} y_n) :

    - log p(y_n|X) = -0.5 y_n^T alpha - 0.5 log det(K) - (n/2) log 2*pi ;
    - moyenne en x* : k*^T alpha ; variance de f : k(x*, x*) - k*^T K^{-1} k* ;
    - leave-one-out (R&W eq. 5.12) : mu_-i = y_n,i - alpha_i / [K^{-1}]_ii,
      var_-i = 1 / [K^{-1}]_ii (variance de y_i, bruit inclus).
    """

    # jitters relatifs essayés si la factorisation de Cholesky échoue
    JITTERS = (0.0, 1e-10, 1e-8, 1e-6)
    NLL_FAILURE = 1e10
    # bornes de recherche (valeurs, pas logs) des hyperparamètres
    LENGTHSCALE_BOUNDS = (1e-2, 1e1)
    SIGNAL_VARIANCE_BOUNDS = (1e-4, 1e2)
    NOISE_BOUNDS = (1e-8, 1e-2)

    def __init__(self, kernel, noise=1e-6, normalize_y=True):
        if not isinstance(kernel, Kernel):
            raise TypeError("kernel doit être un objet gp_lite.kernels.Kernel")
        self.kernel = kernel
        self.noise = float(noise)
        self.normalize_y = bool(normalize_y)
        self.x_train = None
        self.y_train = None
        self.y_mean = 0.0
        self.y_std = 1.0
        self._K = None
        self._Kinv = None
        self._alpha = None
        self._log_det = 0.0
        self._last_jitter = 0.0

    # ------------------------------------------------------------------ #
    # factorisation interne
    # ------------------------------------------------------------------ #
    def _condition(self):
        """Recalcule K, K^{-1}, alpha, log det(K) aux hyperparamètres courants.

        Raises
        ------
        numpy.linalg.LinAlgError
            Si la factorisation de Cholesky échoue même avec jitter.
        """
        X = self.x_train
        n = len(X)
        for j in self.JITTERS:
            if j < self._last_jitter:
                continue
            K = self.kernel.K(X, X) + (self.noise + j) * np.eye(n)
            try:
                L = np.linalg.cholesky(K)  # contrôle la validité + log det
            except np.linalg.LinAlgError:
                continue
            self._K = K
            self._Kinv = np.linalg.inv(K)
            self._alpha = self._Kinv @ self.y_train_n
            self._log_det = 2.0 * np.sum(np.log(np.diag(L)))
            self._last_jitter = j
            return
        raise np.linalg.LinAlgError("Cholesky failed even with jitter")

    # ------------------------------------------------------------------ #
    # vraisemblance
    # ------------------------------------------------------------------ #
    def _nll(self):
        """Négative log-vraisemblance marginale aux hyperparamètres courants.

        Returns
        -------
        float
            Valeur finie, ou ``NLL_FAILURE`` si la factorisation échoue.
        """
        try:
            self._condition()
        except np.linalg.LinAlgError:
            return self.NLL_FAILURE
        n = len(self.y_train_n)
        return (0.5 * float(self.y_train_n @ self._alpha)
                + 0.5 * self._log_det + 0.5 * n * np.log(2.0 * np.pi))

    def log_marginal_likelihood(self):
        """Log-vraisemblance marginale (sur y_n si normalisé) aux HP courants.

        Returns
        -------
        float
        """
        if self._K is None:
            raise RuntimeError("fit() doit être appelé avant log_marginal_likelihood()")
        n = len(self.y_train_n)
        return (-0.5 * float(self.y_train_n @ self._alpha)
                - 0.5 * self._log_det - 0.5 * n * np.log(2.0 * np.pi))

    # ------------------------------------------------------------------ #
    # fit
    # ------------------------------------------------------------------ #
    def fit(self, X, y, optimize=True):
        """Apprend (ou se contente de conditionner) le GP sur (X, y).

        Parameters
        ----------
        X : ndarray, shape (n, d)
            Points d'apprentissage.
        y : ndarray, shape (n,)
            Sorties observées.
        optimize : bool, default=True
            Si True, optimise log(lengthscales), log(variance), log(noise) par
            L-BFGS-B sur la log-vraisemblance négative (3 redémarrages aléatoires,
            graine fixée, meilleur retenu). Si False, garde les hyperparamètres
            actuels de ``self.kernel`` et ``self.noise``.

        Returns
        -------
        self
        """
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        y = np.asarray(y, dtype=float).reshape(-1)
        d = X.shape[1]
        if X.shape[1] != self.kernel.lengthscales.shape[0]:
            raise ValueError(
                f"kernel.lengthscales a {self.kernel.lengthscales.shape[0]} "
                f"dimensions, X en a {d}"
            )

        self.x_train = X
        self.y_train = y
        if self.normalize_y:
            self.y_mean = float(np.mean(y))
            self.y_std = float(np.std(y))
            if self.y_std == 0.0:
                self.y_std = 1.0
            y_n = (y - self.y_mean) / self.y_std
        else:
            self.y_mean = 0.0
            self.y_std = 1.0
            y_n = y
        self.y_train_n = y_n

        if optimize:
            def set_params(log_theta):
                self.kernel.lengthscales = np.exp(log_theta[:d])
                self.kernel.variance = float(np.exp(log_theta[d]))
                self.noise = float(np.exp(log_theta[d + 1]))

            xref = np.log(np.clip(np.concatenate([
                self.kernel.lengthscales, [self.kernel.variance, self.noise]
            ]), 1e-12, None))
            lb = ([np.log(self.LENGTHSCALE_BOUNDS[0])] * d
                  + [np.log(self.SIGNAL_VARIANCE_BOUNDS[0]),
                     np.log(self.NOISE_BOUNDS[0])])
            ub = ([np.log(self.LENGTHSCALE_BOUNDS[1])] * d
                  + [np.log(self.SIGNAL_VARIANCE_BOUNDS[1]),
                     np.log(self.NOISE_BOUNDS[1])])
            bounds = list(zip(lb, ub))

            def objective(log_theta):
                set_params(log_theta)
                return self._nll()

            best = None
            rng = np.random.RandomState(42)
            for it in range(4):  # 1 départ + 3 redémarrages aléatoires
                x0 = xref if it == 0 else np.clip(xref + rng.randn(d + 2) * 0.5,
                                                  min(lb), max(ub))
                res = minimize(objective, x0, method="L-BFGS-B", bounds=bounds)
                f = res.fun if np.isfinite(res.fun) else self._nll()
                if np.isfinite(f) and (best is None or f < best[0]):
                    best = (f, np.array(res.x))
            if best is not None:
                set_params(best[1])

        self._last_jitter = 0.0
        self._condition()
        return self

    # ------------------------------------------------------------------ #
    # prédictions
    # ------------------------------------------------------------------ #
    def predict(self, X, return_std=False):
        """Moyenne (et écart-type de f) a posteriori en des points X.

        Parameters
        ----------
        X : ndarray, shape (m, d)
        return_std : bool, default=False
            Si True, renvoie aussi l'écart-type de f (SANS le bruit).

        Returns
        -------
        mean : ndarray, shape (m,)
        std : ndarray, shape (m,), seulement si ``return_std=True``
        """
        if self._K is None:
            raise RuntimeError("fit() doit être appelé avant predict()")
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        k_star = self.kernel.K(X, self.x_train)  # (m, n)
        mean_n = k_star @ self._alpha            # (m,)
        # variance de f (sans bruit) : k(x*, x*) - k*^T K^{-1} k*
        Kinv_k = self._Kinv @ k_star.T           # (n, m)
        var_n = self.kernel.diag(X) - np.sum(k_star * Kinv_k.T, axis=1)
        mean = mean_n * self.y_std + self.y_mean
        if return_std:
            std = np.sqrt(np.maximum(var_n, 0.0)) * self.y_std
            return mean, std
        return mean

    # ------------------------------------------------------------------ #
    # leave-one-out
    # ------------------------------------------------------------------ #
    def loo(self):
        """Résidus leave-one-out en forme fermée (Rasmussen & Williams, eq. 5.12).

        Utilise [K^{-1}]_ii, diagonale de K^{-1} (K = k(X, X) + noise I), sans
        refaire n ajustements.

        Returns
        -------
        mean : ndarray, shape (n,)
            Moyennes prédictives leave-one-out de y_i, dans les unités de y.
        var : ndarray, shape (n,)
            Variances prédictives leave-one-out (bruit inclus), dans les unités de y.
        """
        if self._K is None:
            raise RuntimeError("fit() doit être appelé avant loo()")
        diag_inv = np.diag(self._Kinv)
        y_n = self.y_train_n
        mean_n = y_n - self._alpha / diag_inv
        var_n = 1.0 / diag_inv
        s = self.y_std
        return mean_n * s + self.y_mean, var_n * s * s
