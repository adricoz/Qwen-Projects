"""Régression par processus gaussien (GP) avec optimisation de la log-vraisemblance marginale."""
import numpy as np
from scipy.optimize import minimize


class GPRegressor:
    """Processus gaussien à moyenne nulle, bruit gaussien isotrope.

    Parameters
    ----------
    kernel : object
        Noyau avec `K(X1, X2)` et `diag(X)` (ex. `gp_lite.kernels.SquaredExponentialARD`).
    noise : float
        Variance a priori du bruit d'observation (optimisable).
    normalize_y : bool
        Si True, y est recentré/réduit (std, ddof = 0) avant conditionnement ;
        les prédictions sont renvoyées dans les unités de y.
    """

    def __init__(self, kernel, noise=1e-6, normalize_y=True):
        self.kernel = kernel
        self.noise = float(noise)
        self.normalize_y = bool(normalize_y)
        self._X = None
        self._yn = None        # y normalisé
        self._y_mean = 0.0
        self._y_std = 1.0
        self._L = None
        self._alpha = None

    # ------------------------------------------------------------------ #
    def fit(self, X, y, optimize=True):
        """
        Fit the GP on (X, y), optionally maximising the marginal likelihood.

        Parameters
        ----------
        X : ndarray, shape (n, d)
            Points d'apprentissage.
        y : ndarray, shape (n,)
            Observations.
        optimize : bool
            Si True, optimise log(lengthscales), log(variance), log(noise)
            par L-BFGS-B avec 3 redémarrages aléatoires (graine fixée).
            Si False, garde les hyperparamètres courants et conditionne seulement.

        Returns
        -------
        GPRegressor
            `self`.
        """
        X = np.atleast_2d(np.asarray(X, dtype=float))
        y = np.asarray(y, dtype=float).ravel()
        if X.shape[0] != y.shape[0]:
            raise ValueError("X et y doivent avoir le même nombre de lignes")

        if self.normalize_y:
            self._y_mean = float(y.mean())
            self._y_std = float(y.std())
            self._yn = (y - self._y_mean) / self._y_std
        else:
            self._y_mean = 0.0
            self._y_std = 1.0
            self._yn = y
        self._X = X

        if optimize:
            self._optimize(X, self._yn)
        self._cache(X, self._yn)
        return self

    # ------------------------------------------------------------------ #
    def _params(self):
        """Paramètres log : [log lengthscales..., log variance, log noise] (ndarray 1D)."""
        return np.concatenate([
            np.log(self.kernel.lengthscales),
            [np.log(self.kernel.variance)],
            [np.log(self.noise)],
        ])

    def _set_params(self, theta):
        """Reconstruit kernel/noise depuis les paramètres log."""
        n_ls = len(self.kernel.lengthscales)
        self.kernel.lengthscales = np.exp(theta[:n_ls])
        self.kernel.variance = float(np.exp(theta[n_ls]))
        self.noise = float(np.exp(theta[n_ls + 1]))

    def _log_likelihood(self, X, yn, jitter=0.0):
        """
        Negative log marginal likelihood (plus l'entropie) via Cholesky.

        Returns
        -------
        (float, bool)
            Valeur de -log L et drapeau de succès de la factorisation.
        """
        n = X.shape[0]
        K = self.kernel.K(X, X) + (self.noise + jitter) * np.eye(n)
        try:
            L = np.linalg.cholesky(K)
            w = np.linalg.solve(L, yn)          # y^T K^-1 y = w^T w
            logdet = 2.0 * np.sum(np.log(np.diag(L)))
            ll = -0.5 * w @ w - 0.5 * logdet - 0.5 * n * np.log(2.0 * np.pi)
        except np.linalg.LinAlgError:
            return np.inf, False
        return -ll, True

    def _neg_lml(self, x, X, yn):
        """Fonction objectif pour `scipy.optimize.minimize` (avec jitter progressif)."""
        self._set_params(x)
        jitter = 0.0
        for _ in range(8):
            val, ok = self._log_likelihood(X, yn, jitter)
            if ok:
                return val
            jitter = 1e-12 if jitter == 0.0 else jitter * 10.0
        return 1e12

    def _optimize(self, X, yn):
        """Optimisation ML avec 3 redémarrages aléatoires (graine fixée) + point initial."""
        rng = np.random.default_rng(0)
        best = None
        n_ls = len(self.kernel.lengthscales)
        # Point de départ : hyperparamètres actuels
        starts = [self._params()]
        for _ in range(3):  # redémarrages aléatoires bornés
            starts.append(np.concatenate([
                rng.uniform(-2.0, 1.0, n_ls),
                rng.uniform(-2.0, 1.0, 1),
                rng.uniform(-8.0, -1.0, 1),
            ]))
        for s in starts:
            res = minimize(self._neg_lml, s, args=(X, yn), method="L-BFGS-B")
            if res.x is None:
                continue
            self._set_params(res.x)
            val, ok = self._log_likelihood(X, yn)
            if not ok:
                val = np.inf
            if best is None or val < best[0]:
                best = (val, res.x.copy())
        if best is not None:
            self._set_params(best[1])

    # ------------------------------------------------------------------ #
    def _K_with_jitter(self, X, yn, base_jitter=1e-10):
        """Cholesky de K + noise*I avec jitter progressif si besoin ; stocke L, alpha."""
        n = X.shape[0]
        K = self.kernel.K(X, X) + self.noise * np.eye(n)
        jitter = base_jitter
        for _ in range(12):
            try:
                L = np.linalg.cholesky(K + jitter * np.eye(n))
                w = np.linalg.solve(L, yn)
                alpha = np.linalg.solve(L.T, w)   # alpha = K^-1 y
                self._L = L
                self._alpha = alpha
                return
            except np.linalg.LinAlgError:
                jitter *= 100.0
        # Dernier recours : décomposition générique
        K += 1e-8 * np.eye(n)
        L = np.linalg.cholesky(K)
        w = np.linalg.solve(L, yn)
        alpha = np.linalg.solve(L.T, w)
        self._L = L
        self._alpha = alpha

    def _cache(self, X, yn):
        """Factorise K + noise*I et précalcule alpha = K^-1 y (sur y_n)."""
        self._K_with_jitter(X, yn)

    # ------------------------------------------------------------------ #
    def predict(self, X, return_std=False):
        """
        Posterior mean and (optional) standard deviation of f at X.

        Parameters
        ----------
        X : ndarray, shape (m, d)
            Points de prédiction.
        return_std : bool
            Si True, renvoie aussi l'écart-type de f (sans le bruit).

        Returns
        -------
        mean : ndarray, shape (m,)
            Moyenne prédictive dans les unités de y.
        std : ndarray, shape (m,)
            Écart-type de f (uniquement si `return_std=True`).
        """
        if self._L is None:
            raise RuntimeError("Appelez d'abord fit()")
        X = np.atleast_2d(np.asarray(X, dtype=float))
        Ks = self.kernel.K(X, self._X)
        v = np.linalg.solve(np.triu(self._L.T).T, Ks.T)   # L^-1 Ks^T
        mean = Ks @ self._alpha
        if return_std:
            var_f = np.maximum(self.kernel.diag(X) - np.sum(v * v, axis=0), 0.0)
            std = np.sqrt(var_f) * self._y_std
            return mean * self._y_std + self._y_mean, std
        return mean * self._y_std + self._y_mean

    # ------------------------------------------------------------------ #
    def log_marginal_likelihood(self):
        """
        Marginal log-likelihood at the current hyperparameters.

        Returns
        -------
        float
            log p(y | X, theta), sur y_n si `normalize_y=True`.
        """
        if self._X is None:
            raise RuntimeError("Appelez d'abord fit()")
        val, ok = self._log_likelihood(self._X, self._yn)
        if not ok:
            raise RuntimeError("La matrice de covariance n'est pas définie positive")
        return -val

    # ------------------------------------------------------------------ #
    def loo(self):
        """
        Leave-one-out posterior moments (Rasmussen & Williams, section 5.4.2).

        Returns
        -------
        (mean, var) : tuple of ndarray, shape (n,)
            Moyenne et variance prédictives leave-one-out de y_i (bruit inclus),
            dans les unités de y, avec la normalisation de tout le jeu d'apprentissage.
        """
        if self._L is None:
            raise RuntimeError("Appelez d'abord fit()")
        L = self._L
        alpha = self._alpha
        yn = self._yn
        # b = (K + noise*I)^-1 ; S_i = 1/b_ii (complément de Schur)
        # mu_loo_i = y_i - S_i * alpha_i ;  var_loo_i = S_i
        invK = np.linalg.solve(L.T, np.linalg.solve(L, np.eye(L.shape[0])))
        S = 1.0 / np.diag(invK)
        mean_n = yn - S * alpha
        var_n = S
        return mean_n * self._y_std + self._y_mean, var_n * self._y_std ** 2
