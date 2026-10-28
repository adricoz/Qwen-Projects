"""Régresseur par processus gaussien (gp_lite/gp.py).

Modèle : f ~ GP(0, k), y = f + eps, eps ~ N(0, sigma_n^2).
Toutes les formules sont celles de Rasmussen & Williams (2005), chapitres 2 et 5.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import cho_factor, cho_solve, cholesky
from scipy.optimize import minimize

from .kernels import _ARDKernel

__all__ = ["GPRegressor"]

# bornes de recherche des hyperparamètres, en espace logarithmique
# (valeur par défaut, assumant des entrées ~ [0, 1] et un y normalisé)
LOG_BOUNDS = {
    "lengthscales": (-5.0, 5.0),
    "variance": (-5.0, 5.0),
    "noise": (-14.0, 0.0),
}
NLL_FAILURE = 1e10


def _factorize_positive_definite(C, max_jitter=1e-6):
    """Cholesky (facteur bas) de C, en ajoutant un jitter si la factorisation échoue."""
    n = C.shape[0]
    for jitter in (0.0, 1e-10, 1e-8, 1e-6):
        try:
            return cholesky(C + jitter * np.eye(n), lower=True, check_finite=False)
        except np.linalg.LinAlgError:
            continue
    raise np.linalg.LinAlgError("cannot factorize the covariance matrix (all jitters failed)")


class GPRegressor:
    """Régression par processus gaussien à noyau ARD.

    Parameters
    ----------
    kernel : _ARDKernel
        Noyau (``lengthscales`` de forme (d,), ``variance`` scalaire).
    noise : float, default 1e-6
        Variance du bruit d'observation.
    normalize_y : bool, default True
        Si True, y est centré-réduit (np.std, ddof=0) avant le conditionnement ;
        la vraisemblance et le bruit sont exprimés sur y_n, et les prédictions
        sont renvoyées dans les unités d'origine.

    Examples
    --------
    >>> from gp_lite import GPRegressor, SquaredExponentialARD
    >>> gp = GPRegressor(SquaredExponentialARD([0.5, 0.5]), noise=1e-3)
    >>> gp.fit(X, y).predict(Xtest)
    """

    def __init__(self, kernel, noise=1e-6, normalize_y=True):
        if not isinstance(kernel, _ARDKernel):
            raise TypeError("kernel must be a gp_lite kernel")
        self.kernel = kernel
        self.noise = float(noise)
        self.normalize_y = bool(normalize_y)
        self._x_train = None
        self._y_train = None   # y brut (unités d'origine)
        self._y_n = None       # cible d'entraînement (normalisée ou brute)
        self._y_mean = 0.0
        self._y_std = 1.0
        self._alpha = None     # (K + noise I)^-1 y_n
        self._L = None         # facteur de Cholesky de (K + noise I)

    # ------------------------------------------------------------------
    # vraisemblance
    # ------------------------------------------------------------------
    @staticmethod
    def _build_K(kernel, lengthscales, variance, X):
        """Matrice de covariance avec les hyperparamètres fournis (sans toucher au noyau)."""
        l0 = kernel.lengthscales
        v0 = kernel.variance
        kernel.lengthscales = lengthscales
        kernel.variance = float(variance)
        try:
            K = kernel.K(X, X)
        finally:
            kernel.lengthscales = l0
            kernel.variance = v0
        return K

    def _negative_log_likelihood(self, log_params, y_n, X, kernel):
        """Negative log marginal likelihood of y_n for the given (log) parameters."""
        d = self.kernel.lengthscales.shape[0]
        lengthscales = np.exp(log_params[:d])
        variance = float(np.exp(log_params[d]))
        noise = float(np.exp(log_params[d + 1]))
        n = y_n.shape[0]

        C = self._build_K(kernel, lengthscales, variance, X) + noise * np.eye(n)
        try:
            L = _factorize_positive_definite(C)
        except np.linalg.LinAlgError:
            return NLL_FAILURE

        alpha = cho_solve((L, True), y_n)
        nll = (0.5 * float(y_n @ alpha) + np.sum(np.log(np.diag(L)))
               + 0.5 * n * np.log(2.0 * np.pi))
        return float(nll)

    def log_marginal_likelihood(self):
        """Log marginal likelihood (sur y_n si normalisé) aux hyperparamètres courants."""
        if self._y_n is None:
            raise RuntimeError("fit() must be called before log_marginal_likelihood()")
        log_params = np.concatenate((
            np.log(self.kernel.lengthscales),
            [np.log(self.kernel.variance), np.log(self.noise)],
        ))
        nll = self._negative_log_likelihood(log_params, self._y_n, self._x_train, self.kernel)
        if nll >= NLL_FAILURE / 2:
            raise np.linalg.LinAlgError(
                "cannot evaluate the log marginal likelihood (Cholesky failed)")
        return -nll

    # ------------------------------------------------------------------
    # entraînement
    # ------------------------------------------------------------------
    def fit(self, X, y, optimize=True):
        """Ajuster le GP sur (X, y) et retourner self.

        Parameters
        ----------
        X : array (n, d)
            Points d'entraînement.
        y : array (n,)
            Cibles.
        optimize : bool, default True
            Si True, optimiser log(lengthscales), log(variance), log(noise) par
            L-BFGS-B (3 redémarrages, graine fixée, conservation du meilleur).
            Si False, conserver les hyperparamètres actuels.
        """
        X = np.atleast_2d(np.asarray(X, dtype=float))
        y = np.asarray(y, dtype=float).ravel()
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of points")
        self._x_train = X
        self._y_train = y

        # normalisation de y (statistiques du jeu complet, np.std ddof=0).
        # Si normalize_y=False, on travaille directement en unités de y
        # (échelle identité, pour que predict/loo ne ré-échelonnent pas).
        if self.normalize_y:
            self._y_mean = float(np.mean(y))
            std = float(np.std(y))
            self._y_std = std if std > 0 else 1.0
            self._y_n = (y - self._y_mean) / self._y_std
        else:
            self._y_mean = 0.0
            self._y_std = 1.0
            self._y_n = y.copy()

        if optimize:
            self._optimize_hyperparameters()

        self._condition()
        return self

    def _optimize_hyperparameters(self):
        """L-BFGS-B sur la négative log-vraisemblance, 3 redémarrages, meilleur gardé."""
        d = self.kernel.lengthscales.shape[0]
        lo_l, hi_l = LOG_BOUNDS["lengthscales"]
        lo_v, hi_v = LOG_BOUNDS["variance"]
        lo_n, hi_n = LOG_BOUNDS["noise"]
        bounds = ([(lo_l, hi_l)] * d + [(lo_v, hi_v)] + [(lo_n, hi_n)])

        rng = np.random.default_rng(42)  # graine fixée : reproductible
        starts = [np.concatenate((
            np.log(self.kernel.lengthscales),
            [np.log(self.kernel.variance), np.log(self.noise)],
        ))]
        for _ in range(2):  # 2 redémarrages aléatoires en plus du point courant
            x0 = np.concatenate((
                rng.uniform(-2.0, 2.0, d),
                [rng.uniform(-1.0, 1.0), rng.uniform(-9.0, -3.0)],
            ))
            starts.append(x0)

        best = None
        for x0 in starts:
            res = minimize(
                self._negative_log_likelihood,
                x0, args=(self._y_n, self._x_train, self.kernel),
                method="L-BFGS-B", bounds=bounds,
                options={"maxiter": 300, "gtol": 1e-6},
            )
            nll = res.fun
            if best is None or nll < best[0] - 1e-12:
                best = (nll, res.x)
        if best is None:
            raise RuntimeError("hyperparameter optimization failed (Cholesky everywhere)")

        log_params = best[1]
        self.kernel.lengthscales = np.exp(log_params[:d]).copy()
        self.kernel.variance = float(np.exp(log_params[d]))
        self.noise = float(np.exp(log_params[d + 1]))

    def _condition(self):
        """Conditionner sur les données : L de Cholesky et alpha = C^-1 y_n."""
        n = self._y_n.shape[0]
        C = self.kernel.K(self._x_train, self._x_train) + self.noise * np.eye(n)
        L = _factorize_positive_definite(C)
        self._L = L
        self._alpha = cho_solve((L, True), self._y_n)

    # ------------------------------------------------------------------
    # prédiction
    # ------------------------------------------------------------------
    def predict(self, X, return_std=False):
        """Prédiction moyenne (et écart-type de f, sans le bruit) aux points X.

        Parameters
        ----------
        X : array (m, d)
            Points de prédiction.
        return_std : bool, default False
            Si True, retourner le couple (mean, std).

        Returns
        -------
        mean : array (m,)
            Moyenne a posteriori, dans les unités de y.
        std : array (m,), si return_std
            Écart-type de f (sans le bruit), dans les unités de y.
        """
        if self._alpha is None:
            raise RuntimeError("fit() must be called before predict()")
        X = np.atleast_2d(np.asarray(X, dtype=float))
        k_xs = self.kernel.K(X, self._x_train)            # (m, n)
        mean_n = k_xs @ self._alpha
        mean = mean_n * self._y_std + self._y_mean

        if not return_std:
            return mean

        # var(f(x)|D) = k(x,x) - k(x)^T C^-1 k(x)  (diagonale seulement)
        w = cho_solve((self._L, True), k_xs.T)            # (n, m)
        v_n = self.kernel.variance - np.einsum("mi,im->m", k_xs, w)
        std = np.sqrt(np.maximum(v_n, 0.0)) * self._y_std
        return mean, std

    # ------------------------------------------------------------------
    # leave-one-out
    # ------------------------------------------------------------------
    def loo(self):
        """Résidus leave-one-out en forme fermée (Rasmussen & Williams, section 5.4.2).

        Pour chaque point d'entraînement i, moyenne et variance prédictives de y_i
        sachant y_{-i} (bruit inclus), aux hyperparamètres courants, en unités de y.
        Calculé sans refaire n ajustements :

            A = C^-1, b = A y_n
            v_i = 1 / A_ii
            m_i = y_i - b_i / A_ii

        Returns
        -------
        mean : array (n,)
            Moyenne leave-one-out de y_i (unités de y).
        var : array (n,)
            Variance prédictive de y_i (bruit inclus, unités de y).
        """
        if self._alpha is None:
            raise RuntimeError("fit() must be called before loo()")
        n = self._x_train.shape[0]
        C = self.kernel.K(self._x_train, self._x_train) + self.noise * np.eye(n)
        L = cholesky(C, lower=True, check_finite=False)
        A = cho_solve((L, True), np.eye(n))                # A = C^-1, (n, n)
        y_n = self._y_n

        Aii = np.diag(A)
        b = A @ y_n
        mean_n = y_n - b / Aii
        var_n = 1.0 / Aii

        mean = mean_n * self._y_std + self._y_mean
        var = var_n * self._y_std ** 2
        return mean, var
