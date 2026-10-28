"""Regression par processus gaussien avec hyperparametres appris en espace log."""
from __future__ import annotations

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize

from .kernels import _ARDKernel

# valeur renvoyee quand la factorisation de Cholesky echoue
_NLL_FAILURE = 1e10
# jitters relatifs sur la diagonale essaies quand Cholesky echoue
_JITTERS = (0.0, 1e-10, 1e-8, 1e-6)


def _nll_and_alpha(K, noise, y, jitter):
    """Negative log-likelihood gaussienne et vecteur alpha = K_full^{-1} y.

    Parameters
    ----------
    K : np.ndarray of shape (n, n)
        Matrice de covariance du signal (sans bruit).
    noise : float
        Variance du bruit additif.
    y : np.ndarray of shape (n,)
        Observations (eventuellement normalisees).
    jitter : float
        Jitter absolu ajoute a la diagonale avant factorisation.

    Returns
    -------
    tuple of (float, np.ndarray)
        (NLL, alpha). NLL = 1e10 si la factorisation echoue pour tous les jitters.
    """
    n = len(y)
    for jit in (jitter, *_JITTERS):
        try:
            c, low = cho_factor(K + (noise + jit) * np.eye(n), check_finite=False)
            alpha = cho_solve((c, low), y, check_finite=False)
            logdet = 2.0 * np.sum(np.log(np.diag(c)))
            nll = 0.5 * float(y @ alpha) + 0.5 * logdet + 0.5 * n * np.log(2.0 * np.pi)
            return float(nll), alpha
        except np.linalg.LinAlgError:
            continue
    return _NLL_FAILURE, None


class GPRegressor:
    """Processus gaussien a moyenne nulle avec noyau ARD et bruit gaussien.

    Parameters
    ----------
    kernel : _ARDKernel
        Noyau (SquaredExponentialARD, Matern32ARD ou Matern52ARD).
    noise : float, optional
        Variance initiale du bruit d'observation.
    normalize_y : bool, optional
        Si True, y est centre/reduit avant l'optimisation et les predictions
        sont ramenees dans l'echelle d'origine (comme scikit-learn).
    seed : int, optional
        Graine des redemarrages aleatoires de l'optimisation.
    """

    def __init__(self, kernel, noise: float = 1e-6, normalize_y: bool = True,
                 seed: int = 0):
        self.kernel = kernel
        self.noise = float(noise)
        self.normalize_y = bool(normalize_y)
        self.seed = int(seed)
        self._L_ = None

    # ------------------------------------------------------------------
    @staticmethod
    def _initial_lengthscales(kernel, d: int) -> np.ndarray:
        """Echelles initiales: celles du noyau, ou 1.0 si incompatibles."""
        l = np.atleast_1d(kernel.lengthscales)
        if l.size == d:
            return l.copy()
        if l.size == 1:
            return np.full(d, l[0])
        return np.ones(d)

    def _set_theta(self, log_params) -> None:
        """Reparametrage log -> (lengthscales, variance, noise), tous > 0."""
        vals = np.exp(np.asarray(log_params, dtype=float))
        self.kernel.lengthscales = vals[:-2].copy()
        self.kernel.variance = float(vals[-2])
        self.noise = float(vals[-1])

    def _nll(self, log_params, X, y) -> float:
        """Negative log-vraisemblance marginale (fonction optimisee)."""
        self._set_theta(log_params)
        K = self.kernel.K(X, X)
        nll, _ = _nll_and_alpha(K, self.noise, y, 0.0)
        return nll

    def fit(self, X, y):
        """Apprend les hyperparametres par maximisation de la vraisemblance marginale.

        Optimisation L-BFGS-B en espace log de [l_1..l_d, variance, noise]
        avec 3 redemarrages aleatoires (graine fixee); le meilleur point est
        conserve et le conditionnement final est re-calculé avec un jitter leger.

        Parameters
        ----------
        X : array-like of shape (n, d)
            Points d'apprentissage.
        y : array-like of shape (n,)
            Observations.

        Returns
        -------
        self
            Le modele conditionne.
        """
        self.X = np.atleast_2d(np.asarray(X, dtype=float))
        y = np.asarray(y, dtype=float).ravel()
        if self.X.shape[0] != len(y):
            raise ValueError("X and y have inconsistent numbers of samples")
        self.y = y
        n, d = self.X.shape
        # normalisation des sorties (comme scikit-learn)
        self.y_mean_ = float(y.mean())
        self.y_std_ = float(y.std()) if self.normalize_y and y.std() > 0 else 1.0
        y_fit = (y - self.y_mean_) / self.y_std_

        # bornes en espace log (les entrees sont supposees de l'ordre de 1)
        dmin, dmax = -8.0, 8.0
        bounds = [(dmin, dmax)] * (d + 2)

        start0 = np.log(np.concatenate([
            self._initial_lengthscales(self.kernel, d),
            [self.kernel.variance, max(self.noise, 1e-12)],
        ]))
        rng = np.random.default_rng(self.seed)
        best = (np.inf, None)
        for i in range(3):
            x0 = start0 if i == 0 else start0 + rng.normal(0.0, 0.5, size=d + 2)
            res = minimize(self._nll, x0, args=(self.X, y_fit), method="L-BFGS-B",
                           bounds=bounds, options=dict(maxiter=200))
            if res.fun < best[0]:
                best = (res.fun, np.array(res.x, dtype=float))
        self.theta_log_ = best[1]

        # conditionnement final (jitter leger pour la stabilite numerique)
        self._set_theta(self.theta_log_)
        self.K_ = self.kernel.K(self.X, self.X)
        nll, alpha = _nll_and_alpha(self.K_, self.noise, y_fit, 1e-10)
        if alpha is None:
            raise np.linalg.LinAlgError("factorisation de Cholesky impossible")
        self.alpha_ = alpha
        self._L_ = None
        return self

    def _L(self):
        """Couple (c, low) de cho_factor de K + noise*I, mis en cache."""
        if self._L_ is None:
            self._L_ = cho_factor(self.K_ + self.noise * np.eye(len(self.K_)),
                                  check_finite=False)
        return self._L_

    def log_marginal_likelihood(self) -> float:
        """Log-vraisemblance marginale a l'optimum: log p(y | X, theta_hat).

        Returns
        -------
        float
            Log-vraisemblance marginale (n'importe laquelle echelle de y).
        """
        if not hasattr(self, "alpha_"):
            raise RuntimeError("fit() must be called before log_marginal_likelihood()")
        y_fit = (self.y - self.y_mean_) / self.y_std_
        nll, _ = _nll_and_alpha(self.K_, self.noise, y_fit, 0.0)
        return -nll

    def predict(self, X, return_std: bool = False):
        """Prediction a posteriori: moyenne, et eventuellement ecart-type.

        Parameters
        ----------
        X : array-like of shape (m, d)
            Points de prediction.
        return_std : bool, optional
            Si True, renvoie aussi l'ecart-type a posteriori.

        Returns
        -------
        y_mean : np.ndarray of shape (m,)
            Moyenne a posteriori (dans l'echelle d'origine de y).
        y_std : np.ndarray of shape (m,), optional
            Ecart-type a posteriori (si return_std=True).
        """
        if not hasattr(self, "alpha_"):
            raise RuntimeError("fit() must be called before predict()")
        X = np.atleast_2d(np.asarray(X, dtype=float))
        K_train_test = self.kernel.K(self.X, X)
        k_test = self.kernel.diag(X)
        f_mean = self.alpha_ @ K_train_test
        v = cho_solve(self._L(), K_train_test.T, check_finite=False)
        f_var = np.maximum(k_test - np.einsum("ij,ij->j", K_train_test, v), 0.0)
        y_mean = self.y_mean_ + f_mean * self.y_std_
        if return_std:
            return y_mean, np.sqrt(f_var) * self.y_std_
        return y_mean

    def loo(self) -> np.ndarray:
        """Residus leave-one-out en forme fermee (Rasmussen & Williams, 5.4.2).

        Avec K = covariance du signal, A = K + noise*I et alpha = A^{-1} y:
            mu_i^{(-i)} = r_i + v_ii (y_i - r_i),
            e_i = y_i - mu_i^{(-i)} = (1 - v_ii) (y_i - r_i)
        ou r = K alpha et v_ii = [A^{-1} K]_ii.

        Returns
        -------
        np.ndarray of shape (n,)
            Residus e_i dans l'echelle d'origine de y.
        """
        if not hasattr(self, "alpha_"):
            raise RuntimeError("fit() must be called before loo()")
        A = self.K_ + self.noise * np.eye(len(self.K_))
        c, low = cho_factor(A, check_finite=False)
        r = self.K_ @ self.alpha_
        v_diag = np.diag(cho_solve((c, low), self.K_, check_finite=False))
        y_fit = (self.y - self.y_mean_) / self.y_std_
        return (1.0 - v_diag) * (y_fit - r) * self.y_std_
