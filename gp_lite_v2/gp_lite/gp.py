"""Regression par processus gaussien (moyenne constante nulle, y normalise)."""
import numpy as np
from scipy.linalg import cho_solve
from scipy.optimize import minimize


def _safe_cholesky(A):
    """Cholesky avec jitter diagonal croissant si la factorisation echoue."""
    eye = np.eye(A.shape[0])
    for j in (0.0, 1e-12, 1e-10, 1e-8, 1e-6):
        try:
            return np.linalg.cholesky(A + j * eye if j > 0 else A), True
        except np.linalg.LinAlgError:
            continue
    return None, False


class GPRegressor:
    """Regressiveur a processus gaussien (y centre/reduit si ``normalize_y``).

    Parameters
    ----------
    kernel : gp_lite.kernels._ARDKernel
        Noyau (doit exposer ``K(X1, X2)``).
    noise : float
        Bruit de mesure initial (optimise en log dans ``fit``).
    normalize_y : bool
        Si vrai, ``fit`` centre/reduit ``y`` et ``predict`` renvoie en unites brutes.
    """

    def __init__(self, kernel, noise=1e-6, normalize_y=True):
        self.kernel = kernel
        self.noise = float(noise)
        self.normalize_y = bool(normalize_y)
        self._fitted = False

    # ------------------------------------------------------------------
    def _condition(self, X, y_n):
        """Cache cholesky et ``alpha = (K + sigma^2 I)^-1 y_n``."""
        n = X.shape[0]
        K = self.kernel.K(X, X) + self.noise * np.eye(n)
        L, ok = _safe_cholesky(K)
        if not ok:
            raise RuntimeError("Cholesky failure")
        self._L = L
        self._alpha = cho_solve((L, True), y_n)
        self._fitted = True

    def _set_from_log(self, log_theta):
        theta = np.exp(log_theta)
        self.kernel.lengthscales = theta[:-2]
        self.kernel.variance = float(theta[-2])
        self.noise = float(theta[-1])

    def _nll_only(self, log_theta):
        """Value of the negative log-marginal likelihood at ``log_theta``."""
        self._set_from_log(log_theta)
        X = self.X_train
        K = self.kernel.K(X, X) + self.noise * np.eye(len(X))
        L, ok = _safe_cholesky(K)
        if not ok:
            return 1e10
        alpha = cho_solve((L, True), self._y_train_n)
        nll = (0.5 * float(self._y_train_n @ alpha)
               + np.log(np.diag(L)).sum()
               + 0.5 * len(X) * np.log(2.0 * np.pi))
        if np.isfinite(nll):
            return n(nll):
            return nll):
            return n np.isfinite(n
