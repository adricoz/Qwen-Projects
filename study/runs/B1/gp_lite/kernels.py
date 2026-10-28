"""Stationary ARD kernels for gp_lite (NumPy only).

Conventions :
- inputs X1, X2 are 2-D arrays of shape (n, d) (a 1-D row is also accepted);
- anisotropic distance r = sqrt(sum_j ((x_j - x'_j) / l_j)^2);
- SquaredExponential : K = sigma^2 exp(-r^2 / 2);
- Matern 3/2         : K = sigma^2 (1 + sqrt(3) r) exp(-sqrt(3) r);
- Matern 5/2         : K = sigma^2 (1 + sqrt(5) r + 5 r^2 / 3) exp(-sqrt(5) r).
"""
import numpy as np


class _ARDKernel:
    """Base class of the ARD kernels.

    Parameters
    ----------
    lengthscales : array_like, shape (d,)
        Anisotropic lengthscale of each input dimension.
    variance : float, optional
        Signal variance (default 1.0).
    """

    def __init__(self, lengthscales, variance=1.0):
        self.lengthscales = np.atleast_1d(np.asarray(lengthscales, dtype=float))
        self.variance = float(variance)

    # -- internal helpers ----------------------------------------------------
    @staticmethod
    def _as2d(X):
        """Accept a single point given as a 1-D row."""
        return np.atleast_2d(np.asarray(X, dtype=float))

    def _r(self, X1, X2):
        """Anisotropic Euclidean distance matrix (n1, n2)."""
        X1 = self._as2d(X1)
        X2 = self._as2d(X2)
        z = X1[:, None, :] / self.lengthscales - X2[None, :, :] / self.lengthscales
        return np.sqrt(np.sum(z * z, axis=-1))

    # -- public API ------------------------------------------------------------
    def K(self, X1, X2):
        """Covariance matrix between X1 (n1, d) and X2 (n2, d), shape (n1, n2)."""
        raise NotImplementedError

    def diag(self, X):
        """Diagonal of K(X, X); for a stationary kernel it is the variance."""
        X = self._as2d(X)
        return np.full(X.shape[0], self.variance)


class SquaredExponentialARD(_ARDKernel):
    """Squared exponential (RBF) kernel with ARD lengthscales."""

    def K(self, X1, X2):
        r2 = self._r(X1, X2) ** 2
        return self.variance * np.exp(-0.5 * r2)


class Matern32ARD(_ARDKernel):
    """Matern 3/2 kernel with ARD lengthscales."""

    def K(self, X1, X2):
        r = self._r(X1, X2)
        return self.variance * (1.0 + np.sqrt(3.0) * r) * np.exp(-np.sqrt(3.0) * r)


class Matern52ARD(_ARDKernel):
    """Matern 5/2 kernel with ARD lengthscales."""

    def K(self, X1, X2):
        r = self._r(X1, X2)
        s5 = np.sqrt(5.0)
        return self.variance * (1.0 + s5 * r + 5.0 * r * r / 3.0) * np.exp(-s5 * r)
