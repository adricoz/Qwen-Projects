"""Stationnary ARD kernels (NumPy only) used by :mod:`gp_lite.gp`.

Conventions: inputs are arrays of shape ``(n, d)``, lengthscales are arrays of
shape ``(d,)``. Matern kernels follow Rasmussen & Williams eq. (4.25):
``r = sqrt(sum(((x - x') / l) ** 2))``.
"""
import numpy as np


class _ARDKernel:
    """Base class of the lengthscale-ARD kernels."""

    def __init__(self, lengthscales, variance=1.0):
        """
        Parameters
        ----------
        lengthscales : array_like of shape (d,)
            One lengthscale per input dimension.
        variance : float, optional
            Signal variance (kernel diagonal at zero distance).
        """
        self.lengthscales = np.atleast_1d(np.asarray(lengthscales, dtype=float)).copy()
        self.variance = float(variance)

    # -- internal helpers -------------------------------------------------
    def _scaled_dist2(self, X1, X2):
        """Return ``r**2 = sum_d (((x1 - x2) / l_d) ** 2)`` of shape (n1, n2)."""
        X1 = np.atleast_2d(X1)
        X2 = np.atleast_2d(X2)
        diff = (X1[:, None, :] - X2[None, :, :]) / self.lengthscales
        return np.sum(diff ** 2, axis=2)

    def _covariance(self, r2):
        """Kernel value as a function of the scaled squared distance."""
        raise NotImplementedError

    # -- public API ---------------------------------------------------------
    def K(self, X1, X2):
        """
        Covariance matrix between two sets of points.

        Parameters
        ----------
        X1 : array_like, shape (n1, d)
        X2 : array_like, shape (n2, d)

        Returns
        -------
        ndarray, shape (n1, n2)
        """
        return self._covariance(self._scaled_dist2(X1, X2))

    def diag(self, X):
        """
        Diagonal of ``K(X, X)``.

        Parameters
        ----------
        X : array_like, shape (n, d)

        Returns
        -------
        ndarray, shape (n,)
        """
        n = np.atleast_2d(X).shape[0]
        return np.full(n, self.variance, dtype=float)


class SquaredExponentialARD(_ARDKernel):
    """Squared-exponential (RBF) kernel with per-dimension lengthscales."""

    def _covariance(self, r2):
        return self.variance * np.exp(-0.5 * r2)


class Matern32ARD(_ARDKernel):
    """Matern 3/2 kernel with per-dimension lengthscales.

    ``K(r) = sigma**2 * (1 + sqrt(3) r) * exp(-sqrt(3) r)``.
    """

    def _covariance(self, r2):
        r = np.sqrt(r2)
        return self.variance * (1.0 + np.sqrt(3.0) * r) * np.exp(-np.sqrt(3.0) * r)


class Matern52ARD(_ARDKernel):
    """Matern 5/2 kernel with per-dimension lengthscales.

    ``K(r) = sigma**2 * (1 + sqrt(5) r + 5 r**2 / 3) * exp(-sqrt(5) r)``.
    """

    def _covariance(self, r2):
        r = np.sqrt(r2)
        a = np.sqrt(5.0)
        return self.variance * (1.0 + a * r + 5.0 * r2 / 3.0) * np.exp(-a * r)
