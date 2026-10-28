"""Kernels (covariance functions) for the `gp_lite` package.

Conventions
-----------
* Inputs are 2-D arrays of shape (n, d) (a 1-D array of length d is treated
  as a single point).
* Matérn distance: ``r = sqrt(sum(((x - x') / l)**2))`` with per-dimension
  (ARD) lengthscales ``l``.
* Matérn 5/2:  ``K = variance * (1 + sqrt(5) r + 5 r**2 / 3) * exp(-sqrt(5) r)``
"""
import numpy as np


class BaseARD:
    """Base class for stationary kernels with ARD lengthscales.

    Parameters
    ----------
    lengthscales : float or array_like
        ARD lengthscale(s). A scalar is broadcast to all dimensions.
    variance : float
        Signal variance (kernel value at zero distance).
    """

    def __init__(self, lengthscales, variance):
        lengthscales = np.atleast_1d(np.asarray(lengthscales, dtype=float))
        if np.any(lengthscales <= 0):
            raise ValueError("lengthscales must be strictly positive")
        if variance <= 0:
            raise ValueError("variance must be strictly positive")
        self.lengthscales = lengthscales
        self.variance = float(variance)

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _broadcast(self, X):
        """Return ``X`` as a 2-D float array (n, d)."""
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X[None, :]
        d = self.lengthscales.shape[0]
        if X.shape[1] != d:
            raise ValueError(
                f"expected {d} input dimensions, got {X.shape[1]}")
        return X

    def _ard_r(self, X1, X2):
        """ARD distance ``r = sqrt(sum(((x - x') / l)**2))``.

        Parameters
        ----------
        X1, X2 : ndarray, shape (n1, d), (n2, d)

        Returns
        -------
        ndarray, shape (n1, n2)
        """
        z1 = X1[:, None, :] / self.lengthscales
        z2 = X2[None, :, :] / self.lengthscales
        return np.sqrt(np.sum((z1 - z2) ** 2, axis=2))

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def K(self, X1, X2):
        """Covariance matrix between two sets of points.

        Parameters
        ----------
        X1 : ndarray, shape (n1, d)
        X2 : ndarray, shape (n2, d)

        Returns
        -------
        ndarray, shape (n1, n2)
        """
        raise NotImplementedError

    def diag(self, X):
        """Diagonal of ``K(X, X)`` (kernel value at zero distance).

        Parameters
        ----------
        X : ndarray, shape (n, d)

        Returns
        -------
        ndarray, shape (n,)
        """
        X = self._broadcast(X)
        return np.full(X.shape[0], self.variance)

    def __repr__(self):
        return (f"{type(self).__name__}(lengthscales={self.lengthscales!r}, "
                f"variance={self.variance!r})")


class SquaredExponentialARD(BaseARD):
    """Squared exponential (RBF) kernel with ARD lengthscales.

    ``K(x, x') = variance * exp(-0.5 * sum(((x - x') / l)**2))``
    """

    def K(self, X1, X2):
        """Covariance matrix between two sets of points (see :meth:`BaseARD.K`)."""
        X1 = self._broadcast(X1)
        X2 = self._broadcast(X2)
        r = self._ard_r(X1, X2)
        return self.variance * np.exp(-0.5 * r * r)


class Matern32ARD(BaseARD):
    """Matérn 3/2 kernel with ARD lengthscales.

    ``K(x, x') = variance * (1 + sqrt(3) r) * exp(-sqrt(3) r)``,
    with ``r = sqrt(sum(((x - x') / l)**2))``.
    """

    GAMMA = np.sqrt(3.0)

    def K(self, X1, X2):
        """Covariance matrix between two sets of points (see :meth:`BaseARD.K`)."""
        X1 = self._broadcast(X1)
        X2 = self._broadcast(X2)
        r = self._ard_r(X1, X2)
        g = self.GAMMA * r
        return self.variance * (1.0 + g) * np.exp(-g)


class Matern52ARD(BaseARD):
    """Matérn 5/2 kernel with ARD lengthscales.

    ``K(x, x') = variance * (1 + sqrt(5) r + 5 r**2 / 3) * exp(-sqrt(5) r)``,
    with ``r = sqrt(sum(((x - x') / l)**2))``.
    """

    GAMMA = np.sqrt(5.0)

    def K(self, X1, X2):
        """Covariance matrix between two sets of points (see :meth:`BaseARD.K`)."""
        X1 = self._broadcast(X1)
        X2 = self._broadcast(X2)
        r = self._ard_r(X1, X2)
        g = self.GAMMA * r
        return self.variance * (1.0 + g + 5.0 * r * r / 3.0) * np.exp(-g)
