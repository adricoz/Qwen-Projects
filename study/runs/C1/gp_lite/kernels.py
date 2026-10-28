"""Stationary covariance kernels for the ``gp_lite`` package.

Tous les noyaux partagent la meme convention Matern :
    r = sqrt(sum(((x - x') / l)^2))
avec des longueurs ARD ``l`` par dimension.
"""

import numpy as np


class StationaryKernel:
    """Base class for the ARD stationary kernels.

    Attributes
    ----------
    lengthscales : numpy.ndarray
        Lengthscale of each input dimension, shape (d,).
    variance : float
        Signal variance (amplitude) of the kernel.
    """

    def __init__(self, lengthscales, variance=1.0):
        """Initialize the kernel.

        Parameters
        ----------
        lengthscales : float or array_like
            Lengthscale per dimension (scalar broadcast to all dimensions).
        variance : float, optional
            Signal variance, by default 1.0.
        """
        self.lengthscales = np.atleast_1d(np.asarray(lengthscales, dtype=float))
        self.variance = float(variance)

    @property
    def dim(self):
        """int: number of input dimensions of the kernel."""
        return int(self.lengthscales.shape[0])

    def _scaled_distance(self, X1, X2):
        """Return the squared scaled distance r^2 between the two point sets.

        Parameters
        ----------
        X1, X2 : array_like
            Points of shapes (n1, d) and (n2, d).

        Returns
        -------
        numpy.ndarray
            Squared scaled distance, shape (n1, n2).
        """
        X1, X2 = np.atleast_2d(X1), np.atleast_2d(X2)
        diff = X1[:, np.newaxis, :] - X2[np.newaxis, :, :]
        return (((diff / self.lengthscales) ** 2).sum(axis=-1))

    def K(self, X1, X2):
        """Evaluate the covariance matrix between X1 (n1, d) and X2 (n2, d).

        Returns
        -------
        numpy.ndarray
            Covariance matrix of shape (n1, n2).
        """
        raise NotImplementedError

    def diag(self, X):
        """Evaluate the diagonal of the covariance matrix on X (n, d).

        For stationary kernels the diagonal is the constant variance.

        Returns
        -------
        numpy.ndarray
            Array of shape (n,) equal to ``variance``.
        """
        X = np.atleast_2d(X)
        return np.full(X.shape[0], self.variance)


class SquaredExponentialARD(StationaryKernel):
    """Squared-exponential (RBF) kernel with ARD lengthscales.

    K(x, x') = sigma^2 exp(-1/2 sum_i ((x_i - x'_i) / l_i)^2)
    """

    def K(self, X1, X2):
        """Return the (n1, n2) covariance matrix between X1 and X2."""
        r2 = self._scaled_distance(X1, X2)
        return self.variance * np.exp(-0.5 * r2)


class Matern32ARD(StationaryKernel):
    """Matern 3/2 kernel with ARD lengthscales.

    K(x, x') = sigma^2 (1 + sqrt(3) r) exp(-sqrt(3) r)
    """

    def K(self, X1, X2):
        """Return the (n1, n2) covariance matrix between X1 and X2."""
        r = np.sqrt(self._scaled_distance(X1, X2))
        return self.variance * (1.0 + np.sqrt(3.0) * r) * np.exp(-np.sqrt(3.0) * r)


class Matern52ARD(StationaryKernel):
    """Matern 5/2 kernel with ARD lengthscales.

    K(x, x') = sigma^2 (1 + sqrt(5) r + 5 r^2 / 3) exp(-sqrt(5) r)
    """

    def K(self, X1, X2):
        """Return the (n1, n2) covariance matrix between X1 and X2."""
        r = np.sqrt(self._scaled_distance(X1, X2))
        return self.variance * (
            1.0 + np.sqrt(5.0) * r + 5.0 * r ** 2 / 3.0
        ) * np.exp(-np.sqrt(5.0) * r)
