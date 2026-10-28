"""Noyaux (kernels) pour processus gaussien, en espace d'entrees ARD."""
import numpy as np


class _ARDKernel:
    """Base commune des noyaux stationnaires ARD (un lengthscale par dimension).

    Parameters
    ----------
    lengthscales : array_like
        Lengthscale par dimension, de longueur ``d`` (ou scalaire).
    variance : float
        Variance du signal (hyperparametre ``sigma^2``).
    """

    def __init__(self, lengthscales, variance=1.0):
        self.lengthscales = np.atleast_1d(np.asarray(lengthscales, dtype=float))
        self.variance = float(variance if variance is not None else 1.0)

    # -- API publique -------------------------------------------------------
    def n_dims(self):
        """Return the number of input dimensions (length of ``lengthscales``)."""
        return self.lengthscales.size

    def K(self, X1, X2):
        """Compute the covariance matrix ``K(X1, X2)``.

        Parameters
        ----------
        X1 : array_like
            Points of shape ``(n1, d)`` (or ``(d,)``).
        X2 : array_like
            Points of shape ``(n2, d)``.

        Returns
        -------
        numpy.ndarray
            Matrix of shape ``(n1, n2)``.
        """
        X1 = np.atleast_2d(np.asarray(X1, dtype=float))
        X2 = np.atleast_2d(np.asarray(X2, dtype=float))
        d = X1.shape[1]
        if self.lengthscales.size == 1:
            lengths = np.full(d, self.lengthscales[0])
        else:
            lengths = self.lengthscales
        if lengths.size != d:
            raise ValueError(
                "dimension mismatch: data has %d dimensions, kernel has %d lengthscales"
                % (d, lengths.size))
        r = np.sqrt(np.sum(
            ((X1[:, None, :] - X2[None, :, :]) / lengths) ** 2, axis=2))
        return self._scale(r)

    def diag(self, X):
        """Return the kernel diagonal at the points ``X``.

        Parameters
        ----------
        X : array_like
            Points of shape ``(n, d)``.

        Returns
        -------
        numpy.ndarray
            Array of shape ``(n,)``, here equal to ``variance`` (stationary kernel).
        """
        X = np.atleast_2d(np.asarray(X, dtype=float))
        return np.full(X.shape[0], self.variance)

    def _scale(self, r):
        """Evaluate the dimensionless kernel ``k(r)`` (to be overridden)."""
        raise NotImplementedError


class SquaredExponentialARD(_ARDKernel):
    """Squared exponential (RBF) kernel with ARD lengthscales:
    ``K = sigma^2 exp(-r^2 / 2)`` with ``r^2 = sum_i ((x_i - x'_i)/l_i)^2``.
    """

    def _scale(self, r):
        return self.variance * np.exp(-0.5 * r ** 2)


class Matern32ARD(_ARDKernel):
    """Matern 3/2 kernel with ARD lengthscales:
    ``K = sigma^2 (1 + sqrt(3) r) exp(-sqrt(3) r)``.
    """

    def _scale(self, r):
        return self.variance * (1.0 + np.sqrt(3.0) * r) * np.exp(-np.sqrt(3.0) * r)


class Matern52ARD(_ARDKernel):
    """Matern 5/2 kernel with ARD lengthscales:
    ``K = sigma^2 (1 + sqrt(5) r + 5 r^2 / 3) exp(-sqrt(5) r)``.
    """

    def _scale(self, r):
        return self.variance * (
            1.0 + np.sqrt(5.0) * r + (5.0 / 3.0) * r ** 2) * np.exp(-np.sqrt(5.0) * r)
