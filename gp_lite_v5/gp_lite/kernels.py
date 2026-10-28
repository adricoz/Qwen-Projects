"""Kernels (stationary covariance functions) for `gp_lite`.

Conventions
-----------
- Inputs are 2D arrays of shape (n, d).
- Distance: r = sqrt(sum(((x - x') / lengthscales)^2)) (Euclidean, per-dimension
  lengthscale, "ARD").
- Matern 3/2: K = sigma^2 (1 + sqrt(3) r) exp(-sqrt(3) r)
- Matern 5/2: K = sigma^2 (1 + sqrt(5) r + 5 r^2 / 3) exp(-sqrt(5) r)
- Squared exponential (RBF): K = sigma^2 exp(-r^2 / 2)
"""
from __future__ import annotations

import numpy as np


class ARDStationaryKernel:
    """Base class of the stationary ARD kernels of `gp_lite`.

    Parameters
    ----------
    lengthscales : array_like, shape (d,)
        Lengthscale of each input dimension (positive).
    variance : float
        Signal variance (positive).
    """

    def __init__(self, lengthscales, variance: float = 1.0):
        self.lengthscales = np.asarray(lengthscales, dtype=float).ravel()
        self.variance = float(variance)
        if np.any(self.lengthscales <= 0) or self.variance <= 0:
            raise ValueError("lengthscales and variance must be strictly positive")

    # -- parameters ---------------------------------------------------------
    @property
    def dimension(self) -> int:
        """Number of input dimensions."""
        return int(self.lengthscales.size)

    def set_params(self, lengthscales, variance: float) -> None:
        """Set kernel hyperparameters (used by the optimizer of `GPRegressor`).

        Parameters
        ----------
        lengthscales : array_like, shape (d,)
            New positive lengthscales.
        variance : float
            New positive signal variance.
        """
        self.lengthscales = np.asarray(lengthscales, dtype=float).ravel()
        self.variance = float(variance)

    # -- mathematics --------------------------------------------------------
    def _scaled_distance(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """Matrix of scaled Euclidean distances r = sqrt(sum(((x-x')/l)^2))."""
        X1 = np.atleast_2d(X1)
        X2 = np.atleast_2d(X2)
        X1s = X1 / self.lengthscales
        X2s = X2 / self.lengthscales
        # (n1, n2) squared distances via ||a||^2 + ||b||^2 - 2 a.b
        d2 = (np.sum(X1s**2, axis=1)[:, None]
              + np.sum(X2s**2, axis=1)[None, :]
              - 2.0 * X1s @ X2s.T)
        return np.sqrt(np.maximum(d2, 0.0))

    def _k_of_r(self, r: np.ndarray) -> np.ndarray:
        """Kernel value as a function of the scaled distance (pure function)."""
        raise NotImplementedError

    def K(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """Covariance matrix between two sets of points.

        Parameters
        ----------
        X1 : array_like, shape (n1, d)
        X2 : array_like, shape (n2, d)

        Returns
        -------
        np.ndarray, shape (n1, n2)
            K(X1, X2) with K_ii = variance (no noise; noise is handled by the
            `GPRegressor`).
        """
        return self.variance * self._k_of_r(self._scaled_distance(X1, X2))

    def diag(self, X: np.ndarray) -> np.ndarray:
        """Diagonal of the covariance matrix of the points of `X`.

        Parameters
        ----------
        X : array_like, shape (n, d)

        Returns
        -------
        np.ndarray, shape (n,)
            Always the signal variance (stationary kernels with K(x, x) = sigma^2).
        """
        n = np.atleast_2d(X).shape[0]
        return np.full(n, self.variance, dtype=float)


class SquaredExponentialARD(ARDStationaryKernel):
    """Squared exponential (RBF) kernel with per-dimension lengthscales.

    K(x, x') = sigma^2 exp(-r^2 / 2), r = sqrt(sum(((x - x')/l)^2)).
    """

    def _k_of_r(self, r: np.ndarray) -> np.ndarray:
        return np.exp(-0.5 * r**2)


class Matern32ARD(ARDStationaryKernel):
    """Matern kernel with nu = 3/2 and per-dimension lengthscales.

    K(x, x') = sigma^2 (1 + sqrt(3) r) exp(-sqrt(3) r).
    """

    def _k_of_r(self, r: np.ndarray) -> np.ndarray:
        return (1.0 + np.sqrt(3.0) * r) * np.exp(-np.sqrt(3.0) * r)


class Matern52ARD(ARDStationaryKernel):
    """Matern kernel with nu = 5/2 and per-dimension lengthscales.

    K(x, x') = sigma^2 (1 + sqrt(5) r + 5 r^2 / 3) exp(-sqrt(5) r).
    """

    def _k_of_r(self, r: np.ndarray) -> np.ndarray:
        return (1.0 + np.sqrt(5.0) * r + (5.0 / 3.0) * r**2) * np.exp(-np.sqrt(5.0) * r)
