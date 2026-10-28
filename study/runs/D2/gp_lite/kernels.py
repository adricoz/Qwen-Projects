"""Noyaux (covariance kernels) de gp_lite, sur des entrées continues de dimension d.

Tous les noyaux sont stationnaires et ARD (un lengthscale par dimension) :
r(X1, X2) = sqrt(sum_j ((x1_j - x2_j) / l_j)^2),
avec lengthscales de forme (d,).
"""
from __future__ import annotations

import numpy as np

__all__ = ["SquaredExponentialARD", "Matern32ARD", "Matern52ARD"]


class _ARDKernel:
    """Base commune : validation des hyperparamètres et distance normalisée."""

    def __init__(self, lengthscales, variance=1.0):
        lengthscales = np.atleast_1d(np.asarray(lengthscales, dtype=float))
        if not np.all(np.isfinite(lengthscales)) or np.any(lengthscales <= 0):
            raise ValueError("lengthscales must be positive")
        if variance <= 0:
            raise ValueError("variance must be positive")
        self.lengthscales = lengthscales
        self.variance = float(variance)

    # -- helpers ---------------------------------------------------------
    def _dim(self):
        return self.lengthscales.shape[0]

    def _scaled_distance(self, X1, X2):
        """Distance r = sqrt(sum(((x - x')/l)^2)) de forme (n1, n2)."""
        n1, d = X1.shape
        if d != self._dim():
            raise ValueError(f"expected {self._dim()} input dims, got {d}")
        diff = (X1[:, None, :] - X2[None, :, :]) / self.lengthscales
        return np.sqrt(np.sum(diff * diff, axis=2))

    # -- API -------------------------------------------------------------
    def K(self, X1, X2):
        """Matrice de covariance (n1, n2). Implémentée par les sous-classes."""
        raise NotImplementedError

    def diag(self, X):
        """Diagonale de K(X, X), array (n,). Implémentée par les sous-classes."""
        raise NotImplementedError

    def __repr__(self):
        return (f"{type(self).__name__}(lengthscales={self.lengthscales}, "
                f"variance={self.variance})")


class SquaredExponentialARD(_ARDKernel):
    """Noyau RBF à longue mémoire : K = sigma^2 exp(-r^2 / 2).

    Parameters
    ----------
    lengthscales : array-like (d,)
        Échelles caractéristiques par dimension.
    variance : float, default 1.0
        Variance du signal.
    """

    def K(self, X1, X2):
        """Compute the covariance matrix (n1, n2) between X1 and X2.

        Parameters
        ----------
        X1, X2 : array (n1, d), (n2, d)
            Points of input space.
        """
        r = self._scaled_distance(X1, X2)
        return self.variance * np.exp(-0.5 * r * r)

    def diag(self, X):
        """Return the diagonal of K(X, X), i.e. variance repeated n times."""
        return np.full(X.shape[0], self.variance, dtype=float)


class Matern32ARD(_ARDKernel):
    """Noyau Matérn nu=3/2 : K = sigma^2 (1 + sqrt(3) r) exp(-sqrt(3) r).

    Parameters
    ----------
    lengthscales : array-like (d,)
        Échelles caractéristiques par dimension.
    variance : float, default 1.0
        Variance du signal.
    """

    def K(self, X1, X2):
        """Compute the covariance matrix (n1, n2) between X1 and X2."""
        r = np.sqrt(3.0) * self._scaled_distance(X1, X2)
        return self.variance * (1.0 + r) * np.exp(-r)

    def diag(self, X):
        """Return the diagonal of K(X, X), i.e. variance repeated n times."""
        return np.full(X.shape[0], self.variance, dtype=float)


class Matern52ARD(_ARDKernel):
    """Noyau Matérn nu=5/2 : K = sigma^2 (1 + sqrt(5) r + 5 r^2 / 3) exp(-sqrt(5) r).

    Parameters
    ----------
    lengthscales : array-like (d,)
        Échelles caractéristiques par dimension.
    variance : float, default 1.0
        Variance du signal.
    """

    def K(self, X1, X2):
        """Compute the covariance matrix (n1, n2) between X1 and X2."""
        rho = np.sqrt(5.0) * self._scaled_distance(X1, X2)
        return self.variance * (1.0 + rho + 5.0 * rho * rho / 3.0) * np.exp(-rho)

    def diag(self, X):
        """Return the diagonal of K(X, X), i.e. variance repeated n times."""
        return np.full(X.shape[0], self.variance, dtype=float)
