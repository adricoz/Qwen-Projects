"""Noyaux stationnaires ARD (isotropes par dimension) pour processus gaussien."""
from __future__ import annotations

from typing import Sequence

import numpy as np


class _ARDKernel:
    """Base commune des noyaux stationnaires ARD.

    Parameters
    ----------
    lengthscales : sequence of float or float
        Echelle caracteristique de chaque dimension (une seule valeur si isotrope).
    variance : float
        Variance du signal (amplitude du noyau), doit etre strictement positive.
    """

    def __init__(self, lengthscales, variance: float = 1.0):
        lengthscales = np.atleast_1d(np.asarray(lengthscales, dtype=float))
        if np.any(lengthscales <= 0.0):
            raise ValueError("lengthscales must be strictly positive")
        if variance <= 0.0:
            raise ValueError("variance must be strictly positive")
        self.lengthscales = lengthscales
        self.variance = float(variance)

    # ------------------------------------------------------------------
    def _numerator(self, r: np.ndarray) -> np.ndarray:
        """Part polynomiale du noyau (a implementer par les sous-classes)."""
        raise NotImplementedError

    def _distances(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """Norme euclidienne anisotrope r = sqrt(sum(((x - x')/l)^2))."""
        z1 = X1 / self.lengthscales
        z2 = X2 / self.lengthscales
        # ||z1 - z2||^2 = ||z1||^2 + ||z2||^2 - 2 z1 . z2^T
        sq1 = np.sum(z1 * z1, axis=1)[:, None]
        sq2 = np.sum(z2 * z2, axis=1)[None, :]
        d2 = np.maximum(sq1 + sq2 - 2.0 * z1 @ z2.T, 0.0)
        return np.sqrt(d2)

    def K(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """Matrice de covariance K(X1, X2) de forme (n1, n2).

        Parameters
        ----------
        X1 : array-like of shape (n1, d)
            Points d'entree.
        X2 : array-like of shape (n2, d)
            Points de sortie.

        Returns
        -------
        np.ndarray of shape (n1, n2)
            Valeurs du noyau K(xi, xj) = variance * c(r_ij).
        """
        X1 = np.atleast_2d(np.asarray(X1, dtype=float))
        X2 = np.atleast_2d(np.asarray(X2, dtype=float))
        self._dim(X1, X2)
        r = self._distances(X1, X2)
        return self.variance * self._numerator(r)

    def diag(self, X: np.ndarray) -> np.ndarray:
        """Diagonale du noyau: K(x, x) pour chaque point, forme (n,).

        Parameters
        ----------
        X : array-like of shape (n, d)
            Points d'entree.

        Returns
        -------
        np.ndarray of shape (n,)
            Valeurs K(x, x), egales a la variance si le noyau est normalise.
        """
        X = np.atleast_2d(np.asarray(X, dtype=float))
        return self.variance * self._numerator(np.zeros((len(X), 1)))[:, 0]

    def _dim(self, X1: np.ndarray, X2: np.ndarray) -> None:
        if X1.shape[1] != X2.shape[1]:
            raise ValueError("X1 and X2 have different numbers of dimensions")
        if X1.shape[1]:
            if len(self.lengthscales) != 1 and X1.shape[1] != len(self.lengthscales):
                raise ValueError("wrong number of lengthscales for input dimension")


class SquaredExponentialARD(_ARDKernel):
    """Noyau squared exponential (RBF) ARD: K = s^2 exp(-r^2 / 2)."""

    def _numerator(self, r):
        return np.exp(-0.5 * r * r)


class Matern32ARD(_ARDKernel):
    """Noyau Matern nu=3/2 ARD: K = s^2 (1 + sqrt(3) r) exp(-sqrt(3) r)."""

    def _numerator(self, r):
        s3 = np.sqrt(3.0)
        return (1.0 + s3 * r) * np.exp(-s3 * r)


class Matern52ARD(_ARDKernel):
    """Noyau Matern nu=5/2 ARD: K = s^2 (1 + sqrt(5) r + 5 r^2 / 3) exp(-sqrt(5) r)."""

    def _numerator(self, r):
        s5 = np.sqrt(5.0)
        return (1.0 + s5 * r + (5.0 / 3.0) * r * r) * np.exp(-s5 * r)
