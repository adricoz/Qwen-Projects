"""Noyaux stationnaires ARD (une longueur d'onde par dimension d'entree).

Convention de distance : r = sqrt(sum(((x - x') / l)^2), puis chaque noyau
est une fonction de r multipliee par la variance sigmad2.
"""
import numpy as np


class SquaredExponentialARD:
    """Noyau Gaussian (Squared Exponential) avec longueurs d'onde ARD.

    Parameters
    ----------
    lengthscales : array_like, shape (d,)
        Longueur d'onde positive par dimension d'entree.
    variance : float, optional
        Variance du signal (defaut 1.0).

    Notes
    -----
    K(x, x') = sigmad2 * exp(-0.5 * r^2), avec
    r = sqrt(sum(((x - x') / l)^2)).
    """

    def __init__(self, lengthscales, variance=1.0):
        self.lengthscales = np.atleast_1d(np.asarray(lengthscales, dtype=float))
        self.variance = float(variance)

    def _scaled_diff(self, X1, X2):
        """(X1[:, i] - X2[:, i]) / l_i en forme (n1, n2)."""
        X1 = np.atleast_2d(X1)
        X2 = np.atleast_2d(np.asarray(X2, dtype=float))
        diff = (X1 - X2.T) / self.lengthscales
        return diff

    def K(self, X1, X2):
        """Matrice de covariance entre X1 (n1, d) et X2 (n2, d).

        Returns
        -------
        K : ndarray, shape (n1, n2)
        """
        diff = self._scaled_diff(X1, X2)
        r2 = np.sum(diff ** 2, axis=1)
        return self.variance * np.exp(-0.5 * r2)

    def diag(self, X):
        """Diagonale du noyau : identiquement sigmad2 (independante de X).

        Returns
        -------
        d : ndarray, shape (n,)
        """
        n = np.atleast_2d(X).shape[0]
        return np.full(n, self.variance)


class Matern32ARD:
    """Noyau Matern nu=3/2 avec longueurs d'onde ARD.

    Parameters
    ----------
    lengthscales : array_like, shape (d,)
        Longueur d'onde positive par dimension d'entree.
    variance : float, optional
        Variance du signal (defaut 1.0).

    Notes
    -----
    K(x, x') = sigmad2 * (1 + sqrt(3) r) * exp(-sqrt(3) r), avec
    r = sqrt(sum(((x - x') / l)^2)).
    """

    def __init__(self, lengthscales, variance=1.0):
        self.lengthscales = np.atleast_1d(np.asarray(lengthscales, dtype=float))
        self.variance = float(variance)

    def _scaled_diff(self, X1, X2):
        X1 = np.atleast_2d(X1)
        X2 = np.atleast_2d(np.asarray(X2, dtype=float))
        return (X1 - X2.T) / self.lengthscales

    def K(self, X1, X2):
        """Matrice de covariance entre X1 (n1, d) et X2 (n2, d).

        Returns
        -------
        K : ndarray, shape (n1, n2)
        """
        diff = self._scaled_diff(X1, X2)
        r = np.sqrt(np.sum(diff ** 2, axis=1))
        s = np.sqrt(3.0)
        return self.variance * (1.0 + s * r) * np.exp(-s * r)

    def diag(self, X):
        """Diagonale du noyau : identiquement sigmad2 (independante de X).

        Returns
        -------
        d : ndarray, shape (n,)
        """
        n = np.atleast_2d(X).shape[0]
        return np.full(n, self.variance)


class Matern52ARD:
    """Noyau Matern nu=5/2 avec longueurs d'onde ARD.

    Parameters
    ----------
    lengthscales : array_like, shape (d,)
        Longueur d'onde positive par dimension d'entree.
    variance : float, optional
        Variance du signal (defaut 1.0).

    Notes
    -----
    K(x, x') = sigmad2 * (1 + sqrt(5) r + 5 r^2 / 3) * exp(-sqrt(5) r), avec
    r = sqrt(sum(((x - x') / l)^2)).
    """

    def __init__(self, lengthscales, variance=1.0):
        self.lengthscales = np.atleast_1d(np.asarray(lengthscales, dtype=float))
        self.variance = float(variance)

    def _scaled_diff(self, X1, X2):
        X1 = np.atleast_2d(X1)
        X2 = np.atleast_2d(np.asarray(X2, dtype=float))
        return (X1 - X2.T) / self.lengthscales

    def K(self, X1, X2):
        """Matrice de covariance entre X1 (n1, d) et X2 (n2, d).

        Returns
        -------
        K : ndarray, shape (n1, n2)
        """
        diff = self._scaled_diff(X1, X2)
        r = np.sqrt(np.sum(diff ** 2, axis=1))
        s = np.sqrt(5.0)
        return self.variance * (1.0 + s * r + 5.0 * r ** 2 / 3.0) * np.exp(-s * r)

    def diag(self, X):
        """Diagonale du noyau : identiquement sigmad2 (independante de X).

        Returns
        -------
        d : ndarray, shape (n,)
        """
        n = np.atleast_2d(X).shape[0]
        return np.full(n, self.variance)
