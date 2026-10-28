"""Noyaux (kernels) du paquet gp_lite.

Toutes les classes implémentent :
- ``K(X1, X2)`` : matrice de covariance (n1, n2) entre deux nuages de points ;
- ``diag(X)``   : valeurs k(x, x) sur la diagonale, forme (n1,).
"""
import numpy as np


class Kernel:
    """Classe de base d'un noyau stationnaire à longueurs variables (ARD).

    Attributs publics
    -----------------
    lengthscales : ndarray, shape (d,)
        Longueurs de caractéristique, une par dimension d'entrée.
    variance : float
        Variance du signal (amplitude du noyau).
    """

    def __init__(self, lengthscales, variance=1.0):
        self.lengthscales = np.asarray(lengthscales, dtype=float).reshape(-1)
        self.variance = float(variance)

    # ------------------------------------------------------------------ #
    def _squared_dist(self, X1, X2):
        """Distances euclidiennes au carré, pondérées par les longueurs.

        Returns
        -------
        ndarray, shape (n1, n2)
        """
        # r^2 = sum_j ((x1j - x2j)^2 / l_j^2)
        X1s = X1 / self.lengthscales
        X2s = X2 / self.lengthscales
        return np.sum(X1s ** 2, axis=1)[:, None] \
             + np.sum(X2s ** 2, axis=1)[None, :] \
             - 2.0 * (X1s @ X2s.T)

    def K(self, X1, X2):
        """Matrice de covariance (n1, n2).

        Parameters
        ----------
        X1 : ndarray, shape (n1, d)
        X2 : ndarray, shape (n2, d)
        """
        raise NotImplementedError

    def diag(self, X):
        """Diagonale k(x, x), shape (n1,)."""
        return np.full(len(X), self.variance)

    def __call__(self, X1, X2):
        """Alias de :meth:`K`."""
        return self.K(X1, X2)


class SquaredExponentialARD(Kernel):
    """Noyau Gaussian / squared exponential avec longueurs ARD.

    k(x, x') = sigma^2 * exp(-0.5 * sum_j ((x_j - x'_j) / l_j)^2)
    """

    def K(self, X1, X2):
        """Matrice de covariance (n1, n2).

        Parameters
        ----------
        X1 : ndarray, shape (n1, d)
        X2 : ndarray, shape (n2, d)
        """
        return self.variance * np.exp(-0.5 * self._squared_dist(X1, X2))


class _MaternARD(Kernel):
    """Base des noyaux Matérn ARD : r = sqrt(sum_j ((x_j - x'_j)/l_j)^2)."""

    def _K_from_r(self, r):
        raise NotImplementedError

    def K(self, X1, X2):
        """Matrice de covariance (n1, n2).

        Parameters
        ----------
        X1 : ndarray, shape (n1, d)
        X2 : ndarray, shape (n2, d)
        """
        r = np.sqrt(np.maximum(self._squared_dist(X1, X2), 0.0))
        return self._K_from_r(r)


class Matern32ARD(_MaternARD):
    """Noyau Matérn nu = 3/2 avec longueurs ARD.

    k(r) = sigma^2 * (1 + sqrt(3) r) * exp(-sqrt(3) r)
    """

    def _K_from_r(self, r):
        return self.variance * (1.0 + np.sqrt(3.0) * r) * np.exp(-np.sqrt(3.0) * r)


class Matern52ARD(_MaternARD):
    """Noyau Matérn nu = 5/2 avec longueurs ARD.

    k(r) = sigma^2 * (1 + sqrt(5) r + 5 r^2 / 3) * exp(-sqrt(5) r)
    """

    def _K_from_r(self, r):
        s5 = np.sqrt(5.0)
        return self.variance * (1.0 + s5 * r + 5.0 * r * r / 3.0) * np.exp(-s5 * r)
