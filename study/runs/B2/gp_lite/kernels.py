"""Noyaux (kernels) stationnaires à longueurs d'onde anisotropes (ARD)."""
import numpy as np


class _ARDKernel:
    """Classe de base : ARD (une lengthscales par dimension)."""

    def __init__(self, lengthscales, variance=1.0):
        """
        Initialize the kernel.

        Parameters
        ----------
        lengthscales : array_like, shape (d,)
            Une longueur caractéristique par dimension d'entrée.
        variance : float
            Amplitude du noyau (variance a priori).
        """
        self.lengthscales = np.atleast_1d(np.asarray(lengthscales, dtype=float)).copy()
        self.variance = float(variance)

    # -- utilitaires internes -------------------------------------------------
    def _r(self, X1, X2):
        """Distance euclidienne échantonnée par les lengthscales, forme (n1, n2)."""
        r2 = np.sum(((X1[:, None, :] - X2[None, :, :]) / self.lengthscales) ** 2, axis=2)
        return np.sqrt(r2)

    def K(self, X1, X2):
        """Matrice du noyau entre X1 (n1, d) et X2 (n2, d), forme (n1, n2)."""
        raise NotImplementedError

    def diag(self, X):
        """Valeurs diagonales : un noyau stationnaire vaut variance en r = 0."""
        X = np.atleast_2d(X)
        return np.full(X.shape[0], self.variance)


class SquaredExponentialARD(_ARDKernel):
    """Noyau carré-exponentiel (RBF) ARD : K = sigma^2 exp(-0.5 r^2)."""

    def K(self, X1, X2):
        """Matrice de covariance SE ARD entre X1 et X2, forme (n1, n2)."""
        X1, X2 = np.atleast_2d(X1), np.atleast_2d(X2)
        r2 = np.sum(((X1[:, None, :] - X2[None, :, :]) / self.lengthscales) ** 2, axis=2)
        return self.variance * np.exp(-0.5 * r2)


class Matern32ARD(_ARDKernel):
    """Noyau Matérn nu = 3/2 ARD : K = sigma^2 (1 + sqrt(3) r) exp(-sqrt(3) r)."""

    def K(self, X1, X2):
        """Matrice de covariance Matérn 3/2 ARD entre X1 et X2, forme (n1, n2)."""
        X1, X2 = np.atleast_2d(X1), np.atleast_2d(X2)
        r = self._r(X1, X2)
        sq3 = np.sqrt(3.0)
        return self.variance * (1.0 + sq3 * r) * np.exp(-sq3 * r)


class Matern52ARD(_ARDKernel):
    """Noyau Matérn nu = 5/2 ARD : K = sigma^2 (1 + sqrt(5) r + 5 r^2 / 3) exp(-sqrt(5) r)."""

    def K(self, X1, X2):
        """Matrice de covariance Matérn 5/2 ARD entre X1 et X2, forme (n1, n2)."""
        X1, X2 = np.atleast_2d(X1), np.atleast_2d(X2)
        r = self._r(X1, X2)
        sq5 = np.sqrt(5.0)
        return self.variance * (1.0 + sq5 * r + 5.0 * r * r / 3.0) * np.exp(-sq5 * r)
