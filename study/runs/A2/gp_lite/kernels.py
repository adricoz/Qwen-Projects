"""Kernels (covariance functions) for `gp_lite`.

Tous les noyaux sont isotropes par dimension (ARD): chaque dimension i a sa
propre longueur d'onde ``lengthscales[i]``.
"""
import numpy as np


class _ARDKernel:
    """Classe de base des noyaux ARD.

    Attributes
    ----------
    lengthscales : numpy.ndarray, shape (d,)
        Longueurs d'onde par dimension.
    variance : float
        Amplitude (variance) du noyau.
    """

    def __init__(self, lengthscales, variance=1.0):
        lengthscales = np.atleast_1d(np.asarray(lengthscales, dtype=float))
        self.lengthscales = lengthscales
        self.variance = float(variance)

    # -- utilitaires ---------------------------------------------------
    def _scaled_diff(self, X1, X2):
        """Différences (X1 - X2) normalisées par les longueurs d'onde."""
        X1 = np.atleast_2d(X1)
        X2 = np.atleast_2d(X2)
        l = self.lengthscales
        if l.size == 1:
            return (X1[:, None, :] - X2[None, :, :]) / l
        return (X1[:, None, :] - X2[None, :, :]) / l[None, None, :]

    def _distance(self, X1, X2):
        """Distance euclidienne pondérée r = sqrt(sum(((x-x')/l)^2))."""
        d = self._scaled_diff(X1, X2)
        return np.sqrt(np.sum(d ** 2, axis=-1))

    # -- API -----------------------------------------------------------
    def K(self, X1, X2):
        """Matrice de covariance entre X1 et X2.

        Parameters
        ----------
        X1 : array-like, shape (n1, d)
            Premier ensemble de points.
        X2 : array-like, shape (n2, d)
            Deuxième ensemble de points.

        Returns
        -------
        numpy.ndarray, shape (n1, n2)
            Matrice de covariance.
        """
        raise NotImplementedError

    def diag(self, X):
        """Diagonale K(X, X).

        Parameters
        ----------
        X : array-like, shape (n, d)
            Points d'évaluation.

        Returns
        -------
        numpy.ndarray, shape (n,)
            Valeurs de la diagonale.
        """
        n = np.atleast_2d(np.asarray(X, dtype=float)).shape[0]
        return np.full(n, self.variance)


class SquaredExponentialARD(_ARDKernel):
    """Noyau RBF (Squared Exponential) à longueurs d'onde ARD.

    ``K(x, x') = sigma^2 exp(-0.5 * sum(((x_i - x'_i) / l_i)^2))``.
    """

    def K(self, X1, X2):
        """Matrice de covariance RBF entre X1 et X2."""
        d = self._scaled_diff(X1, X2)
        r2 = np.sum(d ** 2, axis=-1)
        return self.variance * np.exp(-0.5 * r2)


class Matern32ARD(_ARDKernel):
    """Noyau Matérn nu=3/2 à longueurs d'onde ARD.

    ``K(x, x') = sigma^2 (1 + sqrt(3) r) exp(-sqrt(3) r)``.
    """

    def K(self, X1, X2):
        """Matrice de covariance Matérn 3/2 entre X1 et X2."""
        r = self._distance(X1, X2)
        return self.variance * (1.0 + np.sqrt(3.0) * r) * np.exp(-np.sqrt(3.0) * r)


class Matern52ARD(_ARDKernel):
    """Noyau Matérn nu=5/2 à longueurs d'onde ARD.

    ``K(x, x') = sigma^2 (1 + sqrt(5) r + 5 r^2 / 3) exp(-sqrt(5) r)``.
    """

    def K(self, X1, X2):
        """Matrice de covariance Matérn 5/2 entre X1 et X2."""
        r = self._distance(X1, X2)
        sqrt5 = np.sqrt(5.0)
        return self.variance * (1.0 + sqrt5 * r + 5.0 * r ** 2 / 3.0) * np.exp(-sqrt5 * r)
