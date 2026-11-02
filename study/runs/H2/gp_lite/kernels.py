"""Coefficients de covariance (noyaux) stationnaires avec anisotropie (ARD).

Tous les noyaux sont isotropes apres mise a l'echelle des entrees : la distance
evaluee est r = sqrt(sum(((x - x') / l_i)^2)) avec l_i les lengthscales.
"""
import numpy as np


class _BaseARD:
    """Base commune : gestion des hyperparametres et de la distance evaluee.

    Parameters
    ----------
    lengthscales : array-like, shape (d,)
        Ecart caracteristique de chaque dimension d'entree.
    variance : float, optional
        Variance du signal (amplitude du noyau). Par defaut 1.0.
    """

    def __init__(self, lengthscales, variance=1.0):
        self.lengthscales = np.atleast_1d(np.asarray(lengthscales, dtype=float)).reshape(-1)
        self.variance = float(variance)
        self.d = self.lengthscales.shape[0]

    @staticmethod
    def _as_2d(X):
        """Convertit X en tableau 2D (n, d)."""
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return X

    def _scaled_dist(self, X1, X2):
        """Distance euclidienne evaluee r entre chaque paire (x, x')."""
        X1, X2 = self._as_2d(X1), self._as_2d(X2)
        diff = (X1[:, None, :] - X2[None, :, :]) / self.lengthscales[None, None, :]
        return np.sqrt(np.sum(diff ** 2, axis=-1))

    def K(self, X1, X2):
        """Matrice de covariance entre X1 (n1, d) et X2 (n2, d) -> (n1, n2).

        Parameters
        ----------
        X1, X2 : array_like, shape (n, d)
            Points d'entree.

        Returns
        -------
        ndarray, shape (n1, n2)
            Matrice de covariance K = variance * psi(r).
        """
        r = self._scaled_dist(X1, X2)
        return self.variance * self._psi(r)

    def diag(self, X):
        """Diagonale du noyau sur les points X (valeur K(x, x)).

        Parameters
        ----------
        X : array_like, shape (n, d)

        Returns
        -------
        ndarray, shape (n,)
            Valeurs K(x_i, x_i) = variance.
        """
        return np.full(self._as_2d(X).shape[0], self.variance, dtype=float)

    def _psi(self, r):  # pragma: no cover - interface abstraite
        raise NotImplementedError


class SquaredExponentialARD(_BaseARD):
    """Noyau gaussien (RBF) anisotrope : K(x, x') = sigma^2 exp(-r^2 / 2).

    Parameters
    ----------
    lengthscales : array-like, shape (d,)
        Ecart caracteristique de chaque dimension.
    variance : float, optional
        Variance du signal. Par defaut 1.0.
    """

    def _psi(self, r):
        return np.exp(-0.5 * r ** 2)


class Matern32ARD(_BaseARD):
    """Noyau Matern 3/2 anisotrope : K = sigma^2 (1 + sqrt(3) r) exp(-sqrt(3) r).

    Parameters
    ----------
    lengthscales : array-like, shape (d,)
        Ecart caracteristique de chaque dimension.
    variance : float, optional
        Variance du signal. Par defaut 1.0.
    """

    def _psi(self, r):
        s3 = np.sqrt(3.0)
        return (1.0 + s3 * r) * np.exp(-s3 * r)


class Matern52ARD(_BaseARD):
    """Noyau Matern 5/2 anisotrope : K = sigma^2 (1 + sqrt(5) r + 5 r^2 / 3) exp(-sqrt(5) r).

    Parameters
    ----------
    lengthscales : array-like, shape (d,)
        Ecart caracteristique de chaque dimension.
    variance : float, optional
        Variance du signal. Par defaut 1.0.
    """

    def _psi(self, r):
        s5 = np.sqrt(5.0)
        return (1.0 + s5 * r + (5.0 / 3.0) * r ** 2) * np.exp(-s5 * r)
