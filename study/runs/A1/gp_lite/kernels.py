"""Noyaux stationnaires a anisotropie (ARD) pour le paquet ``gp_lite``.

Toutes les classes partagent la meme interface imposee :

- constructeur ``Classe(lengthscales, variance=1.0)`` ;
- attributs publics ``lengthscales`` (ndarray ``(d,)``) et ``variance`` (float) ;
- methode ``K(X1, X2)`` -> matrice de covariance ``(n1, n2)`` ;
- methode ``diag(X)`` -> vecteur des variances marginales ``(n,)``.

Convention d'ecart : ``r = sqrt(sum(((x - x') / l) ** 2))`` avec ``l`` les
longueurs caracteristiques ARD.
"""

import numpy as np


class _ARDKernelBase:
    """Base commune aux noyaux ARD : validation des hyperparametres et distance ARD."""

    def __init__(self, lengthscales, variance=1.0):
        ls = np.asarray(lengthscales, dtype=float)
        if ls.ndim > 1:
            raise ValueError("lengthscales must be a scalar or a 1-D array")
        if np.any(ls <= 0):
            raise ValueError("lengthscales must be strictly positive")
        if variance <= 0:
            raise ValueError("variance must be strictly positive")
        self.lengthscales = ls  # scalaire ou ndarray (d,)
        self.variance = float(variance)

    def _lengthscale_vector(self, d):
        """Return ``lengthscales`` broadcast to a vector of dimension ``d``."""
        ls = self.lengthscales
        if ls.ndim == 0:
            return np.full(d, float(ls))
        if ls.shape != (d,):
            raise ValueError(
                f"lengthscales has dimension {ls.shape[0]}, expected {d}"
            )
        return ls

    def _distance(self, X1, X2):
        """Standardised ARD distance ``r`` between rows of ``X1`` and ``X2``.

        Parameters
        ----------
        X1 : ndarray (n1, d)
        X2 : ndarray (n2, d)

        Returns
        -------
        ndarray (n1, n2)
        """
        d = X1.shape[1]
        l = self._lengthscale_vector(d)
        dx = (X1[:, None, :] - X2[None, :, :]) / l
        return np.sqrt(np.sum(dx ** 2, axis=2))

    def diag(self, X):
        """Marginal variance at each point of ``X`` (always equal to ``variance``)."""
        return np.full(X.shape[0], self.variance)

    def K(self, X1, X2):
        """Covariance matrix between the rows of ``X1`` and ``X2``.

        Parameters
        ----------
        X1 : ndarray (n1, d)
        X2 : ndarray (n2, d)

        Returns
        -------
        ndarray (n1, n2)
        """
        X1 = np.atleast_2d(np.asarray(X1, dtype=float))
        X2 = np.atleast_2d(np.asarray(X2, dtype=float))
        if X1.shape[1] != X2.shape[1]:
            raise ValueError("X1 and X2 must have the same dimension")
        r = self._distance(X1, X2)
        return self.variance * self._shape(r)

    def _shape(self, r):
        """Kernel of the standardised distance (to override in subclasses)."""
        raise NotImplementedError


class SquaredExponentialARD(_ARDKernelBase):
    """Squared exponential (RBF) kernel with ARD lengthscales:
    ``K(x, x') = variance * exp(-0.5 * r^2)``."""

    def _shape(self, r):
        return np.exp(-0.5 * r ** 2)


class Matern32ARD(_ARDKernelBase):
    """Matern 3/2 kernel with ARD lengthscales:
    ``K(x, x') = variance * (1 + sqrt(3) r) exp(-sqrt(3) r)``."""

    def _shape(self, r):
        s = np.sqrt(3.0)
        return (1.0 + s * r) * np.exp(-s * r)


class Matern52ARD(_ARDKernelBase):
    """Matern 5/2 kernel with ARD lengthscales:
    ``K(x, x') = variance * (1 + sqrt(5) r + 5 r^2 / 3) exp(-sqrt(5) r)``."""

    def _shape(self, r):
        s = np.sqrt(5.0)
        return (1.0 + s * r + 5.0 * r ** 2 / 3.0) * np.exp(-s * r)
