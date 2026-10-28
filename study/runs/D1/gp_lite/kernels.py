"""Noyaux de covariance avec ARD (un lengthscale par dimension).

Conventions :
- ``K(X1, X2)`` renvoie une matrice de forme ``(n1, n2)`` ;
- la distance normalisee est ``r = sqrt(sum(((x - x') / l) ** 2))`` ;
- le noyau de Squared Exponential utilise ``exp(-r**2 / 2)`` (coherent
  avec scikit-learn) ;
- Matern 3/2 : ``sigma^2 (1 + sqrt(3) r) exp(-sqrt(3) r)`` ;
- Matern 5/2 : ``sigma^2 (1 + sqrt(5) r + 5 r**2 / 3) exp(-sqrt(5) r)``.
"""
import numpy as np


class _ARDKernel:
    """Base commune aux noyaux ARD (parametres publics ``lengthscales`` et ``variance``)."""

    def __init__(self, lengthscales, variance=1.0):
        """
        Parameters
        ----------
        lengthscales : array-like, shape (d,)
            Un lengthscale par dimension d'entree.
        variance : float, default 1.0
            Variance du signal (amplitude du noyau).
        """
        ls = np.atleast_1d(np.asarray(lengthscales, dtype=float)).copy()
        if np.any(ls <= 0) or not np.all(np.isfinite(ls)):
            raise ValueError(f"lengthscales must be strictly positive and finite, got {ls}")
        if not np.isfinite(variance) or variance < 0:
            raise ValueError(f"variance must be a finite non-negative number, got {variance}")
        self.lengthscales = ls
        self.variance = float(variance)
        self.n_dims = ls.size

    # -- methodes internes -------------------------------------------------
    def _check_input(self, X):
        X = np.atleast_2d(np.asarray(X, dtype=float))
        if X.ndim != 2 or X.shape[1] != self.n_dims:
            raise ValueError(
                f"expected an array of shape (n, {self.n_dims}), got {X.shape}"
            )
        return X

    def _r2(self, X1, X2):
        """Matrice des distances normalisees au carre, forme (n1, n2)."""
        D = (X1[:, None, :] - X2[None, :, :]) / self.lengthscales
        return np.einsum("ijk,ijk->ij", D, D)

    # -- API publique ------------------------------------------------------
    def K(self, X1, X2):
        """Matrice de covariance entre X1 (n1, d) et X2 (n2, d) : resultat (n1, n2)."""
        X1 = self._check_input(X1)
        X2 = self._check_input(X2)
        return self._kernel_of_r2(self._r2(X1, X2))

    def diag(self, X):
        """Valeurs diagonales du noyau (ici constantes : ``variance * ones(n)``)."""
        X = self._check_input(X)
        return np.full(X.shape[0], self.variance, dtype=float)

    # a specialiser
    def _kernel_of_r2(self, r2):
        raise NotImplementedError


class SquaredExponentialARD(_ARDKernel):
    """Noyau Squared Exponential (RBF) avec ARD : ``sigma^2 exp(-r^2 / 2)``."""

    def _kernel_of_r2(self, r2):
        return self.variance * np.exp(-0.5 * r2)


class Matern32ARD(_ARDKernel):
    """Matern 3/2 avec ARD : ``sigma^2 (1 + sqrt(3) r) exp(-sqrt(3) r)``."""

    def _kernel_of_r2(self, r2):
        r = np.sqrt(r2)
        return self.variance * (1.0 + np.sqrt(3.0) * r) * np.exp(-np.sqrt(3.0) * r)


class Matern52ARD(_ARDKernel):
    """Matern 5/2 avec ARD : ``sigma^2 (1 + sqrt(5) r + 5 r^2 / 3) exp(-sqrt(5) r)``."""

    def _kernel_of_r2(self, r2):
        r = np.sqrt(r2)
        return self.variance * (1.0 + np.sqrt(5.0) * r + (5.0 / 3.0) * r2) * np.exp(-np.sqrt(5.0) * r)
