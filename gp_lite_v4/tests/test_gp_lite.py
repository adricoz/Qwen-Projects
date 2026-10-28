"""Tests du paquet gp_lite."""
import numpy as np
from gp_lite import GPRegressor, Matern32ARD, Matern52ARD, SquaredExponentialARD
from gp_lite.gp import _nll_and_alpha


def lhs(n, d, seed=0):
    """Echantillonnage Latin Hypercube simple (n points en [0, 1)^d)."""
    rng = np.random.default_rng(seed)
    x = np.empty((n, d))
    for j in range(d):
        x[:, j] = (np.arange(n) + rng.uniform(size=n)) / n
    return x


def branin(x):
    """Fonction de Branin (2D), entrees dans [0, 1]."""
    b, c = 5.1 / (4 * np.pi**2), 5.0 / (4 * np.pi)
    x1, x2 = 25.0 * x[:, 0], 5.0 * x[:, 1]
    return (x2 - b * x1**2 + c * x1 - 5.0)**2 + 10.0 * (np.sin(x1) + 10.0) + 10


def _fit_fixed(gp, X, y):
    """Conditionne un GP, hyperparametres FIXES (pas d'optimisation)."""
    X = np.atleast_2d(np.asarray(X, dtype=float))
    gp.X, gp.y = X, np.asarray(y, dtype=float).ravel()
    gp.y_mean_, gp.y_std_ = float(np.mean(gp.y)), 1.0
    gp.K_ = gp.kernel.K(gp.X, gp.X)
    _, gp.alpha_ = _nll_and_alpha(gp.K_, gp.noise, gp.y, 1e-10)
    gp._L_ = None
    return gp


def test_kernel_symmetry_and_diag():
    rng = np.random.default_rng(0)
    X1 = rng.uniform(0, 1, size=(7, 3))
    X2 = rng.uniform