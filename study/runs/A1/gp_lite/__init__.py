"""gp_lite : petit paquet de regression par processus gaussien (NumPy + SciPy uniquement)."""

from gp_lite.kernels import Matern32ARD, Matern52ARD, SquaredExponentialARD
from gp_lite.gp import GPRegressor

__all__ = [
    "SquaredExponentialARD",
    "Matern32ARD",
    "Matern52ARD",
    "GPRegressor",
]
