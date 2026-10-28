"""gp_lite : petit paquet de regression par processus gaussien (NumPy/SciPy)."""

from .kernels import Matern32ARD, Matern52ARD, SquaredExponentialARD
from .gp import GPRegressor

__all__ = [
    "GPRegressor",
    "SquaredExponentialARD",
    "Matern32ARD",
    "Matern52ARD",
]
