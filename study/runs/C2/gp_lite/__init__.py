"""gp_lite : petit paquet de régression par processus gaussien (NumPy + SciPy)."""
from .kernels import (
    Kernel,
    SquaredExponentialARD,
    Matern32ARD,
    Matern52ARD,
)
from .gp import GPRegressor

__all__ = [
    "Kernel",
    "SquaredExponentialARD",
    "Matern32ARD",
    "Matern52ARD",
    "GPRegressor",
]
