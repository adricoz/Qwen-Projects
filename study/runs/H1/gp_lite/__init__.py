"""gp_lite: petit paquet autonome de regression par processus gaussien."""
from .kernels import Matern32ARD, Matern52ARD, SquaredExponentialARD
from .gp import GPRegressor

__all__ = [
    "SquaredExponentialARD",
    "Matern32ARD",
    "Matern52ARD",
    "GPRegressor",
]
__version__ = "0.1.0"
