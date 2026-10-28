"""gp_lite: petit paquet de regression par processus gaussien (NumPy + SciPy)."""
from gp_lite.kernels import Matern32ARD, Matern52ARD, SquaredExponentialARD
from gp_lite.gp import GPRegressor

__all__ = [
    "SquaredExponentialARD",
    "Matern32ARD",
    "Matern52ARD",
    "GPRegressor",
]
__version__ = "0.1.0"
