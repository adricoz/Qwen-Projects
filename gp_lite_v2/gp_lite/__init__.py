"""gp_lite: petit paquet de regression par processus gaussien (NumPy + SciPy)."""
from .kernels import SquaredExponentialARD, Matern32ARD, Matern52ARD
from .gp import GPRegressor

__all__ = ["SquaredExponentialARD", "Matern32ARD", "Matern52ARD", "GPRegressor"]
__version__ = "0.1.0"
