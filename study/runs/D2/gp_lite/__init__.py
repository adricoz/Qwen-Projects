"""gp_lite : régression par processus gaussien (NumPy + SciPy uniquement)."""
from .kernels import SquaredExponentialARD, Matern32ARD, Matern52ARD
from .gp import GPRegressor

__all__ = ["SquaredExponentialARD", "Matern32ARD", "Matern52ARD", "GPRegressor"]
__version__ = "0.1.0"
