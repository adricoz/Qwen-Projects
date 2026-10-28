"""gp_lite : petit paquet de regression par processus gaussien (NumPy + SciPy)."""
from gp_lite.gp import GPRegressor
from gp_lite.kernels import Matern32ARD, Matern52ARD, SquaredExponentialARD

__all__ = ["GPRegressor", "Matern32ARD", "Matern52ARD", "SquaredExponentialARD"]
