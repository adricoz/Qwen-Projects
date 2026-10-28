"""Paquet gp_lite : régression par processus gaussien (NumPy + SciPy)."""
from .kernels import SquaredExponentialARD, Matern32ARD, Matern52ARD
from .gp import GPRegressor

__all__ = ["SquaredExponentialARD", "Matern32ARD", "Matern52ARD", "GPRegressor"]
