"""gp_lite: petit paquet autonome de régression par processus gaussien.

Dépendances: NumPy + SciPy uniquement (scikit-learn est utilisé par les
tests seulement).
"""
from .kernels import SquaredExponentialARD, Matern32ARD, Matern52ARD
from .gp import GPRegressor

__all__ = [
    "SquaredExponentialARD",
    "Matern32ARD",
    "Matern52ARD",
    "GPRegressor",
]
__version__ = "0.1.0"
