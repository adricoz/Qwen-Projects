"""gp_lite : petit paquet autonome de regression par processus gaussien.

Seule dependance : NumPy + SciPy (scikit-learn est utilise uniquement par les tests).
"""
from .kernels import SquaredExponentialARD, Matern32ARD, Matern52ARD
from .gp import GPRegressor

__all__ = [
    "SquaredExponentialARD",
    "Matern32ARD",
    "Matern52ARD",
    "GPRegressor",
]
