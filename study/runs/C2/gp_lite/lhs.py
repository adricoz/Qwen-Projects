"""LHS (Latin hypercube sampling) minimaliste avec numpy.

Mise en cache du module pour réutilisation par les tests et les exemples.
"""
import numpy as np


def lhs(n, d, seed=None):
    """Génère un hypercube latin (LHS) de n points dans [0, 1]^d.

    Parameters
    ----------
    n : int
        Nombre de points.
    d : int
        Dimension de l'espace.
    seed : int or None
        Graine du générateur pseudo-aléatoire (pour la reproductibilité).

    Returns
    -------
    ndarray, shape (n, d)
        Points de l'hypercube latin, dans [0, 1]^d.
    """
    rng = np.random.RandomState(seed)
    P = np.zeros((n, d))
    for j in range(d):
        perm = rng.permutation(n)
        P[:, j] = (perm + rng.uniform(size=n)) / n
    return P
