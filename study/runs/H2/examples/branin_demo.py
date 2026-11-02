"""Demonstration : regression par processus gaussien sur la fonction de Branin.

Entraine un GP (noyau SE ARD, hyperparametres optimises) sur 30 points LHS,
puis ecrit dans ``examples/branin_demo.png`` la moyenne predite sur une grille
2D avec les points d'entrainement en surimpression.
"""
import os
import sys

# permet d'executer le script depuis n'importe quel dossier
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from gp_lite import GPRegressor, SquaredExponentialARD


def branin(x1, x2):
    """Fonction de Branin (2D), version definie sur [0, 1]^2."""
    a, b, c = 1.0, 5.1 / (4.0 * np.pi ** 2), 5.0 / np.pi
    return (a * (x2 - b * x1 ** 2 + c * x1 - 5.0) ** 2 + 10.0 *
            (np.cos(x1) + 10.0)) * np.pi


def lhs(n, d, seed=0):
    """Latin Hyper Sampling simple : n points, d dimensions (numpy uniquement)."""
    rng = np.random.RandomState(seed)
    U = np.empty((n, d))
    for j in range(d):
        U[:, j] = (np.arange(n) + rng.rand(n)) / n
        U[:, j] = U[rng.permutation(n), j]
    return U


def main():
    # donnees d'entrainement : 30 points LHS
    X = lhs(30, 2)
    y = branin(X[:, 0], X[:, 1])

    gp = GPRegressor(SquaredExponentialARD([0.5, 0.5]))
    gp.fit(X, y, optimize=True)
    print("Hyperparametres ajustes :")
    print("  lengthscales =", gp.kernel.lengthscales)
    print("  variance     =", gp.kernel.variance)
    print("  noise        =", gp.noise)

    # grille de prediction
    g1, g2 = np.meshgrid(np.linspace(0, 1, 80), np.linspace(0, 1, 80))
    Xg = np.column_stack([g1.ravel(), g2.ravel()])
    mean, std = gp.predict(Xg, return_std=True)
    Z = mean.reshape(g1.shape)

    yt = branin(g1, g2)
    rmse = np.sqrt(np.mean((mean - yt.ravel()) ** 2))
    print("RMSE sur la grille : %.4f (std = %.4f)" % (rmse, yt.std()))

    fig, ax = plt.subplots(figsize=(7, 5))
    cont = ax.contourf(g1, g2, Z, levels=40, cmap="viridis")
    fig.colorbar(cont, ax=ax, label="moyenne predite")
    ax.scatter(X[:, 0], X[:, 1], c="red", edgecolor="k", s=30,
               label="points d'entrainement (LHS)")
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_title("GP sur la fonction de Branin (30 points LHS)")
    ax.legend(loc="upper right")
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "branin_demo.png")
    fig.savefig(out, dpi=120)
    print("Figure ecrite dans", out)


if __name__ == "__main__":
    main()
