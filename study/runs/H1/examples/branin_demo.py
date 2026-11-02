"""Demonstration : regression par processus gaussien sur la fonction de Branin.

Entraîne un ``gp_lite.GPRegressor`` (noyau Matérn 5/2 ARD) sur 30 points
LHS de la fonction de Branin, prédit sur une grille de test et écrit la
figure dans ``examples/branin_demo.png`` (backend Agg).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from gp_lite import GPRegressor, Matern52ARD


def branin(X):
    """Fonction de Branin (2D)."""
    x1, x2 = X[:, 0], X[:, 1]
    return ((x2 - 5.1 / (4 * np.pi**2) * x1**2 + 5 / np.pi * x1 - 6) ** 2
            + 10 * (1 - 1 / (8 * np.pi)) * np.cos(x1) + 10)


def lhs(n, d, seed=0):
    """Plan factoriel latin (Latin Hypercube Sampling) simple avec NumPy."""
    rng = np.random.default_rng(seed)
    pts = np.empty((n, d))
    for j in range(d):
        idx = rng.permutation(np.arange(n))
        pts[:, j] = (idx + rng.uniform(size=n)) / n
    return pts


def main():
    npoints = 30
    X = lhs(npoints, 2, seed=0) * np.array([15.0, 15.0]) + np.array([-5.0, 0.0])
    y = branin(X)

    gp = GPRegressor(Matern52ARD(lengthscales=np.ones(2), variance=1.0),
                     noise=1e-6, normalize_y=True)
    gp.fit(X, y)

    # Grille de test 2D
    g1 = np.linspace(-5, 10, 40)
    g2 = np.linspace(0, 15, 40)
    G1, G2 = np.meshgrid(g1, g2)
    Xg = np.column_stack([G1.ravel(), G2.ravel()])
    mean, std = gp.predict(Xg, return_std=True)
    rmse = float(np.sqrt(np.mean((mean.ravel() - branin(Xg)) ** 2)))
    print("RMSE sur la grille 40x40 :", rmse)
    print("Hyperparamètres : lengthscales =", gp.kernel.lengthscales,
          "variance =", gp.kernel.variance, "noise =", gp.noise)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.contourf(G1, G2, mean.reshape(40, 40), levels=30, cmap="viridis")
    ax.scatter(X[:, 0], X[:, 1], c="red", s=25, edgecolors="black",
               linewidths=0.5, label="points d'entraînement")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_title("GP sur la fonction de Branin (moyenne prédite)")
    ax.legend(loc="upper right")
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "branin_demo.png")
    fig.savefig(out, dpi=120)
    print("Figure écrite dans :", out)


if __name__ == "__main__":
    main()
