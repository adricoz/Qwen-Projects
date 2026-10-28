"""Démo gp_lite : GP sur la fonction de Branin, export d'un graphique PNG.

Lancement :
    python examples/branin_demo.py
Écrit examples/branin_demo.png (moyenne prédite + points d'entraînement).
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")  # backend sans affichage
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gp_lite import GPRegressor, SquaredExponentialARD  # noqa: E402


def branin(x1, x2):
    """Fonction de Branin 2D (x1 dans [0, 10], x2 dans [0, 12])."""
    a, b, c, r, t = 1.0, 5.1 / (4 * np.pi ** 2), 5 / np.pi, 6.0, 10 / (3 * np.pi)
    return a * (x2 - b * x1 ** 2 + c * x1 - r) ** 2 + r * (1 - t) * np.cos(x1) + 12.0 * np.pi


def lhs(n, d, seed=0):
    """Lattice hyper-carré simple en [0, 1)^d (permutation par dimension)."""
    rng = np.random.default_rng(seed)
    grid = np.linspace(0.5, n - 0.5, n) / n
    out = np.empty((n, d))
    for j in range(d):
        out[:, j] = grid[rng.permutation(n)]
    return out


def main():
    domain = np.array([10.0, 12.0])
    X = lhs(30, 2, seed=42) * domain
    y = branin(X[:, 0], X[:, 1])

    # entraînement avec optimisation des hyperparamètres
    gp = GPRegressor(SquaredExponentialARD([3.0, 3.0]), noise=1e-3)
    gp.fit(X, y)
    print(f"hyperparamètres ajustés : lengthscales={np.round(gp.kernel.lengthscales, 4)}, "
          f"variance={gp.kernel.variance:.4g}, noise={gp.noise:.4g}")
    print(f"log-vraisemblance marginale : {gp.log_marginal_likelihood():.3f}")

    # grille de prédiction 100x100
    gx = np.linspace(0.0, 10.0, 100)
    gy = np.linspace(0.0, 12.0, 100)
    G1, G2 = np.meshgrid(gx, gy)
    G = np.column_stack([G1.ravel(), G2.ravel()])
    mean, std = gp.predict(G, return_std=True)

    # RMSE normalisée sur 200 points de test
    Xt = lhs(200, 2, seed=99) * domain
    yt = branin(Xt[:, 0], Xt[:, 1])
    rmse = np.sqrt(np.mean((gp.predict(Xt) - yt) ** 2))
    print(f"RMSE = {rmse:.3f}   RMSE normalisée = {rmse / np.std(yt):.4f}")

    fig, ax = plt.subplots(figsize=(7, 6), dpi=150)
    ax.contourf(G1, G2, mean.reshape(100, 100), levels=25, cmap="viridis")
    ax.contour(G1, G2, mean.reshape(100, 100), levels=8, colors="k", linewidths=0.5)
    ax.scatter(X[:, 0], X[:, 1], c="red", s=35, edgecolor="k", zorder=5,
               label="points d'entraînement")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_title("GP sur la fonction de Branin (moyenne prédite)")
    ax.legend(loc="upper right")
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "branin_demo.png")
    fig.savefig(out, bbox_inches="tight")
    print(f"graphique écrit : {out}")


if __name__ == "__main__":
    main()
