"""Demonstration gp_lite sur la fonction de Branin (2D).

Entraîne un GPRegressor (Matern 5/2 ARD) sur 30 points LHS puis trace la
moyenne prédite et les points d'apprentissage dans ``branin_demo.png``.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from gp_lite import GPRegressor, Matern52ARD


def branin(x):
    """Fonction de Branin 2D sur [0, 1]^2."""
    x1, x2 = x[:, 0], x[:, 1]
    return ((x2 - 5.0 * x1 ** 2 + 5.0 * x1 - 5.0) ** 2
            + 10.0 * (np.sin(6.0 * x1) + np.sin(2.0 * x2)) + 10.0)


def lhs(n, d, seed=0):
    """Latin hypercube simple (une permutation aleatoire par dimension)."""
    r = np.random.default_rng(seed)
    pts = np.empty((n, d))
    for j in range(d):
        perm = r.permutation(n)
        u = r.uniform(size=n)
        pts[perm, j] = (np.arange(n) + u) / n
    return pts


def main():
    # -- apprentissage
    Xtr = lhs(30, 2, seed=3)
    ytr = branin(Xtr)
    gp = GPRegressor(Matern52ARD([0.3, 0.3])).fit(Xtr, ytr)

    # -- prediction sur une grille
    g = np.linspace(0.0, 1.0, 80)
    G1, G2 = np.meshgrid(g, g)
    Xg = np.column_stack([G1.ravel(), G2.ravel()])
    mean = gp.predict(Xg).reshape(G1.shape)
    yte = branin(Xg).reshape(G1.shape)
    rmse = float(np.sqrt(np.mean((mean - yte) ** 2)))
    rmse_norm = rmse / float(np.std(ytr))

    # -- figure
    fig, ax = plt.subplots(figsize=(6.5, 5.0))
    cf = ax.contourf(G1, G2, mean, levels=24, cmap="viridis")
    fig.colorbar(cf, ax=ax, label="moyenne predite")
    ax.scatter(Xtr[:, 0], Xtr[:, 1], c="white", edgecolors="red",
               s=45, zorder=5, label="points d'apprentissage (LHS)")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_title("GP (Matérn 5/2) sur la fonction de Branin")
    ax.legend(loc="upper right")
    fig.tight_layout()
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "branin_demo.png")
    fig.savefig(out, dpi=150)
    print(f"RMSE        : {rmse:.4f}")
    print(f"RMSE normal.: {rmse_norm:.4f}")
    print(f"figure      : {out}")


if __name__ == "__main__":
    main()
