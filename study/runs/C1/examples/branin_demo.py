"""Demonstration gp_lite : surrogate par processus gaussien de Branin.

Entraine un GP (Matern 5/2 ARD) sur 30 points LHS de la fonction de Branin,
puis trace la moyenne predite et les points d'entrainement dans
``examples/branin_demo.png``.

Lancement :
    python examples/branin_demo.py
"""

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Permet d'importer gp_lite depuis la racine du projet.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from gp_lite import GPRegressor, Matern52ARD


def branin(x):
    """Fonction de Branin 2D."""
    x1, x2 = x[..., 0], x[..., 1]
    return (
        (x2 - 5.1 / (4 * np.pi ** 2) * x1 ** 2) ** 2
        + 5.1 / (4 * np.pi) * x1 ** 2
        - 10.0 / np.pi * x1
        + 10.0
    )


def lhs(n, d, seed=0):
    """Latin hypercube simple de n points dans [0, 1]^d (pur NumPy)."""
    rng = np.random.default_rng(seed)
    pts = np.zeros((n, d))
    for j in range(d):
        edges = (np.arange(n) + rng.random(n)) / n
        order = rng.permutation(n)
        pts[order, j] = edges
    return pts


def main():
    # Echantillonnage d'entrainement et de test (domaine de Branin).
    X = lhs(30, 2, seed=1)
    X[:, 0] *= 10.0
    X[:, 1] *= 12.0
    y = branin(X)

    gp = GPRegressor(Matern52ARD(np.array([1.0, 1.0])), noise=1e-6,
                     normalize_y=True)
    gp.fit(X, y, optimize=True)
    print("Hyperparametres ajustes :")
    print("  lengthscales =", gp.kernel.lengthscales)
    print("  variance     =", gp.kernel.variance)
    print("  noise        =", gp.noise)
    print("log-vraisemblance marginale =", gp.log_marginal_likelihood())

    # Grille de test pour le dessin et le score.
    g1 = np.linspace(0.0, 10.0, 100)
    g2 = np.linspace(0.0, 12.0, 120)
    G1, G2 = np.meshgrid(g1, g2)
    G = np.column_stack([G1.ravel(), G2.ravel()])
    mean, std = gp.predict(G, return_std=True)

    X_test = lhs(200, 2, seed=2)
    X_test[:, 0] *= 10.0
    X_test[:, 1] *= 12.0
    y_test = branin(X_test)
    rmse_rel = np.sqrt(np.mean((gp.predict(X_test) - y_test) ** 2)) / np.std(y_test)
    print("RMSE normalisee (200 points test) =", rmse_rel)

    fig, ax = plt.subplots(figsize=(7, 5))
    cont = ax.contourf(G1, G2, mean.reshape(G1.shape), levels=25,
                       cmap="viridis")
    ax.plot(X[:, 0], X[:, 1], "r.", markersize=14,
            label="30 points d'entrainement (LHS)")
    fig.colorbar(cont, ax=ax, label="moyenne predite")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_title("gp_lite : moyenne predite de Branin (Matern 5/2 ARD)")
    ax.legend(loc="upper right")
    out = os.path.join(os.path.dirname(__file__), "branin_demo.png")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print("Figure sauvegardee :", out)


if __name__ == "__main__":
    main()
