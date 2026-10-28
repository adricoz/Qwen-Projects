"""Demonstration `gp_lite` sur la fonction de Branin.

Utilisation :
    python examples/branin_demo.py

S'entraine un GP (Matern 5/2 ARD) sur 30 points LHS et ecris
`examples/branin_demo.png` (contours de la moyenne predite + points d'entrainement).
"""
import os
import sys

import matplotlib

matplotlib.use("Agg")  # sans fenetre
import matplotlib.pyplot as plt
import numpy as np

# autorise l'execution directe depuis la racine du projet
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gp_lite import GPRegressor, Matern52ARD  # noqa: E402


def branin(x):
    """Fonction de Branin 2D (x[:, 0] dans [-5, 10], x[:, 1] dans [0, 15])."""
    a, b = x[:, 0], x[:, 1]
    return (b - 5.1 / (4 * np.pi**2) * a**2 + 5 / np.pi * a - 6)**2 \
        + 10 * (1 - 1 / (8 * np.pi)) * np.cos(a) + 10


def lhs_points(n, d, seed=0):
    """n points LHS dans [0, 1]^d (un par cellule par dimension)."""
    rng = np.random.default_rng(seed)
    P = np.empty((n, d))
    for j in range(d):
        P[:, j] = (rng.permutation(n) + rng.random(n)) / n
    return P


def main():
    # points d'entraiement dans le domaine de Branin
    X = lhs_points(30, 2, seed=7)
    X = np.column_stack([X[:, 0] * 15.0 - 5.0, X[:, 1] * 15.0])
    y = branin(X)

    kernel = Matern52ARD([0.5, 0.5], variance=1.0)
    gp = GPRegressor(kernel, noise=1e-6)
    gp.fit(X, y)
    print("hyperparametres estimes:")
    print("  lengthscales =", np.round(gp.kernel.lengthscales, 4))
    print("  variance     =", round(gp.kernel.variance, 4))
    print("  bruit        =", round(gp.noise_, 8))
    print("log-vraisemblance marginale =", round(gp.log_marginal_likelihood(), 3))

    # grille de test + RMSE normalisee
    Xt = lhs_points(200, 2, seed=11)
    Xt = np.column_stack([Xt[:, 0] * 15.0 - 5.0, Xt[:, 1] * 15.0])
    yt = branin(Xt)
    mean_t = gp.predict(Xt)
    rmse = float(np.sqrt(np.mean((mean_t - yt) ** 2)))
    print(f"RMSE (200 points test) = {rmse:.3f}")
    print(f"RMSE normalisee        = {rmse / np.std(yt):.4f}")

    # figure : contours de la moyenne + points d'entraiement
    gx = np.linspace(-5.0, 10.0, 80)
    gy = np.linspace(0.0, 15.0, 80)
    GX, GY = np.meshgrid(gx, gy)
    G = gp.predict(np.column_stack([GX.ravel(), GY.ravel()])).reshape(GX.shape)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.contourf(GX, GY, G, levels=20, cmap="viridis")
    cs = ax.contour(GX, GY, G, levels=8, colors="w", linewidths=0.6)
    ax.clabel(cs, fmt="%.0f", fontsize=7)
    ax.scatter(X[:, 0], X[:, 1], s=28, c="red", edgecolor="white",
               label="30 points d'entrainement (LHS)", zorder=3)
    ax.set_title("gp_lite — Branin : moyenne predite par GP (Matern 5/2 ARD)")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "branin_demo.png")
    fig.savefig(out, dpi=150)
    print("figure ecrite :", out)


if __name__ == "__main__":
    main()
