"""Démo gp_lite sur la fonction de Branin : écrit examples/branin_demo.png."""
import os
import sys
import matplotlib
matplotlib.use("Agg")  # backend sans fenêtre
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gp_lite import GPRegressor, SquaredExponentialARD


def branin(x1, x2):
    """Fonction de Branin 2D."""
    return ((x2 - 5.1 / (4 * np.pi**2) * x1**2 + 5 / np.pi * x1 - 6)**2
            + 10 * (1 - 1 / (8 * np.pi)) * np.cos(x1) + 10)


def lhs(n, d, seed=0):
    """Latin hypercube simple dans [0, 1]^d."""
    rng = np.random.default_rng(seed)
    u = rng.uniform(size=(n, d))
    P = np.column_stack([rng.permutation(n) for _ in range(d)])
    return (P - 0.5 + u) / n


def main():
    # Échantillonnage LHS 2D dans le domaine de Branin
    X = np.column_stack([
        lhs(30, 2, seed=1)[:, 0] * 15.0 - 5.0,
        lhs(30, 2, seed=1)[:, 1] * 15.0,
    ])
    y = branin(X[:, 0], X[:, 1])

    # Ajustement du GP (optimisation ML, 3 redémarrages)
    gp = GPRegressor(SquaredExponentialARD(np.ones(2), 1.0), noise=1e-6)
    gp.fit(X, y)
    print("lengthscales:", np.round(gp.kernel.lengthscales, 3))
    print("variance    :", round(gp.kernel.variance, 3))
    print("noise       :", f"{gp.noise:.3e}")
    print("LML         :", round(gp.log_marginal_likelihood(), 3))

    # Grille de prédiction
    g1 = np.linspace(-5, 10, 80)
    g2 = np.linspace(0, 15, 80)
    G1, G2 = np.meshgrid(g1, g2)
    Xg = np.column_stack([G1.ravel(), G2.ravel()])
    mean, std = gp.predict(Xg, return_std=True)

    # RMSE normalisée sur 200 points de test
    Xt = np.column_stack([
        lhs(200, 2, seed=2)[:, 0] * 15.0 - 5.0,
        lhs(200, 2, seed=2)[:, 1] * 15.0,
    ])
    yt = branin(Xt[:, 0], Xt[:, 1])
    nrmse = np.sqrt(np.mean((np.ravel(gp.predict(Xt)) - yt) ** 2)) / yt.std()
    print(f"RMSE normalisée (200 pts test) : {nrmse:.4f}")

    fig, ax = plt.subplots(figsize=(8, 6))
    cs = ax.contourf(G1, G2, mean.reshape(G1.shape), levels=25, cmap="viridis")
    ax.contour(G1, G2, mean.reshape(G1.shape), levels=10, colors="k", linewidths=0.4, alpha=0.4)
    ax.scatter(X[:, 0], X[:, 1], c="red", edgecolors="k", s=35, zorder=5, label="points d'apprentissage (LHS, n=30)")
    fig.colorbar(cs, label="moyenne prédite")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_title("gp_lite — Branin : moyenne prédictive (GP, SE ARD)")
    ax.legend(loc="upper left")
    fig.tight_layout()
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "branin_demo.png")
    fig.savefig(out, dpi=150)
    print("figure écrite :", out)


if __name__ == "__main__":
    main()
