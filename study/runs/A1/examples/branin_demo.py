"""Demonstration : regression par processus gaussien sur la fonction de Branin.

Entraîne un ``GPRegressor`` (noyau SE-ARD) sur 30 points LHS de Branin,
prédit sur une grille de 200 points et écrit ``examples/branin_demo.png``
(moyenne prédite + points d'apprentissage).

Lancement : ``python examples/branin_demo.py``
"""

import os
import sys

import matplotlib

matplotlib.use("Agg")  # backend sans fenetre
import matplotlib.pyplot as plt
import numpy as np

# rend la racine du projet importable sans installation
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gp_lite import GPRegressor, SquaredExponentialARD  # noqa: E402


def branin(x1, x2):
    """Fonction de Branin 2D (domaine [0, 50] x [0, 10])."""
    a, b, c = 1.0, 5.1 / (4.0 * np.pi**2), 5.0 / np.pi
    d, e, f = 0.0, 6.0, 10.0
    return (a * (x2 - b * x1**2 + c * x1 - d) ** 2 + e * (np.cos(x1) + 1.0)) * f


def lhs(n, d, seed=1):
    """Low-discrepancy sampling simple par Latin hypercube (1 sample par case)."""
    rng = np.random.default_rng(seed)
    pts = np.empty((n, d))
    for j in range(d):
        edges = rng.permutation(n) / n
        pts[:, j] = np.clip(edges + rng.uniform(size=n) * (1.0 / n), 0.0, 1.0)
    return pts


def main():
    # 30 points d'apprentissage (LHS) et 200 points de test
    X = np.column_stack([lhs(30, 2)[:, 0] * 50.0, lhs(30, 2)[:, 1] * 10.0])
    y = branin(X[:, 0], X[:, 1])
    Xte = np.column_stack([lhs(200, 2, seed=2)[:, 0] * 50.0,
                           lhs(200, 2, seed=2)[:, 1] * 10.0])
    yte = branin(Xte[:, 0], Xte[:, 1])

    gp = GPRegressor(SquaredExponentialARD(np.array([10.0, 2.0])), noise=1e-6)
    gp.fit(X, y)

    mean, std = gp.predict(Xte, return_std=True)
    rmse = np.sqrt(np.mean((mean - yte) ** 2))
    norm = rmse / np.sqrt(np.mean(yte**2))
    print(f"Hyperparametres : l = {gp.kernel.lengthscales}, "
          f"variance = {gp.kernel.variance:.4f}, bruit = {gp.noise:.2e}")
    print(f"Log-vraisemblance marginale : {gp.log_marginal_likelihood():.2f}")
    print(f"RMSE test : {rmse:.4f}  (normalisee : {norm:.4f})")

    fig, ax = plt.subplots(figsize=(8, 5))
    sc = ax.scatter(X[:, 0], X[:, 1], c="red", s=40,
                    label="points d'apprentissage", zorder=3)
    pc = ax.scatter(Xte[:, 0], Xte[:, 1], c=mean, cmap="viridis",
                    s=14, alpha=0.8, label="moyenne prédite", zorder=2)
    cb = fig.colorbar(pc, ax=ax)
    cb.set_label("valeur prédite de Branin")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_title("gp_lite : moyenne prédite GP sur la fonction de Branin")
    ax.legend(loc="upper right", handles=[sc, pc])
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "branin_demo.png")
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"Figure écrite : {out}")


if __name__ == "__main__":
    main()
