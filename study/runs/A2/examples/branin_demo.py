"""Démonstration gp_lite : régression par processus gaussien sur Branin.

Entraîne un GP (noyau RBF ARD) sur 30 points LHS de la fonction de Branin
2D et trace la moyenne prédite sur une grille de test.
Sortie : examples/branin_demo.png

Lancer : python examples/branin_demo.py
"""
import os
import sys

import numpy as np

# permet l'import de gp_lite quand on lance depuis le racine du repo
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from gp_lite import GPRegressor, SquaredExponentialARD  # noqa: E402

B = np.array([[0.0, 0.0], [30.0, 12.0]])


def branin(x):
    """Fonction de Branin 2D (x1 in [0, 30], x2 in [0, 12])."""
    x1, x2 = x[..., 0], x[..., 1]
    a, b = 1.0, 5.1 / (4.0 * np.pi**2)
    c, r = 5.0 / np.pi, 6.0
    t = 10.0 / (3.0 * np.pi)
    return (a * (x2 - b * x1**2 + c * x1 - r))**2 + r * (1 - t) * np.cos(x1) + r


def lhs(n, d, seed=0):
    """Latin hypercube simple avec numpy (n points dans [0, 1]^d)."""
    rng = np.random.default_rng(seed)
    pts = np.empty((n, d))
    for j in range(d):
        order = np.arange(n)[rng.permutation(n)]
        pts[:, j] = (order + rng.random(n)) / n
    return pts


def main():
    # --- apprentissage: 30 points LHS ---
    X_train = B[0] + lhs(30, 2, seed=1) * (B[1] - B[0])
    y_train = branin(X_train)

    # --- ajustement du GP (optimisation de la LVM) ---
    gp = GPRegressor(SquaredExponentialARD([5.0, 3.0], variance=100.0), noise=1.0)
    gp.fit(X_train, y_train, optimize=True)

    # --- grille de test: moyenne prédictive ---
    gx = np.linspace(B[0][0], B[1][0], 60)
    gy = np.linspace(B[0][1], B[1][1], 50)
    GX, GY = np.meshgrid(gx, gy)
    G = np.column_stack([GX.ravel(), GY.ravel()])
    mean = gp.predict(G).reshape(GX.shape)

    # --- métriques ---
    y_true = branin(G).reshape(GX.shape)
    rmse = np.sqrt(np.mean((mean - y_true) ** 2))
    print(f"RMSE (Branin, 30 pts LHS): {rmse:.3f}")
    print(f"RMSE normalisée (par std du test): {rmse / y_true.std():.4f}")
    print(f"Hyperparamètres finaux: l={np.round(gp.kernel.lengthscales, 3)}"
          f", variance={gp.kernel.variance:.3f}, noise={gp.noise:.3f}")
    print(f"log-vraisemblance marginale: {gp.log_marginal_likelihood():.3f}")

    # --- figure ---
    fig, ax = plt.subplots(figsize=(7, 5))
    cf = ax.contourf(GX, GY, mean, levels=40, cmap="viridis")
    fig.colorbar(cf, ax=ax, label="moyenne prédite")
    ax.scatter(X_train[:, 0], X_train[:, 1], c="white", edgecolors="black",
               s=30, label="points d'apprentissage (LHS)")
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_title("gp_lite: moyenne prédite sur la fonction de Branin")
    ax.legend(loc="upper right")
    fig.tight_layout()
    out = os.path.join(os.path.dirname(__file__), "branin_demo.png")
    fig.savefig(out, dpi=120)
    print(f"Figure écrite dans: {out}")


if __name__ == "__main__":
    main()
