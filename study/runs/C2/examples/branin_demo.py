"""Démo gp_lite : régression par processus gaussien sur la fonction de Branin.

Entraîne un GP (noyau squared-exponential ARD, hyperparamètres optimisés)
sur 30 points LHS, puis écrit `examples/branin_demo.png` : grille de la
moyenne prédite (couleurs), points d'apprentissage et courbe de la fonction
vraie.
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gp_lite import GPRegressor, SquaredExponentialARD
from gp_lite.lhs import lhs


def branin2d(x):
    """Fonction de Branin (2D) sur [0, 1] x [0, 1]."""
    x1, x2 = x[..., 0], x[..., 1]
    a, b = 1.0, 5.1 / (4.0 * np.pi ** 2)
    c, r = 5.0 / np.pi, 6.0
    s, t = 10.0, 1.0 / (8.0 * np.pi)
    return a * (x2 - b * x1 ** 2 + c * x1 - r) ** 2 + s * (1.0 - t) * np.cos(x1) + s


def main():
    # données : 30 points LHS + 200 points test
    X = lhs(30, 2, seed=3)
    y = branin2d(X)
    X_test = lhs(200, 2, seed=123)
    y_true = branin2d(X_test)

    # apprentissage avec optimisation des hyperparamètres
    gp = GPRegressor(SquaredExponentialARD(lengthscales=[0.3, 0.3], variance=1.0),
                     noise=1e-6)
    gp.fit(X, y, optimize=True)
    print(f"hyperparamètres ajustés : lengthscales={gp.kernel.lengthscales}, "
          f"variance={gp.kernel.variance:.4g}, noise={gp.noise:.3g}")
    print(f"log-vraisemblance marginale : {gp.log_marginal_likelihood():.3f}")

    y_pred = gp.predict(X_test)
    rmse = np.sqrt(np.mean((y_pred - y_true) ** 2))
    print(f"RMSE : {rmse:.4f}   (normalisée : {rmse / np.std(y_true):.4f})")

    # grille pour la visualisation
    gx = np.linspace(0.0, 1.0, 120)
    G = np.stack(np.meshgrid(gx, gx, indexing="ij"), axis=-1).reshape(-1, 2)
    # grid[i, j] = valeur en (x1=gx[i], x2=gx[j]) ; pour imshow (col=x1, row=x2)
    # on transpose et on place l'origine en bas
    mean_grid = gp.predict(G).reshape(gx.size, gx.size).T
    true_grid = branin2d(G).reshape(gx.size, gx.size).T

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    ax = axes[0]
    im = ax.imshow(mean_grid, extent=[0, 1, 0, 1], origin="lower",
                   aspect="auto", cmap="viridis")
    ax.plot(X[:, 0], X[:, 1], "r.", ms=10, label="points d'apprentissage")
    ax.set_title("Moyenne prédite (GP) + points")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    fig.colorbar(im, ax=ax)
    ax = axes[1]
    im = ax.imshow(true_grid, extent=[0, 1, 0, 1], origin="lower",
                   aspect="auto", cmap="viridis")
    ax.set_title("Fonction de Branin (vraie)")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    fig.colorbar(im, ax=ax)
    ax = axes[2]
    ax.plot(y_true, y_pred, "b.", alpha=0.5, label="GP vs vraie")
    lim = min(y_true.min(), y_pred.min())
    ax.plot([lim, max(y_true.max(), y_pred.max())],
            [lim, max(y_true.max(), y_pred.max())], "k--")
    ax.set_title(f"RMSE normalisée = {rmse / np.std(y_true):.3f}")
    ax.set_xlabel("vraie")
    ax.set_ylabel("prédite")
    ax.legend()
    fig.tight_layout()
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "branin_demo.png")
    fig.savefig(out, dpi=120)
    print(f"figure écrite : {out}")


if __name__ == "__main__":
    main()
