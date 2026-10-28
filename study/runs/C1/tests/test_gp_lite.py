"""Tests de gp_lite (noyaux, GPRegressor, comparaison scikit-learn, Branin)."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from gp_lite import GPRegressor, Matern32ARD, Matern52ARD, SquaredExponentialARD

RNG = np.random.default_rng(42)


# ---------------------------------------------------------------------- #
# noyaux                                                                  #
# ---------------------------------------------------------------------- #
def test_kernels_symmetry_and_diagonal():
    """Symetrie K, diagonale = variance, Matern 5/2 a r=0 vaut sigma^2."""
    n, d = 6, 3
    X = RNG.random((n, d))
    kernels = [
        SquaredExponentialARD(np.array([0.5, 1.0, 2.0]), variance=0.7),
        Matern32ARD(np.array([0.5, 1.0, 2.0]), variance=0.7),
        Matern52ARD(np.array([0.5, 1.0, 2.0]), variance=0.7),
    ]
    for k in kernels:
        K = k.K(X, X)
        assert K.shape == (n, n)
        np.testing.assert_allclose(K, K.T, atol=1e-14)
        np.testing.assert_allclose(np.diag(K), np.full(n, 0.7), atol=1e-14)
        np.testing.assert_allclose(k.diag(X), np.full(n, 0.7))
        # symetrie entre deux jeux differents
        X2 = RNG.random((4, d))
        np.testing.assert_allclose(k.K(X2, X), k.K(X, X2).T, atol=1e-14)


def test_matern52_formula_at_r():
    """Valeur exacte du Matern 5/2 pour une distance donnee (d=1)."""
    k = Matern52ARD(1.0, variance=2.0)
    x1, x2 = np.array([[0.0]]), np.array([[0.3]])
    r = 0.3
    expected = 2.0 * (1 + np.sqrt(5) * r + 5 * r ** 2 / 3) * np.exp(-np.sqrt(5) * r)
    np.testing.assert_allclose(k.K(x1, x2)[0, 0], expected, rtol=1e-12)


# ---------------------------------------------------------------------- #
# interpolation / loo                                                     #
# ---------------------------------------------------------------------- #
def _small_dataset():
    X = RNG.random((12, 2))
    y = np.sin(X[:, 0] * 4.0) + 0.5 * X[:, 1]
    return X, y


def test_predict_interpolates_at_zero_noise():
    """A bruit nul, predict doit interpoler les points d'entrainement."""
    X, y = _small_dataset()
    gp = GPRegressor(Matern52ARD(np.array([0.5, 0.5])), noise=1e-12,
                     normalize_y=True)
    gp.fit(X, y, optimize=False)
    mean, std = gp.predict(X, return_std=True)
    np.testing.assert_allclose(mean, y, atol=1e-6)
    assert np.all(std < 1e-4)


def test_loo_matches_retraining():
    """loo (forme fermee) == recalcul explicite par n re-ajustements."""
    X, y = _small_dataset()
    kernel = Matern52ARD(np.array([0.6, 0.9]), variance=0.8)
    noise = 1e-3
    gp = GPRegressor(kernel, noise=noise, normalize_y=True)
    gp.fit(X, y, optimize=False)
    mean, var = gp.loo()

    # Normalisation du jeu complet (celle utilisee par gp.loo()).
    y_mean, y_std = gp._y_mean, gp._y_std
    for i in range(len(y)):
        mask = np.arange(len(y)) != i
        sub = GPRegressor(Matern52ARD(np.array([0.6, 0.9]), variance=0.8),
                          noise=noise, normalize_y=False)
        sub.fit(X[mask], (y[mask] - y_mean) / y_std, optimize=False)
        m, s = sub.predict(X[i], return_std=True)
        np.testing.assert_allclose(m, (mean[i] - y_mean) / y_std, atol=1e-6,
                                   err_msg=f"mean mismatch at {i}")
        # y_i = f_i + e_i avec e_i independant de y_-i :
        #   var(y_i|y_-i) = var(f_i|y_-i) + noise
        np.testing.assert_allclose(s ** 2 + noise, var[i] / y_std ** 2,
                                   atol=1e-6, err_msg=f"var mismatch at {i}")


# ---------------------------------------------------------------------- #
# comparaison scikit-learn (hyperparametres FIXES)                        #
# ---------------------------------------------------------------------- #
def test_matches_sklearn_fixed_hyperparameters():
    """Moyennes identiques a 1e-6 pres avec sklearn (RBF ARD + White, fixes)."""
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, WhiteKernel

    X = RNG.random((15, 2))
    y = np.cos(X[:, 0] * 5.0) + X[:, 1] ** 2
    X_test = RNG.random((20, 2))

    lengthscales = np.array([0.4, 1.3])
    variance = 1.2
    noise = 3e-5

    # gp_lite : bruit sur y_n ; sklearn : bruit sur y (normalisation identique)
    gp = GPRegressor(SquaredExponentialARD(lengthscales.copy(), variance),
                     noise=noise, normalize_y=True)
    gp.fit(X, y, optimize=False)
    mean_lite = gp.predict(X_test)

    sk = GaussianProcessRegressor(
        kernel=RBF(length_scale=lengthscales) * variance
        + WhiteKernel(noise_level=noise),
        normalize_y=True, optimizer=None,
    )
    sk.fit(X, y)
    mean_sk = sk.predict(X_test)

    np.testing.assert_allclose(mean_lite, mean_sk, atol=1e-6)


# ---------------------------------------------------------------------- #
# de bout en bout : Branin                                                #
# ---------------------------------------------------------------------- #
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


def test_branin_end_to_end():
    """Branin 2D, 30 points LHS, 200 points test : RMSE normalisee < 0.1."""
    X = lhs(30, 2, seed=1)
    X[:, 0] *= 10.0
    X[:, 1] *= 12.0
    y = branin(X)
    y_std = np.std(y)

    gp = GPRegressor(Matern52ARD(np.array([1.0, 1.0])), noise=1e-6,
                     normalize_y=True)
    gp.fit(X, y, optimize=True)

    X_test = lhs(200, 2, seed=2)
    X_test[:, 0] *= 10.0
    X_test[:, 1] *= 12.0
    y_test = branin(X_test)
    mean = gp.predict(X_test)
    rmse_rel = np.sqrt(np.mean((mean - y_test) ** 2)) / y_std
    assert rmse_rel < 0.1, f"RMSE normalisee {rmse_rel:.4f} >= 0.1"
