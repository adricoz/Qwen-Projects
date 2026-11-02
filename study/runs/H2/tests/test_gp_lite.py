"""Tests de gp_lite (pytest)."""
import numpy as np
import pytest

from gp_lite import (GPRegressor, Matern32ARD, Matern52ARD,
                     SquaredExponentialARD)

D = 2
rng = np.random.RandomState(0)


def _points(n, seed=1):
    """n points uniformes dans [0, 1]^D."""
    r = np.random.RandomState(seed)
    return r.rand(n, D)


# ------------------------------------------------------------------ #
# noyaux
# ------------------------------------------------------------------ #
@pytest.mark.parametrize("kernel_cls", [SquaredExponentialARD, Matern32ARD, Matern52ARD])
def test_kernel_symmetry(kernel_cls):
    k = kernel_cls(lengthscales=[0.3, 0.7], variance=2.0)
    X1 = _points(5)
    X2 = _points(7)
    K = k.K(X1, X2)
    assert K.shape == (5, 7)
    assert np.allclose(K, k.K(X2, X1).T, atol=1e-12)


@pytest.mark.parametrize("kernel_cls", [SquaredExponentialARD, Matern32ARD, Matern52ARD])
def test_kernel_diag_equals_variance(kernel_cls):
    k = kernel_cls(lengthscales=[0.3, 0.7], variance=2.5)
    X = _points(4)
    K = k.K(X, X)
    assert np.allclose(np.diag(K), 2.5)
    assert np.allclose(k.diag(X), 2.5)


def test_matern52_value_at_r_zero():
    # K(x, x) = sigma^2 quand r = 0
    k = Matern52ARD(lengthscales=[1.0, 1.0], variance=0.5)
    X = np.array([[0.2, 0.8]])
    assert k.K(X, X)[0, 0] == pytest.approx(0.5, abs=1e-12)


def test_matern52_known_value():
    # valeur de referance : 1D, l = 1, sigma^2 = 1, x - x' = 0.5
    k = Matern52ARD(lengthscales=[1.0], variance=1.0)
    val = k.K(np.array([[0.0]]), np.array([[0.5]]))[0, 0]
    s5 = np.sqrt(5.0)
    expected = (1 + s5 * 0.5 + 5.0 / 3.0 * 0.25) * np.exp(-s5 * 0.5)
    assert val == pytest.approx(expected, rel=1e-10)


# ------------------------------------------------------------------ #
# interpolation
# ------------------------------------------------------------------ #
def test_predict_interpolates_zero_noise():
    X = _points(10)
    f = lambda X: np.sin(2 * np.pi * X[:, 0]) + X[:, 1] ** 2
    gp = GPRegressor(SquaredExponentialARD([0.5, 0.5]), noise=1e-9,
                     normalize_y=True)
    gp.fit(X, f(X), optimize=False)
    mean = gp.predict(X)
    assert np.max(np.abs(mean - f(X))) < 1e-6


# ------------------------------------------------------------------ #
# leave-one-out
# ------------------------------------------------------------------ #
def test_loo_matches_explicit_refits():
    """loo() doit coïncider avec n re-ajustements sans re-optimisation."""
    X = _points(8)
    y = np.exp(-np.sum(X ** 2, axis=1)) + 0.1 * np.sin(X[:, 1] * 10.0)
    kern = SquaredExponentialARD([0.4, 0.6], variance=1.5)
    gp = GPRegressor(kern, noise=1e-4, normalize_y=True)
    gp.fit(X, y, optimize=False)

    mean_loo, var_loo = gp.loo()
    n = X.shape[0]
    # reference : complement de Schur calcule explicitement, point par point
    # (equivalent a n re-ajustements conditionnels, hyperparametres fixes)
    K = kern.K(X, X)
    yn = (y - gp.y_mean) / gp.y_std
    for i in range(n):
        k_i = np.delete(K[i], i)                     # (n-1,)
        K_ii = K[i, i] + gp.noise
        K_mi = np.delete(np.delete(K, i, 0), i, 1)   # (n-1, n-1)
        y_mi = np.delete(yn, i)
        P = K_mi + gp.noise * np.eye(n - 1)
        # moyenne conditionnelle de l'observation y_i sachant y_{-i}
        mu_i = k_i @ np.linalg.solve(P, y_mi)
        # K_ii - k_i^T (K_{-i} + noise I)^{-1} k_i
        var_i = K_ii - k_i @ np.linalg.solve(P, k_i)
        m = gp.y_mean + gp.y_std * mu_i
        v = (gp.y_std ** 2) * var_i
        assert m == pytest.approx(mean_loo[i], abs=1e-6)
        assert v == pytest.approx(var_loo[i], abs=1e-6)


# ------------------------------------------------------------------ #
# comparaison scikit-learn (tests seulement)
# ------------------------------------------------------------------ #
def test_against_sklearn_fixed_hyperparams():
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, WhiteKernel

    X = _points(12)
    Xt = _points(15)
    f = lambda X: np.cos(3.0 * X[:, 0]) * np.sin(4.0 * X[:, 1]) + X[:, 0]
    y = f(X)

    lengthscales = [0.4, 0.7]
    variance = 1.3
    noise = 1e-4

    gp = GPRegressor(SquaredExponentialARD(lengthscales, variance),
                     noise=noise, normalize_y=False)
    gp.fit(X, y, optimize=False)
    mean_gplite = gp.predict(Xt)

    skl = GaussianProcessRegressor(
        kernel=RBF(length_scale=lengthscales) * variance + WhiteKernel(noise_level=noise),
        optimizer=None, normalize_y=False)
    skl.fit(X, y)
    mean_skl = skl.predict(Xt)

    assert np.max(np.abs(mean_gplite - mean_skl)) < 1e-6


# ------------------------------------------------------------------ #
# bout en bout : Branin
# ------------------------------------------------------------------ #
def branin(x1, x2):
    """Fonction de Branin (2D)."""
    a, b, c = 1.0, 5.1 / (4.0 * np.pi ** 2), 5.0 / np.pi
    return (a * (x2 - b * x1 ** 2 + c * x1 - 5.0) ** 2 + 10.0 *
            (np.cos(x1) + 10.0)) * np.pi


def lhs(n, d, seed=0):
    """Latin Hyper Sampling simple : n points, d dimensions (numpy uniquement)."""
    rng = np.random.RandomState(seed)
    # une valeur uniforme par cellule de largeur 1/n, permutation differente
    # par colonne : c'est ce qui fait du LHS un echantillonnage stratifie
    U = np.empty((n, d))
    for j in range(d):
        U[:, j] = (np.arange(n) + rng.rand(n)) / n
        U[:, j] = U[rng.permutation(n), j]
    return U


def test_branin_end_to_end():
    X = lhs(30, 2)
    y = branin(X[:, 0], X[:, 1])
    gp = GPRegressor(SquaredExponentialARD([0.5, 0.5]))
    gp.fit(X, y, optimize=True)

    Xt = np.random.RandomState(3).rand(200, 2)
    yt = branin(Xt[:, 0], Xt[:, 1])
    pred = gp.predict(Xt)
    rmse = np.sqrt(np.mean((pred - yt) ** 2))
    scale = np.std(yt)
    assert rmse / scale < 0.1
