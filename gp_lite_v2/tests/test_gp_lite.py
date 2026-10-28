"""Tests de gp_lite (paquet de regression par processus gaussien)."""
import numpy as np
import pytest

from gp_lite import GPRegressor, Matern32ARD, Matern52ARD, SquaredExponentialARD


def branin2d(x1, x2):
    """Fonction de Branin (2D) sur [-5, 10] x [0, 10]."""
    a = 1.0
    b = 5.1 / (4.0 * np.pi ** 2)
    c = 5.0 / np.pi
    r = 6.0
    s = 10.0
    t = 1.0 / (8.0 * np.pi)
    return (a * (x2 - b * x1 ** 2 + c * x1 - r) ** 2
            + s * (1.0 - t) * np.cos(x1) + s)


def lhs_2d(n, seed=0):
    """Latin hypercube simple 2D : n points, une par strate ligne et colonne."""
    rng = np.random.default_rng(seed)
    u = (np.arange(n) + rng.random(n)) / n
    v = (rng.permutation(np.arange(n)).astype(float) + rng.random(n)) / n
    lo = np.array([-5.0, 0.0])
    hi = np.array([10.0, 10.0])
    return lo + np.column_stack([u, v]) * (hi - lo)


# ---------------------------------------------------------------- noyaux
def test_kernels_symmetry_and_diag():
    rng = np.random.default_rng(1)
    X = rng.random((7, 2))
    Y = rng.random((4, 2))
    for cls in (SquaredExponentialARD, Matern32ARD, Matern52ARD):
        k = cls(lengthscales=[0.5, 1.5], variance=2.0)
        K12, K21 = k.K(X, Y), k.K(Y, X)
        assert np.allclose(K12, K21.T, atol=1e-12)
        assert np.allclose(k.diag(X), 2.0 * np.ones(7))
        assert np.allclose(np.diag(k.K(X, X)), 2.0 * np.ones(7), atol=1e-10)


def test_matern52_at_zero_distance():
    k = Matern52ARD(lengthscales=[0.3], variance=3.5)
    x = np.array([[1.0, 2.0]])
    assert np.isclose(k.K(x, x)[0, 0], 3.5, atol=1e-12)


# ---------------------------------------------------------------- fit/predict
def test_predict_interpolates_training_points():
    """A bruit quasi nul, predict interpole les points d'entrainement."""
    rng = np.random.default_rng(2)
    X = rng.random((12, 2))
    y = np.sin(X[:, 0] * 2.0) * X[:, 1]
    gp = GPRegressor(SquaredExponentialARD(lengthscales=[1.0, 1.0], variance=1.0),
                     noise=1e-6, normalize_y=True)
    gp.fit(X, y, optimize=False)
    mu = gp.predict(X)
    assert np.max(np.abs(mu - y)) < 1e-6


def test_loo_matches_n_refits():
    """loo() doit coincider avec n ajustements sans re-optimisation."""
    rng = np.random.default_rng(3)
    X = rng.random((10, 2))
    y = np.cos(X[:, 0]) + 0.5 * X[:, 1]
    kernel = SquaredExponentialARD(lengthscales=[0.8, 1.2], variance=1.5)
    gp = GPRegressor(kernel, noise=1e-4, normalize_y=False)
    gp.refit(X, y)
    loo = gp.loo()
    explicit = np.empty(len(y))
    for i in range(len(y)):
        mask = np.arange(len(y)) != i
        gp_i = GPRegressor(SquaredExponentialARD(lengthscales=[0.8, 1.2],
                                                 variance=1.5),
                           noise=1e-4, normalize_y=False)
        gp_i.refit(X[mask], y[mask])
        explicit[i] = y[i] - gp_i.predict(X[i:i + 1])[0]
    assert np.allclose(loo, explicit, atol=1e-6)


def test_matches_sklearn_fixed_hyperparams():
    """Moyennes identiques a sklearn (RBF ARD + WhiteKernel, hyperparam fixes)."""
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, WhiteKernel
    rng = np.random.default_rng(4)
    X = rng.uniform(0, 1, (15, 2))
    y = np.sin(2.0 * np.pi * X[:, 0]) * np.cos(2.0 * np.pi * X[:, 1])
    Xs = rng.uniform(0, 5, (30, 2))

    lengths = [0.4, 0.7]
    var, noise = 1.0, 1e-4
    gp = GPRegressor(SquaredExponentialARD(lengthscales=lengths, variance=var),
                     noise=noise, normalize_y=False)
    gp.refit(X, y)  # hyperparametres fixes, sans optimisation
    mu = gp.predict(Xs)

    # length_scale vectoriel => RBF ARD (equivalent ard=True des versions recentes)
    sk = GaussianProcessRegressor(
        RBF(length_scale=lengths) + WhiteKernel(noise_level=noise),
        optimizer=None, normalize_y=False)
    sk.fit(X, y)
    mu_sk = sk.predict(Xs)
    assert np.allclose(mu, mu_sk, atol=1e-6), \
        (mu[:3], mu_sk[:3], np.max(np.abs(mu - mu_sk)))


# ---------------------------------------------------------------- e2e Branin
def test_branin_e2e():
    X_train = lhs_2d(30)
    y_train = branin2d(X_train[:, 0], X_train[:, 1])
    gp = GPRegressor(SquaredExponentialARD(lengthscales=[1.0, 1.0])).fit(X_train, y_train)
    rng = np.random.default_rng(5)
    X_test = rng.uniform([-5.0, 0.0], [10.0, 10.0], (200, 2))
    y_test = branin2d(X_test[:, 0], X_test[:, 1])
    mu = gp.predict(X_test)
    rmse = np.sqrt(np.mean((mu - y_test) ** 2))
    scale = np.std(y_test)
    assert rmse / scale < 0.1, f"RMSE normalisee = {rmse / scale:.4f}"
