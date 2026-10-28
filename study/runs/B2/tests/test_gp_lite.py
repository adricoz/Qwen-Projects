"""Tests unitaires de gp_lite (pytest)."""
import numpy as np
import pytest

from gp_lite import (
    GPRegressor,
    Matern32ARD,
    Matern52ARD,
    SquaredExponentialARD,
)


KERNELS = {"se": SquaredExponentialARD, "m32": Matern32ARD, "m52": Matern52ARD}


def _toy(seed=0, n=15, d=2):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-2, 2, (n, d))
    y = np.sin(X[:, 0]) + X[:, 1] ** 2
    return X, y


# ---------- noyaux ----------

@pytest.mark.parametrize("name", KERNELS)
def test_kernel_symmetry(name):
    """K(X, X) doit être symétrique."""
    X = np.random.default_rng(1).normal(size=(9, 3))
    k = KERNELS[name](np.array([0.5, 1.0, 2.0]), 2.0)
    K = k.K(X, X)
    np.testing.assert_allclose(K, K.T, atol=1e-12)


@pytest.mark.parametrize("name", KERNELS)
def test_kernel_diagonal_equals_variance(name):
    """La diagonale vaut variance (r = 0)."""
    X = np.random.default_rng(2).normal(size=(6, 2))
    k = KERNELS[name](np.array([0.7, 1.3]), 3.5)
    np.testing.assert_allclose(k.diag(X), np.full(6, 3.5), rtol=1e-12)
    np.testing.assert_allclose(np.diag(k.K(X, X)), np.full(6, 3.5), rtol=1e-12)


def test_matern52_at_zero():
    """Matérn 5/2 en r = 0 vaut sigma^2 (formule 1 + sqrt(5)r + 5r^2/3 -> 1)."""
    x = np.zeros((1, 2))
    k = Matern52ARD(np.ones(2), 2.5)
    assert k.K(x, x)[0, 0] == pytest.approx(2.5, rel=1e-14)


# ---------- prédiction / interpolation ----------

def test_predict_interpolates_zero_noise():
    """À bruit quasi nul, le GP interpole les points d'apprentissage."""
    X, y = _toy()
    gp = GPRegressor(SquaredExponentialARD(np.ones(2), 1.0), noise=1e-12, normalize_y=False)
    gp.fit(X, y, optimize=False)
    mean = gp.predict(X)
    np.testing.assert_allclose(mean, y, atol=1e-6)


def test_predict_returns_tuple_with_std():
    """predict(return_std=True) renvoie (moyenne, écart-type) deux arrays."""
    X, y = _toy()
    gp = GPRegressor(SquaredExponentialARD(np.ones(2), 1.0), noise=1e-6)
    gp.fit(X, y, optimize=False)
    Xs = np.random.default_rng(3).uniform(-2, 2, (10, 2))
    mean, std = gp.predict(Xs, return_std=True)
    assert mean.shape == (10,)
    assert std.shape == (10,)
    assert np.all(np.isfinite(std)) and np.all(std >= 0)


# ---------- leave-one-out ----------

def _loo_bruteforce(gp, X, y):
    """Référence : n conditionnements sans le point i (hyperparamètres fixés)."""
    ls = gp.kernel.lengthscales
    var = gp.kernel.variance
    noise = gp.noise
    n = len(X)
    mu = np.empty(n)
    v = np.empty(n)
    for i in range(n):
        keep = np.arange(n) != i
        k = gp.kernel
        K = k.K(X[keep], X[keep]) + noise * np.eye(keep.sum())
        ks = k.K(X[i:i + 1], X[keep])[0]
        # en unités de y_n (la normalisation interne du GP)
        if gp.normalize_y:
            m, s = gp._y_mean, gp._y_std
            yn = (y - m) / s
            mu[i] = (ks @ np.linalg.solve(K, yn[keep])) * s + m
            v[i] = (var + noise - ks @ np.linalg.solve(K, ks)) * s ** 2
        else:
            mu[i] = ks @ np.linalg.solve(K, y[keep])
            v[i] = var + noise - ks @ np.linalg.solve(K, ks)
    return mu, v


@pytest.mark.parametrize("normalize", [False, True])
def test_loo_matches_n_refits(normalize):
    """loo() doit coïncider avec n ajustements explicites (tol 1e-6)."""
    X, y = _toy(n=10)
    gp = GPRegressor(SquaredExponentialARD(np.array([0.8, 1.2]), 1.5),
                     noise=1e-3, normalize_y=normalize)
    gp.fit(X, y, optimize=False)
    mu, v = gp.loo()
    ref_mu, ref_v = _loo_bruteforce(gp, X, y)
    np.testing.assert_allclose(mu, ref_mu, rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(v, ref_v, rtol=1e-6, atol=1e-8)


# ---------- comparaison scikit-learn ----------

def test_matches_sklearn_fixed_hyperparams():
    """Moyennes identiques à scikit-learn (RBF ARD + WhiteKernel, opt. désactivée)."""
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import ConstantKernel, Matern, RBF, WhiteKernel

    rng = np.random.default_rng(7)
    X = rng.uniform(-3, 3, (25, 2))
    y = 3.1 * np.cos(X[:, 0]) + 0.7 * X[:, 1] ** 2 + np.sin(X[:, 0] + X[:, 1])
    Xs = rng.uniform(-3, 3, (40, 2))

    ls = [0.4, 1.6]
    var = 2.3
    noise = 5e-4
    gp = GPRegressor(SquaredExponentialARD(np.array(ls), var), noise=noise, normalize_y=True)
    gp.fit(X, y, optimize=False)

    kern = ConstantKernel(var) * RBF(ls) + WhiteKernel(noise)
    skl = GaussianProcessRegressor(kern, normalize_y=True, optimizer=None).fit(X, y)

    np.testing.assert_allclose(gp.predict(Xs), skl.predict(Xs), rtol=1e-6, atol=1e-8)


# ---------- test de bout en bout (Branin) ----------

def lhs(n, d, seed=0):
    """Latin hypercube simple : permutation de n stratums par dimension."""
    rng = np.random.default_rng(seed)
    u = rng.uniform(size=(n, d))
    P = np.column_stack([rng.permutation(n) for _ in range(d)])
    X = (P - 0.5 + u) / n
    return X  # dans [0, 1] ^ d


def branin(x1, x2):
    """Fonction de Branin 2D."""
    return ((x2 - 5.1 / (4 * np.pi ** 2) * x1 ** 2 + 5 / np.pi * x1 - 6) ** 2
            + 10 * (1 - 1 / (8 * np.pi)) * np.cos(x1) + 10)


def test_branin_end_to_end():
    """30 points LHS, 200 points test : RMSE normalisée < 0.1 après fit."""
    rng = np.random.default_rng(0)
    X = np.column_stack([
        lhs(30, 2, seed=1)[:, 0] * 15.0 - 5.0,
        lhs(30, 2, seed=1)[:, 1] * 15.0,
    ])
    y = branin(X[:, 0], X[:, 1])

    gp = GPRegressor(SquaredExponentialARD(np.ones(2), 1.0), noise=1e-6)
    gp.fit(X, y)  # optimisation maximale de la vraisemblance marginale

    X = np.column_stack([
        lhs(200, 2, seed=2)[:, 0] * 15.0 - 5.0,
        lhs(200, 2, seed=2)[:, 1] * 15.0,
    ])
    yt = branin(X[:, 0], X[:, 1])
    pred = np.ravel(gp.predict(X))
    nrmse = np.sqrt(np.mean((pred - yt) ** 2)) / yt.std()
    assert nrmse < 0.1, f"RMSE normalisée trop élevée : {nrmse:.4f}"


def test_lml_positive_and_finite():
    """log_marginal_likelihood est fini après fit."""
    X, y = _toy()
    gp = GPRegressor(SquaredExponentialARD(np.ones(2), 1.0), noise=1e-3)
    gp.fit(X, y, optimize=False)
    assert np.isfinite(gp.log_marginal_likelihood())
