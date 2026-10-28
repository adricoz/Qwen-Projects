"""Acceptance tests for the gp_lite agent runs, written by Claude (never shown to the agent unless the
prompt variant says so). Each value is compared with an independent NumPy reference (slogdet / solve,
no Cholesky) or with scikit-learn's optimised GP.

Run from the project directory, or point GP_LITE_DIR to it:
    python -m pytest -q C:/Users/SIM/Desktop/Code-Adri/Qwen-Projects/acceptance/test_acceptance.py
"""
import os
import sys

import numpy as np
import pytest
from scipy.spatial.distance import cdist

sys.path.insert(0, os.environ.get("GP_LITE_DIR", os.getcwd()))
from gp_lite.gp import GPRegressor  # noqa: E402
from gp_lite.kernels import Matern32ARD, Matern52ARD, SquaredExponentialARD  # noqa: E402

KERNELS = {"se": SquaredExponentialARD, "m32": Matern32ARD, "m52": Matern52ARD}


# ---------- independent reference ----------

def ref_k(name, X1, X2, ls, var):
    r = cdist(X1 / ls, X2 / ls)
    if name == "se":
        return var * np.exp(-0.5 * r**2)
    if name == "m32":
        return var * (1 + np.sqrt(3) * r) * np.exp(-np.sqrt(3) * r)
    return var * (1 + np.sqrt(5) * r + 5 * r**2 / 3) * np.exp(-np.sqrt(5) * r)


def ref_norm(y, normalize):
    return (y.mean(), y.std()) if normalize else (0.0, 1.0)


def ref_lml(name, X, y, ls, var, noise, normalize):
    m, s = ref_norm(y, normalize)
    yn = (y - m) / s
    K = ref_k(name, X, X, ls, var) + noise * np.eye(len(X))
    _, logdet = np.linalg.slogdet(K)
    return -0.5 * yn @ np.linalg.solve(K, yn) - 0.5 * logdet - 0.5 * len(X) * np.log(2 * np.pi)


def ref_predict(name, X, y, Xs, ls, var, noise, normalize):
    m, s = ref_norm(y, normalize)
    yn = (y - m) / s
    K = ref_k(name, X, X, ls, var) + noise * np.eye(len(X))
    Ks = ref_k(name, Xs, X, ls, var)
    mean = Ks @ np.linalg.solve(K, yn)
    var_f = var - np.sum(Ks * np.linalg.solve(K, Ks.T).T, axis=1)
    return mean * s + m, np.sqrt(np.maximum(var_f, 0.0)) * s


def ref_loo(name, X, y, ls, var, noise, normalize):
    """Brute force: n conditionings without point i, normalisation of the full set, variance of y_i."""
    m, s = ref_norm(y, normalize)
    yn = (y - m) / s
    mu, v = np.empty(len(X)), np.empty(len(X))
    for i in range(len(X)):
        keep = np.arange(len(X)) != i
        K = ref_k(name, X[keep], X[keep], ls, var) + noise * np.eye(keep.sum())
        ks = ref_k(name, X[i:i + 1], X[keep], ls, var)[0]
        mu[i] = ks @ np.linalg.solve(K, yn[keep])
        v[i] = var + noise - ks @ np.linalg.solve(K, ks)
    return mu * s + m, v * s**2


def branin(X):
    x1, x2 = X[:, 0], X[:, 1]
    return ((x2 - 5.1 / (4 * np.pi**2) * x1**2 + 5 / np.pi * x1 - 6)**2
            + 10 * (1 - 1 / (8 * np.pi)) * np.cos(x1) + 10)


def branin_design(n, seed):
    rng = np.random.default_rng(seed)
    return np.column_stack([rng.uniform(-5, 10, n), rng.uniform(0, 15, n)])


def toy_data(seed=0, n=12, d=3):
    rng = np.random.default_rng(seed)
    X = rng.uniform(0, 1, (n, d))
    y = np.sin(3 * X[:, 0]) + X[:, 1]**2 - 0.5 * X[:, 2] + 2.0
    return X, y


def fixed_gp(name, X, y, ls, var, noise, normalize):
    gp = GPRegressor(KERNELS[name](np.array(ls, dtype=float), var), noise=noise, normalize_y=normalize)
    gp.fit(X, y, optimize=False)
    return gp


LS, VAR, NOISE = [0.3, 0.7, 1.5], 1.7, 1e-3


# ---------- kernels ----------

@pytest.mark.parametrize("name", KERNELS)
def test_kernel_values(name):
    rng = np.random.default_rng(1)
    X1, X2 = rng.normal(size=(7, 3)), rng.normal(size=(5, 3))
    k = KERNELS[name](np.array(LS), VAR)
    K = np.asarray(k.K(X1, X2))
    assert K.shape == (7, 5)
    np.testing.assert_allclose(K, ref_k(name, X1, X2, np.array(LS), VAR), rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(np.asarray(k.diag(X1)), np.full(7, VAR), rtol=1e-12)
    np.testing.assert_allclose(k.lengthscales, LS)
    assert k.variance == pytest.approx(VAR)


# ---------- GP at fixed hyperparameters ----------

@pytest.mark.parametrize("normalize", [False, True])
@pytest.mark.parametrize("name", ["se", "m52"])
def test_lml_fixed(name, normalize):
    X, y = toy_data()
    gp = fixed_gp(name, X, y, LS, VAR, NOISE, normalize)
    ref = ref_lml(name, X, y, np.array(LS), VAR, NOISE, normalize)
    assert gp.log_marginal_likelihood() == pytest.approx(ref, rel=1e-6, abs=1e-6)


@pytest.mark.parametrize("normalize", [False, True])
@pytest.mark.parametrize("name", ["se", "m32"])
def test_predict_fixed(name, normalize):
    X, y = toy_data()
    Xs = np.random.default_rng(2).uniform(0, 1, (20, 3))
    gp = fixed_gp(name, X, y, LS, VAR, NOISE, normalize)
    mean, std = gp.predict(Xs, return_std=True)
    ref_mean, ref_std = ref_predict(name, X, y, Xs, np.array(LS), VAR, NOISE, normalize)
    np.testing.assert_allclose(np.ravel(mean), ref_mean, rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(np.ravel(std), ref_std, rtol=1e-5, atol=1e-8)
    assert np.ravel(gp.predict(Xs)).shape == (20,)


def test_interpolation_zero_noise():
    X, y = toy_data()
    gp = fixed_gp("se", X, y, LS, VAR, 1e-10, False)
    mean, std = gp.predict(X, return_std=True)
    np.testing.assert_allclose(np.ravel(mean), y, atol=1e-6)
    assert np.max(np.ravel(std)) < 1e-3


@pytest.mark.parametrize("normalize", [False, True])
@pytest.mark.parametrize("name", ["se", "m52"])
def test_loo_fixed(name, normalize):
    X, y = toy_data()
    gp = fixed_gp(name, X, y, LS, VAR, NOISE, normalize)
    mu, v = gp.loo()
    ref_mu, ref_v = ref_loo(name, X, y, np.array(LS), VAR, NOISE, normalize)
    np.testing.assert_allclose(np.ravel(mu), ref_mu, rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(np.ravel(v), ref_v, rtol=1e-5, atol=1e-10)


def test_near_singular_does_not_crash():
    X, y = toy_data()
    X, y = np.vstack([X, X[:3]]), np.concatenate([y, y[:3]])   # duplicated points, tiny noise
    gp = fixed_gp("se", X, y, LS, VAR, 1e-12, True)
    mean, std = gp.predict(X[:5], return_std=True)
    assert np.all(np.isfinite(mean)) and np.all(np.isfinite(std))


# ---------- fit (maximum likelihood) on Branin, random designs ----------

def sklearn_optimum(name, X, y):
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Matern, WhiteKernel
    base = RBF([1.0, 1.0], (1e-3, 1e3)) if name == "se" else Matern([1.0, 1.0], (1e-3, 1e3), nu=2.5)
    kern = ConstantKernel(1.0, (1e-3, 1e3)) * base + WhiteKernel(1e-4, (1e-10, 1e1))
    gp = GaussianProcessRegressor(kern, normalize_y=True, n_restarts_optimizer=5, random_state=0).fit(X, y)
    k = gp.kernel_
    return k.k1.k2.length_scale, k.k1.k1.constant_value, k.k2.noise_level


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("name", ["se", "m52"])
def test_fit_branin(name, seed):
    X = branin_design(30, seed)
    y = branin(X)
    gp = GPRegressor(KERNELS[name](np.ones(2), 1.0), noise=1e-6, normalize_y=True)
    gp.fit(X, y)
    ls, var, noise = np.asarray(gp.kernel.lengthscales), float(gp.kernel.variance), float(gp.noise)
    mine = ref_lml(name, X, y, ls, var, noise, True)
    # 1. the objective it optimised is the true log marginal likelihood
    assert gp.log_marginal_likelihood() == pytest.approx(mine, rel=1e-3, abs=0.05)
    # 2. it reaches the optimum found by scikit-learn (same model), within 2 nats
    sk = ref_lml(name, X, y, *sklearn_optimum(name, X, y), True)
    assert mine >= sk - 2.0, f"LML {mine:.2f} < scikit-learn optimum {sk:.2f}"
    # 3. accuracy on 200 test points
    Xt = branin_design(200, 100 + seed)
    yt = branin(Xt)
    nrmse = np.sqrt(np.mean((np.ravel(gp.predict(Xt)) - yt)**2)) / yt.std()
    assert nrmse < 0.1
