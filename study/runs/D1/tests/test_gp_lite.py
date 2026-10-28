"""Tests fonctionnels de gp_lite (noyaux, GP, sklearn, Branin)."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gp_lite import GPRegressor, Matern32ARD, Matern52ARD, SquaredExponentialARD
from tests.reference_naive import (
    naive_kernel_matrix,
    naive_loo,
    naive_predict,
)

RNG = np.random.default_rng(42)
D = 2


def make_data(n=12, d=D, seed=0, noisy=True):
    r = np.random.default_rng(seed)
    X = r.uniform(0, 1, (n, d))
    y = np.sin(3 * X[:, 0]) * np.cos(2 * X[:, 1])
    if noisy:
        y = y + 0.01 * r.normal(size=n)
    return X, y


HYPS = [
    ("se", [0.2, 0.8], 3.0, 1e-2),
    ("m32", [0.4, 0.4], 1.5, 1e-4),
    ("m52", [0.3, 0.5], 2.0, 1e-3),
    ("m52", [1.0, 0.1], 0.5, 1e-1),
    ("m52", [0.05, 2.0], 4.0, 1e-5),
]


def make_kernel(kind, lengthscales, variance):
    if kind == "se":
        return SquaredExponentialARD(lengthscales, variance)
    if kind == "m32":
        return Matern32ARD(lengthscales, variance)
    return Matern52ARD(lengthscales, variance)


# --------------------------------------------------------------------- #
# Noyaux
# --------------------------------------------------------------------- #
@pytest.mark.parametrize("cls", [SquaredExponentialARD, Matern32ARD, Matern52ARD])
def test_kernel_symmetry(cls):
    X1 = RNG.uniform(0, 1, (5, D))
    X2 = RNG.uniform(0, 1, (7, D))
    k = cls([0.3, 0.7], variance=2.0)
    K12 = k.K(X1, X2)
    K21 = k.K(X2, X1)
    assert K12.shape == (5, 7)
    assert np.allclose(K12.T, K21)
    assert np.allclose(K12, K12.T.T)


@pytest.mark.parametrize("cls", [SquaredExponentialARD, Matern32ARD, Matern52ARD])
def test_kernel_diagonal_equals_variance(cls):
    X = RNG.uniform(0, 1, (9, D))
    k = cls([0.2, 1.1], variance=3.5)
    assert np.allclose(np.diag(k.K(X, X)), 3.5)
    assert np.allclose(k.diag(X), 3.5)


def test_matern52_at_zero_distance():
    k = Matern52ARD([0.5, 0.5], variance=2.5)
    x = np.array([[0.3, 0.7]])
    assert np.allclose(k.K(x, x), 2.5)
    # m32 et se idem
    for cls in (Matern32ARD, SquaredExponentialARD):
        assert np.allclose(cls([0.5], variance=1.3).K([[0.0]], [[0.0]]), 1.3)


@pytest.mark.parametrize("kind,ls,var,noise", HYPS)
def test_kernel_matches_naive_reference(kind, ls, var, noise):
    X1 = RNG.uniform(0, 1, (4, D))
    X2 = RNG.uniform(0, 1, (6, D))
    K = make_kernel(kind, ls, var).K(X1, X2)
    ref = naive_kernel_matrix(X1, X2, kind, np.array(ls), var)
    assert np.allclose(K, ref, atol=1e-12)


# --------------------------------------------------------------------- #
# GP : prediction vs reference naive, avec et sans normalize_y
# --------------------------------------------------------------------- #
@pytest.mark.parametrize("normalize", [True, False])
@pytest.mark.parametrize("kind,ls,var,noise", HYPS)
def test_predict_matches_naive_reference(kind, ls, var, noise, normalize):
    X, y = make_data()
    Xnew = RNG.uniform(0, 1, (8, D))
    gp = GPRegressor(make_kernel(kind, ls, var), noise=noise,
                     normalize_y=normalize).fit(X, y, optimize=False)
    mean, std = gp.predict(Xnew, return_std=True)
    y_used = (y - gp.y_mean) / gp.y_std if gp.normalize_y else y
    ref_mean, ref_std = naive_predict(X, Xnew, y_used, ls, var, noise, kind, return_std=True)
    shift = gp.y_mean if gp.normalize_y else 0.0
    scale = gp.y_std if gp.normalize_y else 1.0
    assert np.allclose(mean, shift + scale * ref_mean, atol=1e-8)
    assert np.allclose(std, scale * ref_std, rtol=1e-6)


@pytest.mark.parametrize("normalize", [True, False])
@pytest.mark.parametrize("kind,ls,var,noise", HYPS)
def test_log_marginal_likelihood_matches_naive(kind, ls, var, noise, normalize):
    X, y = make_data()
    gp = GPRegressor(make_kernel(kind, ls, var), noise=noise,
                     normalize_y=normalize).fit(X, y, optimize=False)
    y_used = (y - gp.y_mean) / gp.y_std if gp.normalize_y else y
    ref = naive_log_mll(X, y_used, ls, var, noise, kind)
    assert gp.log_marginal_likelihood() == pytest.approx(ref, rel=1e-9)


def naive_log_mll(X, y, ls, var, noise, kind):
    from tests.reference_naive import naive_log_marginal_likelihood
    return naive_log_marginal_likelihood(X, y, np.array(ls), var, noise, kind)


# --------------------------------------------------------------------- #
# Interpolation a bruit nul
# --------------------------------------------------------------------- #
def test_predict_interpolates_at_zero_noise():
    X, y = make_data(noisy=False)
    gp = GPRegressor(Matern52ARD([0.4, 0.4]), noise=1e-10).fit(X, y, optimize=False)
    assert np.max(np.abs(gp.predict(X) - y)) < 1e-6

    # et avec optimisation (le bruit doit descendre pres de zero)
    gp = GPRegressor(Matern52ARD([0.4, 0.4]), noise=1e-6).fit(X, y)
    assert np.max(np.abs(gp.predict(X) - y)) < 1e-6


# --------------------------------------------------------------------- #
# LOO : forme fermee vs n re-ajustements explicites
# --------------------------------------------------------------------- #
@pytest.mark.parametrize("normalize", [True, False])
@pytest.mark.parametrize("kind,ls,var,noise", HYPS)
def test_loo_matches_explicit_refits(kind, ls, var, noise, normalize):
    X, y = make_data(n=10)
    kernel = make_kernel(kind, ls, var)
    gp = GPRegressor(kernel, noise=noise, normalize_y=normalize).fit(X, y, optimize=False)
    mean, var_p = gp.loo()
    y_used = (y - gp.y_mean) / gp.y_std if gp.normalize_y else y
    ref_mean, ref_var = naive_loo(X, y_used, ls, var, noise, kind)
    shift = gp.y_mean if gp.normalize_y else 0.0
    scale = gp.y_std if gp.normalize_y else 1.0
    assert np.allclose(mean, shift + scale * ref_mean, atol=1e-8)
    assert np.allclose(var_p, scale ** 2 * ref_var, rtol=1e-6)


# --------------------------------------------------------------------- #
# Parite scikit-learn (hyperparametres FIXES)
# --------------------------------------------------------------------- #
def test_parity_with_sklearn_fixed_hyperparams():
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF as SkRBF
    from sklearn.gaussian_process.kernels import WhiteKernel

    X, y = make_data(n=14, seed=7)
    Xnew = RNG.uniform(0, 1, (60, D))
    ls, var, noise = [0.35, 0.55], 2.0, 1e-4

    gp = GPRegressor(SquaredExponentialARD(ls, variance=var), noise=noise,
                     normalize_y=True).fit(X, y, optimize=False)
    sk = GaussianProcessRegressor(
        kernel=SkRBF(length_scale=ls) * var + WhiteKernel(noise),
        normalize_y=True,
        optimizer=None,
    ).fit(X, y)

    mean_gp, std_gp = gp.predict(Xnew, return_std=True)
    mean_sk, std_sk = sk.predict(Xnew, return_std=True)
    # sklearn inclut le WhiteKernel dans return_std ; gp_lite renvoie l'ecart-type
    # de f seul : on en deduit var_f = var_sk - noise (variance additive).
    var_f = std_sk ** 2 - gp.noise * gp.y_std ** 2
    assert np.max(np.abs(mean_gp - mean_sk)) < 1e-6
    assert np.max(np.abs(std_gp ** 2 - var_f)) < 1e-8
    assert np.max(np.abs(std_gp - np.sqrt(var_f)) / std_sk) < 1e-5

    # log-vraissemblance marginale : sklearn la calcule sur y normalise (ddof=0)
    ll_gp = gp.log_marginal_likelihood()
    ll_sk = sk.log_marginal_likelihood_value_
    assert ll_gp == pytest.approx(ll_sk, rel=1e-8)


# --------------------------------------------------------------------- #
# Branin : plan LHS et test de bout en bout
# --------------------------------------------------------------------- #
def branin(x):
    """Fonction de Branin 2D sur [0, 1]^2."""
    x1, x2 = x[:, 0], x[:, 1]
    return ((x2 - 5.0 * x1 ** 2 + 5.0 * x1 - 5.0) ** 2
            + 10.0 * (np.sin(6.0 * x1) + np.sin(2.0 * x2)) + 10.0)


def lhs(n, d, seed=0):
    """Latin hypercube simple : permutation aleatoire par dimension."""
    r = np.random.default_rng(seed)
    pts = np.empty((n, d))
    for j in range(d):
        perm = r.permutation(n)
        u = r.uniform(size=n)
        pts[perm, j] = (np.arange(n) + u) / n
    return pts


def test_branin_end_to_end_rmse_normalized():
    Xtr = lhs(30, 2, seed=3)
    ytr = branin(Xtr)
    gp = GPRegressor(Matern52ARD([0.3, 0.3])).fit(Xtr, ytr)
    Xte = lhs(200, 2, seed=11)
    pred = gp.predict(Xte)
    rmse = np.sqrt(np.mean((pred - branin(Xte)) ** 2))
    assert rmse / np.std(ytr) < 0.1


def test_branin_fit_beats_random_reference():
    """La vraissemblance apres fit doit battre celle d'un choix aleatoire d'hyperparametres."""
    Xtr, ytr = lhs(25, 2, seed=5), None
    Xtr = lhs(25, 2, seed=5)
    ytr = branin(Xtr)
    gp = GPRegressor(Matern52ARD([0.5, 0.5])).fit(Xtr, ytr)
    y_n = (ytr - gp.y_mean) / gp.y_std
    from tests.reference_naive import naive_log_marginal_likelihood
    ll_fit = gp.log_marginal_likelihood()
    ll_naive = naive_log_marginal_likelihood(Xtr, y_n, gp.kernel.lengthscales,
                                             gp.kernel.variance, gp.noise, "m52")
    assert ll_fit == pytest.approx(ll_naive, rel=1e-9)
    ll_rand = naive_log_marginal_likelihood(Xtr, y_n, [1.0, 1.0], 1.0, 1e-6, "m52")
    assert ll_fit > ll_rand
