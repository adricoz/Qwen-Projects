"""Tests pytest du paquet gp_lite."""
import numpy as np
import pytest

from gp_lite import (
    GPRegressor,
    SquaredExponentialARD,
    Matern32ARD,
    Matern52ARD,
)
from gp_lite.lhs import lhs


# --------------------------------------------------------------------- #
# fonctions test
# --------------------------------------------------------------------- #
def branin2d(x):
    """Fonction de Branin (2D), domaine canonique [0, 1]^2 (x2 ∈ [-15, 30])."""
    x1, x2 = x[..., 0], x[..., 1]
    a = 1.0
    b = 5.1 / (4.0 * np.pi ** 2)
    c = 5.0 / np.pi
    r = 6.0
    s = 10.0
    t = 1.0 / (8.0 * np.pi)
    y = a * (x2 - b * x1 ** 2 + c * x1 - r) ** 2 + s * (1.0 - t) * np.cos(x1) + s
    return y


def make_data(n=30, seed=0):
    """30 points LHS dans [0, 1]^2 et leurs valeurs de Branin."""
    X = lhs(n, 2, seed=seed)
    y = branin2d(X)
    return X, y


# --------------------------------------------------------------------- #
# noyaux
# --------------------------------------------------------------------- #
@pytest.mark.parametrize("cls", [SquaredExponentialARD, Matern32ARD, Matern52ARD])
def test_kernel_symmetry_and_diag(cls):
    rng = np.random.RandomState(0)
    X1 = rng.rand(5, 2)
    X2 = rng.rand(7, 2)
    k = cls(lengthscales=[0.4, 0.8], variance=2.0)
    K12 = k.K(X1, X2)
    K21 = k.K(X2, X1)
    assert K12.shape == (5, 7)
    assert np.allclose(K12, K21.T, atol=1e-12)
    # diagonale = variance
    assert np.allclose(k.K(X1, X1).diagonal(), 2.0, atol=1e-12)
    assert np.allclose(k.diag(X1), 2.0, atol=1e-12)


def test_matern52_at_zero_distance():
    # Matérn 5/2 à r = 0 vaut sigma^2
    k = Matern52ARD(lengthscales=[1.0], variance=3.5)
    x = np.array([[0.0], [1.0], [2.0]])
    assert np.allclose(k.K(x, x).diagonal(), 3.5, atol=1e-12)


def test_matern52_formula():
    # vérification de la formule sur une distance simple
    k = Matern52ARD(lengthscales=[1.0], variance=1.0)
    x0 = np.array([[0.0]])
    x1 = np.array([[2.0]])
    r = 2.0
    expected = (1.0 + np.sqrt(5.0) * r + 5.0 * r * r / 3.0) * np.exp(-np.sqrt(5.0) * r)
    assert np.allclose(k.K(x0, x1)[0, 0], expected, atol=1e-12)


# --------------------------------------------------------------------- #
# interpolation à bruit nul
# --------------------------------------------------------------------- #
def test_predict_interpolates_zero_noise():
    X, y = make_data(n=15)
    gp = GPRegressor(SquaredExponentialARD(lengthscales=[0.4, 0.4], variance=1.0),
                     noise=0.0, normalize_y=False)
    gp.fit(X, y, optimize=False)
    mean = gp.predict(X)
    assert np.max(np.abs(mean - y)) < 1e-6


# --------------------------------------------------------------------- #
# leave-one-out vs n re-ajustements explicites
# --------------------------------------------------------------------- #
def test_loo_matches_explicit_retraining():
    X, y = make_data(n=12)
    # hyperparamètres FIXÉS : on ne refait que le conditionnement
    def fit_fixed(Xi, yi):
        return (GPRegressor(SquaredExponentialARD(lengthscales=[0.5, 0.3], variance=1.0),
                            noise=1e-3, normalize_y=False)
                .fit(Xi, yi, optimize=False))

    gp = fit_fixed(X, y)
    mean_loo, var_loo = gp.loo()

    ref_mean, ref_var = [], []
    for i in range(len(X)):
        Xr = np.delete(X, i, axis=0)
        yr = np.delete(y, i)
        gi = fit_fixed(Xr, yr)
        mi, si = gi.predict(X[i:i + 1], return_std=True)
        # variance prédictive de y_i = variance de f + bruit
        ref_mean.append(mi[0])
        ref_var.append(si[0] ** 2 + 1e-3)
    ref_mean = np.array(ref_mean)
    ref_var = np.array(ref_var)
    assert np.allclose(mean_loo, ref_mean, atol=1e-6)
    assert np.allclose(var_loo, ref_var, atol=1e-6)


# --------------------------------------------------------------------- #
# comparaison avec scikit-learn (hyperparamètres fixés)
# --------------------------------------------------------------------- #
def test_matches_sklearn():
    pytest.importorskip("sklearn")
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, WhiteKernel

    X, y = make_data(n=20)
    x_test = lhs(50, 2, seed=99)

    ls = [0.3, 0.7]
    var = 1.2
    noise = 1e-4

    gp = GPRegressor(SquaredExponentialARD(lengthscales=ls, variance=var),
                     noise=noise, normalize_y=False)
    gp.fit(X, y, optimize=False)
    mean_lite = gp.predict(x_test)

    kernel = var * RBF(length_scale=ls) + WhiteKernel(noise_level=noise)
    gpr = GaussianProcessRegressor(kernel=kernel, normalize_y=False, optimizer=None)
    gpr.fit(X, y)
    mean_sk = gpr.predict(x_test)

    assert np.allclose(mean_lite, mean_sk, atol=1e-6)


# --------------------------------------------------------------------- #
# bout en bout : Branin
# --------------------------------------------------------------------- #
def test_branin_end_to_end():
    X, y = make_data(n=30, seed=3)
    X_test = lhs(200, 2, seed=123)

    gp = GPRegressor(SquaredExponentialARD(lengthscales=[0.3, 0.3], variance=1.0),
                     noise=1e-6)
    gp.fit(X, y, optimize=True)

    y_true = branin2d(X_test)
    mean = gp.predict(X_test)
    rmse = np.sqrt(np.mean((mean - y_true) ** 2))
    # RMSE normalisée par l'écart-type de la fonction sur le nuage de test
    norm_rmse = rmse / np.std(y_true)
    assert norm_rmse < 0.1, f"RMSE normalisée trop grande : {norm_rmse:.3f}"
    # le GP interpole grossièrement les points d'apprentissage
    # (bruit ajusté > 0 ; l'interpolation stricte est testée à bruit nul)
    assert np.max(np.abs(gp.predict(X) - y)) < 1e-2 * np.std(y)
