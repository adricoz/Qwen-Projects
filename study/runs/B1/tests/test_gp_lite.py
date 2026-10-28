"""Pytest suite for gp_lite (NumPy / SciPy ; scikit-learn uniquement ici, pour les tests)."""
import numpy as np
import pytest

from gp_lite.gp import GPRegressor
from gp_lite.kernels import Matern32ARD, Matern52ARD, SquaredExponentialARD

KERNELS = {"se": SquaredExponentialARD, "m32": Matern32ARD, "m52": Matern52ARD}


# --------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------- #
def toy2d(seed=0, n=15):
    """Petit jeu 2D deterministe (sin/cos)."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(0, 1, (n, 2))
    y = np.sin(3 * X[:, 0]) + X[:, 1] ** 2
    return X, y


def lhs(n, d, seed=0):
    """Latin hypercube simple : permutation de n tranches par dimension."""
    rng = np.random.default_rng(seed)
    u = np.stack([rng.permutation(n) / n for _ in range(d)], axis=1)
    return (u + rng.random((n, d))) / n


def branin(X):
    x1, x2 = X[:, 0], X[:, 1]
    return ((x2 - 5.1 / (4 * np.pi ** 2) * x1 ** 2 + 5 / np.pi * x1 - 6) ** 2
            + 10 * (1 - 1 / (8 * np.pi)) * np.cos(x1) + 10)


def fixed_gp(name, X, y, ls, var, noise, normalize=True):
    gp = GPRegressor(KERNELS[name](np.array(ls, dtype=float), var),
                     noise=noise, normalize_y=normalize)
    gp.fit(X, y, optimize=False)
    return gp


# --------------------------------------------------------------------- #
# kernels
# --------------------------------------------------------------------- #
@pytest.mark.parametrize("name", KERNELS)
def test_kernel_symmetry_and_diagonal(name):
    rng = np.random.default_rng(1)
    X = rng.uniform(0, 1, (8, 3))
    k = KERNELS[name](np.array([0.3, 0.7, 1.5]), 1.7)
    K = k.K(X, X)
    np.testing.assert_allclose(K, K.T, rtol=1e-12, atol=1e-14)          # symetrie
    np.testing.assert_allclose(np.diag(K), np.full(8, 1.7), rtol=1e-12)  # diagonale = variance
    np.testing.assert_allclose(k.diag(X), np.full(8, 1.7), rtol=1e-12)


def test_matern52_at_zero():
    k = Matern52ARD([0.5, 2.0], 2.5)
    assert k.K(np.array([[0.0, 0.0]]), np.array([[0.0, 0.0]]))[0, 0] == pytest.approx(2.5)


def test_kernel_attributes():
    k = SquaredExponentialARD([1.0, 0.5])
    np.testing.assert_allclose(k.lengthscales, [1.0, 0.5])
    assert k.variance == 1.0


# --------------------------------------------------------------------- #
# GP fixed hyperparameters
# --------------------------------------------------------------------- #
def test_interpolation_zero_noise():
    X, y = toy2d()
    gp = fixed_gp("se", X, y, [0.3, 0.6], 1.0, 1e-10, normalize=False)
    mean, std = gp.predict(X, return_std=True)
    np.testing.assert_allclose(mean, y, atol=1e-6)
    assert np.max(std) < 1e-3


@pytest.mark.parametrize("name", ["se", "m52"])
@pytest.mark.parametrize("normalize", [True, False])
def test_loo_matches_refit(name, normalize):
    """loo() = n refits sans re-optimisation (tol 1e-6)."""
    X, y = toy2d(n=10)
    ls, var, noise = [0.3, 0.6], 1.3, 1e-3
    gp = fixed_gp(name, X, y, ls, var, noise, normalize)
    mu, v = gp.loo()

    # reference brute force : n conditionnements sans le point i
    Ktr = gp.kernel
    mu_ref, v_ref = np.empty(len(X)), np.empty(len(X))
    if normalize:
        m, s = float(y.mean()), float(y.std())
        yn = (y - m) / s
    else:
        m, s, yn = 0.0, 1.0, y
    for i in range(len(X)):
        keep = np.arange(len(X)) != i
        K = Ktr.K(X[keep], X[keep]) + noise * np.eye(keep.sum())
        ks = Ktr.K(X[i:i + 1], X[keep])[0]
        mu_ref[i] = ks @ np.linalg.solve(K, yn[keep])
        v_ref[i] = Ktr.variance + noise - ks @ np.linalg.solve(K, ks)
    mu_ref = mu_ref * s + m
    v_ref = v_ref * s ** 2
    np.testing.assert_allclose(np.ravel(mu), mu_ref, rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(np.ravel(v), v_ref, rtol=1e-6, atol=1e-8)


def test_lml_matches_reference():
    X, y = toy2d()
    ls, var, noise = [0.3, 0.6], 1.3, 1e-3
    gp = fixed_gp("se", X, y, ls, var, noise, normalize=True)
    m, s = float(y.mean()), float(y.std())
    yn = (y - m) / s
    K = SquaredExponentialARD(ls, var).K(X, X) + noise * np.eye(len(X))
    sign, logdet = np.linalg.slogdet(K)
    ref = (-0.5 * yn @ np.linalg.solve(K, yn) - 0.5 * logdet
           - 0.5 * len(X) * np.log(2 * np.pi))
    assert gp.log_marginal_likelihood() == pytest.approx(ref, rel=1e-8)


def test_predict_matches_sklearn_fixed():
    """Moyennes egales a 1e-6 pres avec sklearn (hyperparametres fixes identiques)."""
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel

    ls, var, noise = [0.3, 0.6], 1.3, 1e-4
    X, y = toy2d(seed=7, n=12)
    Xs = np.random.default_rng(2).uniform(0, 1, (20, 2))

    gp = fixed_gp("se", X, y, ls, var, noise, normalize=True)
    mine = np.ravel(gp.predict(Xs))

    sk = (ConstantKernel(var) * RBF(ls) + WhiteKernel(noise))
    skg = GaussianProcessRegressor(sk, normalize_y=True, optimizer=None).fit(X, y)
    theirs = skg.predict(Xs)
    np.testing.assert_allclose(mine, theirs, rtol=0.0, atol=1e-6)


# --------------------------------------------------------------------- #
# end to end
# --------------------------------------------------------------------- #
def test_branin_end_to_end():
    X = lhs(30, 2, seed=0)
    X = np.column_stack([X[:, 0] * 15 - 5, X[:, 1] * 15])
    y = branin(X)
    gp = GPRegressor(SquaredExponentialARD(np.ones(2), 1.0), noise=1e-6).fit(X, y)
    Xt = lhs(200, 2, seed=1)
    Xt = np.column_stack([Xt[:, 0] * 15 - 5, Xt[:, 1] * 15])
    yt = branin(Xt)
    nrmse = np.sqrt(np.mean((np.ravel(gp.predict(Xt)) - yt) ** 2)) / yt.std()
    assert nrmse < 0.1
