"""Tests de bout en bout du paquet ``gp_lite``."""

import numpy as np
import pytest

from gp_lite import GPRegressor, Matern32ARD, Matern52ARD, SquaredExponentialARD

RNG = np.random.default_rng(0)


# ---------------------------------------------------------------------- #
# Noyaux                                                                 #
# ---------------------------------------------------------------------- #
def test_kernels_symmetry():
    """Symetrie K(X1, X2) = K(X2, X1).T pour les trois noyaux ARD."""
    X1 = RNG.uniform(size=(7, 3))
    X2 = RNG.uniform(size=(5, 3))
    for cls in (SquaredExponentialARD, Matern32ARD, Matern52ARD):
        k = cls(np.array([0.3, 0.7, 1.1]), variance=2.0)
        K12, K21 = k.K(X1, X2), k.K(X2, X1)
        assert K12.shape == (7, 5)
        np.testing.assert_allclose(K12, K21.T, atol=1e-12)


def test_kernels_diag_and_matern52_at_zero():
    """Diagonale = variance ; Matern 5/2 vaut sigma^2 a r = 0."""
    X = RNG.uniform(size=(4, 2))
    for cls, var in ((SquaredExponentialARD, 1.5), (Matern32ARD, 0.8), (Matern52ARD, 2.0)):
        k = cls(np.array([0.4, 0.9]), variance=var)
        np.testing.assert_allclose(np.asarray(k.diag(X)), var, atol=1e-14)
        np.testing.assert_allclose(k.K(X, X).diagonal(), var, atol=1e-12)
    k52 = Matern52ARD(0.6, variance=3.0)
    x = np.array([[0.1, 0.4]])
    assert abs(k52.K(x, x)[0, 0] - 3.0) < 1e-12


def test_kernel_attributes():
    """Attributs publics lengthscales / variance."""
    k = Matern52ARD(np.array([0.2, 0.5]), variance=2.0)
    np.testing.assert_array_equal(k.lengthscales, [0.2, 0.5])
    assert k.variance == 2.0


# ---------------------------------------------------------------------- #
# GPRegressor                                                            #
# ---------------------------------------------------------------------- #
def test_predict_interpolates_at_zero_noise():
    """A bruit nul, predict interpole les points d'apprentissage (err < 1e-6)."""
    X = RNG.uniform(size=(15, 2))
    y = np.sin(3.0 * X[:, 0]) + X[:, 1] ** 2
    gp = GPRegressor(Matern52ARD(0.4), noise=1e-12, normalize_y=False)
    gp.fit(X, y, optimize=False)
    mean = gp.predict(X)
    assert np.abs(mean - y).max() < 1e-6


def _branin(x1, x2):
    """Fonction de Branin 2D (domaine [0, 50] x [0, 10])."""
    a, b, c = 1.0, 5.1 / (4.0 * np.pi**2), 5.0 / np.pi
    d, e, f = 0.0, 6.0, 10.0
    return (a * (x2 - b * x1**2 + c * x1 - d)**2 + e * (np.cos(x1) + 1.0)) * f


def _lhs(n, d, seed=1):
    """Low-discrepancy sampling simple par Latin hypercube (1 sample / case)."""
    rng = np.random.default_rng(seed)
    pts = np.empty((n, d))
    for j in range(d):
        edges = rng.permutation(n) / n
        pts[:, j] = edges + rng.uniform(size=n) * (1.0 / n)
        # re-borne dans [0, 1)
        pts[:, j] = np.clip(pts[:, j], 0.0, 1.0 - 1e-9)
    return pts


def test_branin_end_to_end():
    """Bout en bout : Branin, 30 pts LHS, RMSE normalisee < 0.1 sur 200 pts."""
    X = np.column_stack([_lhs(30, 2)[:, 0] * 50.0, _lhs(30, 2)[:, 1] * 10.0])
    y = _branin(X[:, 0], X[:, 1])
    Xte = np.column_stack([_lhs(200, 2, seed=2)[:, 0] * 50.0,
                           _lhs(200, 2, seed=2)[:, 1] * 10.0])
    yte = _branin(Xte[:, 0], Xte[:, 1])

    gp = GPRegressor(SquaredExponentialARD(np.array([10.0, 2.0])), noise=1e-6)
    gp.fit(X, y)
    mean, std = gp.predict(Xte, return_std=True)
    assert (std >= 0).all()

    rmse = np.sqrt(np.mean((mean - yte) ** 2))
    scale = np.sqrt(np.mean(yte ** 2))
    print(f"\nBranin RMSE normalisee = {rmse / scale:.4f}")
    assert rmse / scale < 0.1


# ---------------------------------------------------------------------- #
# Comparaison scikit-learn (tests uniquement)                            #
# ---------------------------------------------------------------------- #
def test_against_sklearn_fixed_hyperparameters():
    """Moyennes egales a 1e-6 pres avec sklearn (hyperparametres fixes)."""
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, WhiteKernel

    X = RNG.uniform(size=(25, 2))
    y = np.sin(2.0 * X[:, 0]) * X[:, 1] + 0.5

    lengthscales = np.array([0.35, 0.8])
    variance = 1.7
    noise = 1e-4

    # gp_lite, hyperparametres fixes, pas de normalisation (comparaison directe)
    gp = GPRegressor(SquaredExponentialARD(lengthscales, variance=variance),
                     noise=noise, normalize_y=False)
    gp.fit(X, y, optimize=False)

    kern = RBF(length_scale=lengthscales,
               length_scale_bounds=(0.0, np.inf)) * variance
    gpr = GaussianProcessRegressor(kernel=kern + WhiteKernel(noise),
                                   optimizer=None, normalize_y=False)
    gpr.fit(X, y)

    Xq = RNG.uniform(size=(12, 2))
    np.testing.assert_allclose(gp.predict(Xq), gpr.predict(Xq),
                               atol=1e-6, rtol=1e-6)


# ---------------------------------------------------------------------- #
# Leave-one-out                                                          #
# ---------------------------------------------------------------------- #
def test_loo_matches_refit():
    """loo() coincide avec n ajustements explicites (hyperparametres fixes)."""
    X = RNG.uniform(size=(12, 2))
    y = 2.0 * np.sin(X[:, 0]) + X[:, 1] ** 2 + 0.1

    # hyperparametres fixes et bruit non negligeable pour que LOO soit net ;
    # normalize_y=False pour que loo() et le recalcul parlent la meme langue (y bruts)
    kernel = Matern52ARD(np.array([0.5, 0.6]), variance=2.0)
    noise = 1e-3
    gp = GPRegressor(kernel, noise=noise, normalize_y=False)
    gp.fit(X, y, optimize=False)

    lloo, vloo = gp.loo()

    # recalcul explicite : n GP entraunes sans optimisation (hyperparametres fixes)
    means_exp, vars_exp = [], []
    for i in range(X.shape[0]):
        gi = GPRegressor(Matern52ARD(np.array([0.5, 0.6]), variance=2.0), noise=noise)
        Xs = np.delete(X, i, axis=0)
        y_rest = np.delete(y, i)
        K = gi.kernel.K(Xs, Xs)
        K.flat[:: K.shape[0] + 1] += gi.noise
        kx = gi.kernel.K(X[i:i + 1], Xs)[0]
        # moyenne predictive de y_i (bruit inclus) = E[f_i | y_rest] (moyenne du bruit nulle)
        means_exp.append(float(kx @ np.linalg.solve(K, y_rest)))
        # variance predictive de y_i = var(f_i | y_-i) + bruit
        vars_exp.append(float(gi.kernel.diag(X[i:i + 1])[0] + gi.noise -
                            kx @ np.linalg.solve(K, kx)))
    np.testing.assert_allclose(lloo, np.array(means_exp), atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(vloo, np.array(vars_exp), atol=1e-6, rtol=1e-6)


# ---------------------------------------------------------------------- #
# Divers                                                                 #
# ---------------------------------------------------------------------- #
def test_fit_returns_self_and_updates_hyperparameters():
    """fit renvoie self et met a jour kernel / noise ; optimize=False conserve."""
    X = RNG.uniform(size=(10, 2))
    y = X[:, 0] + 0.3
    k1 = SquaredExponentialARD(np.array([0.9, 0.9]), variance=1.0)
    gp1 = GPRegressor(k1, noise=1e-6)
    out = gp1.fit(X, y)
    assert out is gp1
    assert gp1.kernel.variance > 0 and gp1.noise > 0

    k2 = SquaredExponentialARD(np.array([0.4, 0.4]), variance=3.0)
    gp2 = GPRegressor(k2, noise=2e-4)
    gp2.fit(X, y, optimize=False)
    assert gp2.kernel.variance == 3.0
    assert gp2.noise == 2e-4
    np.testing.assert_array_equal(gp2.kernel.lengthscales, [0.4, 0.4])


def test_predict_without_std():
    """predict sans std renvoie un seul array."""
    X = RNG.uniform(size=(8, 2))
    y = X.sum(axis=1)
    gp = GPRegressor(SquaredExponentialARD(0.5)).fit(X, y, optimize=False)
    m = gp.predict(X)
    assert isinstance(m, np.ndarray) and m.shape == (8,)


def test_log_marginal_likelihood_finite():
    X = RNG.uniform(size=(12, 2))
    y = np.sin(4 * X[:, 0])
    gp = GPRegressor(Matern32ARD(0.6)).fit(X, y)
    lml = gp.log_marginal_likelihood()
    assert np.isfinite(lml)
