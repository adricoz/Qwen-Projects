"""Tests de gp_lite : noyaux, prédictions, LOO, et références naïves indépendantes."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gp_lite import GPRegressor, Matern32ARD, Matern52ARD, SquaredExponentialARD  # noqa: E402

TOL = 1e-7


# ---------------------------------------------------------------------------
# références naïves et indépendantes (np.linalg, boucles)
# ---------------------------------------------------------------------------
def naive_kernel_matrix(kind, lengthscales, variance, X1, X2):
    """Matrice de covariance calculée ligne par ligne, sans le code du paquet."""
    d = len(lengthscales)
    out = np.empty((X1.shape[0], X2.shape[0]))
    for i, x in enumerate(X1):
        for j, xp in enumerate(X2):
            r = np.sqrt(np.sum(((x - xp) / lengthscales) ** 2))
            if kind == "rbf":
                out[i, j] = variance * np.exp(-0.5 * r * r)
            elif kind == "m32":  # rho = sqrt(3) * d
                rho = np.sqrt(3.0) * r
                out[i, j] = variance * (1 + rho) * np.exp(-rho)
            elif kind == "m52":  # rho = sqrt(5) * d
                rho = np.sqrt(5.0) * r
                out[i, j] = variance * (1 + rho + 5 * rho * rho / 3) * np.exp(-rho)
    return out


def naive_log_likelihood(kind, lengthscales, variance, noise, X, y_n):
    """Log-vraisemblance marginale via np.linalg.inv / slogdet (boucles incluses)."""
    n = X.shape[0]
    C = naive_kernel_matrix(kind, lengthscales, variance, X, X) + noise * np.eye(n)
    A = np.linalg.inv(C)
    logdet = np.linalg.slogdet(C)[1]
    quad = 0.0
    for i in range(n):
        quad += y_n[i] * float(y_n @ A[i, :])
    return -0.5 * quad - 0.5 * logdet - 0.5 * n * np.log(2.0 * np.pi)


def naive_predict(kind, lengthscales, variance, noise, Xtr, ytr, Xte):
    """Moyenne et écart-type a posteriori (sans bruit) par np.linalg.solve."""
    n = Xtr.shape[0]
    C = naive_kernel_matrix(kind, lengthscales, variance, Xtr, Xtr) + noise * np.eye(n)
    alpha = np.linalg.solve(C, ytr)
    k = naive_kernel_matrix(kind, lengthscales, variance, Xte, Xtr)
    mean = k @ alpha
    v = np.empty(len(Xte))
    vmat = np.linalg.inv(C)
    for i in range(len(Xte)):
        v[i] = variance - float(k[i, :] @ (vmat @ k[i, :]))
    return mean, np.sqrt(np.maximum(v, 0.0))


def naive_loo(kind, lengthscales, variance, noise, X, y):
    """LOO par conditionnement explicite sur C_{-i} (une résolution par point)."""
    n = X.shape[0]
    C = naive_kernel_matrix(kind, lengthscales, variance, X, X) + noise * np.eye(n)
    mean = np.empty(n)
    var = np.empty(n)
    for i in range(n):
        idx = np.arange(n) != i
        D = C[np.ix_(idx, idx)]
        k = C[idx, i]
        yi = y[idx]
        mean[i] = float(k @ np.linalg.solve(D, yi))
        var[i] = C[i, i] - float(k @ np.linalg.solve(D, k))
    return mean, var


# jeux d'hyperparamètres fixes, variés (avec des échelles et bruits différents)
HYPER_SETS = [
    dict(lengthscales=[0.3, 0.7], variance=1.0, noise=1e-5),
    dict(lengthscales=[1.5, 0.2], variance=4.0, noise=1e-3),
    dict(lengthscales=[0.05, 0.05], variance=0.2, noise=1e-1),
]


def make_data(seed=0, n=25):
    rng = np.random.default_rng(seed)
    X = rng.random((n, 2)) * 2.0
    y = np.sin(3 * X[:, 0]) * np.cos(2 * X[:, 1]) + 0.5 * X[:, 0] ** 2
    return X, y


def make_points(seed=0, n=25):
    """Deux matrices de points (X1, X2) en [0, 2)^2."""
    rng = np.random.default_rng(seed)
    X1 = rng.random((n, 2)) * 2.0
    X2 = rng.random((n, 2)) * 2.0
    return X1, X2


# ---------------------------------------------------------------------------
# noyaux
# ---------------------------------------------------------------------------
def test_kernels_symmetry_diag_m52():
    X, Y = make_points()
    for cls in (SquaredExponentialARD, Matern32ARD, Matern52ARD):
        k = cls([0.4, 0.9], variance=2.5)
        K = k.K(X, Y)
        assert np.allclose(K, k.K(Y, X).T, atol=1e-12)
        assert np.allclose(k.diag(X), 2.5, atol=1e-12)
        assert np.allclose(np.diag(k.K(X, X)), 2.5, atol=1e-12)
    # Matérn 5/2 à r = 0 vaut sigma^2 ; valeurs exactes à r connu
    k = Matern52ARD(1.0, variance=0.8)
    assert k.K(np.zeros((1, 1)), np.zeros((1, 1)))[0, 0] == pytest.approx(0.8)
    a, b = np.array([[0.0]]), np.array([[1.0]])
    # longueur standardisée d = 1 / l = 1, paramètre rho = sqrt(5) * d
    rho = np.sqrt(5.0)
    expect = 0.8 * (1 + rho + 5 * rho * rho / 3) * np.exp(-rho)
    assert k.K(a, b)[0, 0] == pytest.approx(expect, rel=1e-12)


@pytest.mark.parametrize("kind,cls", [("rbf", SquaredExponentialARD),
                                      ("m32", Matern32ARD),
                                      ("m52", Matern52ARD)])
@pytest.mark.parametrize("h", HYPER_SETS)
def test_kernel_against_naive(kind, cls, h):
    X, Y = make_points(seed=7, n=12)
    k = cls(h["lengthscales"], variance=h["variance"])
    assert np.allclose(k.K(X, Y),
                       naive_kernel_matrix(kind, h["lengthscales"], h["variance"], X, Y),
                       atol=TOL * h["variance"])


# ---------------------------------------------------------------------------
# log-vraisemblance, prédiction, LOO vs références naïves
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("normalize_y", [True, False])
@pytest.mark.parametrize("h", HYPER_SETS)
def test_ll_predict_loo_vs_naive(h, normalize_y):
    X, y = make_data(seed=3)
    kind, cls = "rbf", SquaredExponentialARD
    gp = GPRegressor(cls(h["lengthscales"], variance=h["variance"]),
                     noise=h["noise"], normalize_y=normalize_y)
    gp.fit(X, y, optimize=False)

    y_mean, y_std = np.mean(y), np.std(y)
    y_n = (y - y_mean) / y_std if normalize_y else y

    # log-vraisemblance
    ref = naive_log_likelihood(kind, h["lengthscales"], h["variance"], h["noise"], X, y_n)
    assert gp.log_marginal_likelihood() == pytest.approx(ref, abs=1e-8)

    # prédiction moyenne / écart-type
    Xte = np.array([[0.3, 0.4], [1.7, 0.9], [0.1, 1.3], [2.0, 2.0]])
    m, s = gp.predict(Xte, return_std=True)
    m_ref, s_ref = naive_predict(kind, h["lengthscales"], h["variance"], h["noise"], X, y_n, Xte)
    m_ref = m_ref * y_std + y_mean if normalize_y else m_ref
    s_ref = s_ref * y_std if normalize_y else s_ref
    assert np.allclose(m, m_ref, atol=1e-8)
    assert np.allclose(s, s_ref, atol=1e-8)

    # LOO
    mean, var = gp.loo()
    m_ref, v_ref = naive_loo(kind, h["lengthscales"], h["variance"], h["noise"], X, y_n)
    if normalize_y:
        m_ref, v_ref = m_ref * y_std + y_mean, v_ref * y_std ** 2
    assert np.allclose(mean, m_ref, atol=1e-7)
    assert np.allclose(var, v_ref, atol=1e-7)


# ---------------------------------------------------------------------------
# tests fonctionnels demandés
# ---------------------------------------------------------------------------
def test_predict_interpolates_zero_noise():
    X, y = make_data(seed=1)
    gp = GPRegressor(SquaredExponentialARD([0.5, 0.5]), noise=1e-12, normalize_y=True)
    gp.fit(X, y, optimize=False)
    m = gp.predict(X)
    assert np.max(np.abs(m - y)) < 1e-6


def test_loo_matches_n_refits_fixed_hp():
    X, y = make_data(seed=2)
    lengthscales, variance, noise = [0.4, 0.8], 2.0, 1e-4
    gp = GPRegressor(SquaredExponentialARD(lengthscales), noise=noise, normalize_y=True)
    gp.fit(X, y, optimize=False)
    mean, var = gp.loo()

    y_mean, y_std = np.mean(y), np.std(y)
    y_n = (y - y_mean) / y_std
    n = X.shape[0]
    K = gp.kernel.K(X, X)
    for i in range(n):
        idx = np.arange(n) != i
        Csub = K[np.ix_(idx, idx)] + noise * np.eye(n - 1)
        k = K[idx, i]
        # fit explicite sans ré-optimisation, normalisation du jeu complet
        alpha = np.linalg.solve(Csub, y_n[idx])
        m_i = float(k @ alpha) * y_std + y_mean
        v_i = (K[i, i] + noise - float(k @ np.linalg.solve(Csub, k))) * y_std ** 2
        assert abs(mean[i] - m_i) < 1e-6
        assert abs(var[i] - v_i) < 1e-6


def test_sklearn_fixed_hp_agreement():
    try:
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import RBF, WhiteKernel
    except ImportError:
        pytest.skip("scikit-learn unavailable")
    X, y = make_data(seed=4, n=40)
    Xte = np.array([[0.5, 0.5], [1.0, 1.0], [0.2, 1.8], [1.9, 0.3]])
    ls, var, noise = [0.6, 0.4], 1.5, 1e-4
    gp = GPRegressor(SquaredExponentialARD(ls, variance=var), noise=noise,
                     normalize_y=True)
    gp.fit(X, y, optimize=False)
    m_ours = gp.predict(Xte)

    sk = GaussianProcessRegressor(RBF(length_scale=ls) * var + WhiteKernel(noise),
                                  normalize_y=True, optimizer=None)
    sk.fit(X, y)
    m_sk = sk.predict(Xte)
    assert np.allclose(m_ours, m_sk, atol=1e-6)


# ---------------------------------------------------------------------------
# Branin + LHS + optimisation
# ---------------------------------------------------------------------------
def branin(x1, x2):
    """Fonction de Branin 2D (x1 dans [0, 10], x2 dans [0, 12])."""
    a, b, c, r, t = 1.0, 5.1 / (4 * np.pi ** 2), 5 / np.pi, 6.0, 10 / (3 * np.pi)
    return a * (x2 - b * x1 ** 2 + c * x1 - r) ** 2 + r * (1 - t) * np.cos(x1) + 12.0 * np.pi


def lhs(n, d, seed=0):
    """Lattice hyper-carré simple en [0, 1)^d (permutation par dimension)."""
    rng = np.random.default_rng(seed)
    grid = np.linspace(0.5, n - 0.5, n) / n
    out = np.empty((n, d))
    for j in range(d):
        order = rng.permutation(n)
        out[:, j] = grid[order]
    return out


def branin_designs():
    """Branin sur [0,10]x[0,12] : 1 LHS + 2 plans aléatoires."""
    X_lhs = lhs(30, 2, seed=42) * np.array([10.0, 12.0])
    rng = np.random.default_rng(1)
    X_r1 = rng.random((30, 2)) * np.array([10.0, 12.0])
    rng = np.random.default_rng(2)
    X_r2 = rng.random((30, 2)) * np.array([10.0, 12.0])
    return [X_lhs, X_r1, X_r2]


def test_branin_end_to_end_normalized_rmse():
    X = lhs(30, 2, seed=42) * np.array([10.0, 12.0])
    y = branin(X[:, 0], X[:, 1])
    gp = GPRegressor(SquaredExponentialARD([3.0, 3.0]), noise=1e-3)
    gp.fit(X, y)
    Xt = lhs(200, 2, seed=99) * np.array([10.0, 12.0])
    yt = branin(Xt[:, 0], Xt[:, 1])
    pred = gp.predict(Xt)
    rmse = np.sqrt(np.mean((pred - yt) ** 2))
    rmse_norm = rmse / np.std(yt)
    assert rmse_norm < 0.1, f"RMSE normalisée = {rmse_norm}"


def test_fit_matches_naive_ll_on_branin():
    for X in branin_designs():
        y = branin(X[:, 0], X[:, 1])
        gp = GPRegressor(SquaredExponentialARD([1.0, 1.0]), noise=1e-4)
        gp.fit(X, y, optimize=True)
        y_mean, y_std = np.mean(y), np.std(y)
        y_n = (y - y_mean) / y_std
        ref = naive_log_likelihood("rbf", gp.kernel.lengthscales, gp.kernel.variance,
                                   gp.noise, X, y_n)
        assert gp.log_marginal_likelihood() == pytest.approx(ref, abs=1e-6)


def test_fit_vs_sklearn_optimized_branin():
    try:
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import RBF, WhiteKernel
    except ImportError:
        pytest.skip("scikit-learn unavailable")
    for X in branin_designs():
        y = branin(X[:, 0], X[:, 1])

        # notre GP optimisé
        gp = GPRegressor(SquaredExponentialARD([1.0, 1.0]), noise=1e-4)
        gp.fit(X, y, optimize=True)
        y_mean, y_std = np.mean(y), np.std(y)
        y_n = (y - y_mean) / y_std
        ll_ours = naive_log_likelihood("rbf", gp.kernel.lengthscales, gp.kernel.variance,
                                       gp.noise, X, y_n)

        # scikit-learn optimisé, même modèle
        sk = GaussianProcessRegressor(RBF(length_scale=[1.0, 1.0]) * 1.0 + WhiteKernel(1e-4),
                                      normalize_y=True,
                                      optimizer="fmin_l_bfgs_b",
                                      n_restarts_optimizer=5)
        sk.fit(X, y)
        ksk = sk.kernel_                              # (RBF * const) + White
        ls = np.asarray(ksk.k1.k1.length_scale)
        var = float(ksk.k1.k2.constant_value)
        noise = float(ksk.k2.noise_level)
        ll_sk = naive_log_likelihood("rbf", ls, var, noise, X, y_n)

        # la notre ne doit pas être nettement plus basse (marge : 20 unités de log-ll)
        assert ll_ours >= ll_sk - 20.0, f"ll_ours={ll_ours}, ll_sk={ll_sk}"
