"""Tests pytest du paquet `gp_lite`."""
import os
import sys

import numpy as np

# garantit que gp_lite est importable depuis le repository
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from gp_lite import GPRegressor, Matern32ARD, Matern52ARD, SquaredExponentialARD  # noqa: E402

D = 2
X = np.array([[0.0, 0.0], [0.3, 0.7], [1.0, 0.5], [2.2, 1.1]])


# ---------------------------------------------------------------------------
# noyaux
# ---------------------------------------------------------------------------
def test_kernels_symmetry_and_diag():
    """Les noyaux sont symétriques et leur diagonale vaut la variance."""
    for kern in (
        SquaredExponentialARD([0.5, 1.0], 1.0),
        Matern32ARD([0.7, 0.9], 1.0),
        Matern52ARD([1.0, 1.0], 1.0),
    ):
        K = kern.K(X, X)
        np.testing.assert_allclose(K, K.T, atol=1e-12)
        np.testing.assert_allclose(np.diag(K), [1.0] * X.shape[0], atol=1e-12)
        np.testing.assert_allclose(kern.diag(X), [1.0] * X.shape[0], atol=1e-12)


def test_matern52_zero_distance():
    """Matérn 5/2 à r = 0 vaut sigma² (id. 3/2 et RBF)."""
    m = Matern52ARD([1.0, 1.0], 2.5)
    np.testing.assert_allclose(m.K(X, X).diagonal(), 2.5, atol=1e-12)
    m3 = Matern32ARD([1.0, 1.0], 2.5)
    np.testing.assert_allclose(m3.K(X, X).diagonal(), 2.5, atol=1e-12)
    s = SquaredExponentialARD([1.0, 1.0], 2.5)
    np.testing.assert_allclose(s.K(X, X).diagonal(), 2.5, atol=1e-12)


def test_kernel_interface():
    """Constructeur Classe(lengthscales, variance=1.0) et attributs publics."""
    k = SquaredExponentialARD([0.5, 0.25])
    assert k.variance == 1.0
    assert k.lengthscales.shape == (2,)
    np.testing.assert_allclose(k.lengthscales, [0.5, 0.25])
    k2 = SquaredExponentialARD([0.5, 0.25], variance=3.0)
    assert k2.variance == 3.0


def test_kernel_rect_cross_block():
    """K(X1, X2) renvoie bien une matrice (n1, n2) pour n1 != n2."""
    k = SquaredExponentialARD([1.0, 1.0])
    K12 = k.K(X[:2], X[1:])
    K21 = k.K(X[1:], X[:2])
    assert K12.shape == (2, 3)
    assert K21.shape == (3, 2)
    np.testing.assert_allclose(K12, K21.T, atol=1e-12)


# ---------------------------------------------------------------------------
# GP: interpolation, loo, vraisemblance
# ---------------------------------------------------------------------------
def _toy_data():
    """Fonction test 2D lisse avec un tirage déterministe."""
    rng = np.random.default_rng(42)
    X = rng.uniform(-1.0, 1.0, size=(24, D))
    y = np.sin(3.0 * X[:, 0]) * X[:, 1] + 0.3 * X[:, 0] ** 2
    return X, y


def test_predict_interpolates_noiseless():
    """Avec bruit quasi nul, predict interpole les points d'apprentissage.

    Les hyperparamètres sont conservés (optimize=False): la vraie condition
    d'interpolation est noise -> 0, pas le choix des hyperparamètres.
    """
    X, y = _toy_data()
    gp = GPRegressor(SquaredExponentialARD([0.5, 0.5]), noise=1e-10)
    gp.fit(X, y, optimize=False)
    pred = gp.predict(X)
    assert np.max(np.abs(pred - y)) < 1e-6


def test_loo_matches_retraining():
    """loo() égale le recalcul explicite par n ajustements (hyperparams fixés).

    On travaille en ``normalize_y=False`` pour que le recalcul de référence
    utilise exactement les mêmes unités que la formule fermée (la formule
    utilise la normalisation du jeu d'apprentissage complet).
    """
    X, y = _toy_data()
    gp = GPRegressor(SquaredExponentialARD([0.4, 0.6], 2.0), noise=0.01, normalize_y=False)
    gp.fit(X, y, optimize=False)
    mean, var = gp.loo()

    for i in range(X.shape[0]):
        # on exclut le point i et on prédit au point i (vrai leave-one-out)
        mask = np.arange(X.shape[0]) != i
        # refit sans réoptimisation (optimize=False garde les hyperparamètres)
        gp_i = GPRegressor(
            SquaredExponentialARD(gp.kernel.lengthscales, gp.kernel.variance),
            noise=gp.noise, normalize_y=False,
        )
        gp_i.fit(X[mask], y[mask], optimize=False)
        m_i, s_i = gp_i.predict(X[i], return_std=True)
        # variance de y_i (bruit inclus) = var(f) + bruit, en unités de y
        v_i = s_i ** 2 + gp_i.noise
        np.testing.assert_allclose(mean[i], m_i, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(var[i], v_i, rtol=1e-6, atol=1e-6)


def test_lml_and_noise_units():
    """La LML sur y_n dépend de noise exprimé sur y_n; cohérence fit/predict."""
    X, y = _toy_data()
    gp = GPRegressor(SquaredExponentialARD([0.5, 0.5]), noise=0.01)
    gp.fit(X, y, optimize=False)
    lml = gp.log_marginal_likelihood()
    assert np.isfinite(lml)
    # bruit normalisé attendu = noise * std(y)^2 sur les données brutes
    assert gp.noise > 0
    # prédiction de moyenne et de variance dans les unités de y
    m = gp.predict(X[:3])
    assert m.shape == (3,)
    assert m.std() > 0.1  # même ordre que y, pas de normalisation résiduelle


# ---------------------------------------------------------------------------
# comparaison scikit-learn (hyperparamètres fixés, identiques)
# ---------------------------------------------------------------------------
def test_matches_sklearn_fixed_hyperparams():
    """Moyennes prédites égales (1e-6) à celles de sklearn, hyperparams fixés."""
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, WhiteKernel

    X, y = _toy_data()
    lens = np.array([0.4, 0.6])
    var = 2.0
    noise = 0.01

    k_sk = RBF(length_scale=list(lens)) * var + WhiteKernel(noise)
    gpr = GaussianProcessRegressor(k_sk, normalize_y=True, optimizer=None)
    gpr.fit(X, y)
    X_test = np.array([[0.2, -0.7], [-1.1, 0.9], [0.4, 0.1]])
    mean_sk = gpr.predict(X_test)

    gp = GPRegressor(SquaredExponentialARD(lens, variance=var), noise=noise, normalize_y=True)
    gp.fit(X, y, optimize=False)
    mean_gp = gp.predict(X_test)
    np.testing.assert_allclose(mean_gp, mean_sk, atol=1e-6)

    # la LML doit aussi coïncider (tolerance 1e-6)
    lml_sk = gpr.log_marginal_likelihood(np.log(np.array([lens[0], lens[1], var, noise])))
    np.testing.assert_allclose(gp.log_marginal_likelihood(), lml_sk, atol=1e-6)


# ---------------------------------------------------------------------------
# branin + LHS
# ---------------------------------------------------------------------------
def branin(x):
    """Fonction de Branin 2D (x1 in [0, 30], x2 in [0, 12])."""
    x1, x2 = x[..., 0], x[..., 1]
    a = 1.0
    b = 5.1 / (4.0 * np.pi**2)
    c = 5.0 / np.pi
    r = 6.0
    t = 10.0 / (3.0 * np.pi)
    return (a * (x2 - b * x1**2 + c * x1 - r)) ** 2 + r * (1 - t) * np.cos(x1) + r


def lhs(n, d, seed=0):
    """Latin hypercube simple avec numpy: n points dans [0, 1]^d.

    Chaque colonne est un tirage sans remise (permutation) des n cellules
    [j/n, (j+1)/n), uniformément complété à l'intérieur de chaque cellule.
    """
    rng = np.random.default_rng(seed)
    pts = np.empty((n, d))
    for j in range(d):
        order = np.arange(n)[rng.permutation(n)]
        pts[:, j] = (order + rng.random(n)) / n
    return pts


def test_branin_end_to_end():
    """Branin: 30 points LHS -> RMSE normalisée < 0.1 sur 200 points."""
    n_train, n_test = 30, 200
    bounds = np.array([[0.0, 0.0], [30.0, 12.0]])
    Xtr = lhs(n_train, D, seed=1)
    Xtr = bounds[0] + Xtr * (bounds[1] - bounds[0])
    ytr = branin(Xtr)

    gp = GPRegressor(SquaredExponentialARD([5.0, 3.0], 100.0), noise=1.0)
    gp.fit(Xtr, ytr, optimize=True)

    rng = np.random.default_rng(2)
    Xte = bounds[0] + rng.random((n_test, D)) * (bounds[1] - bounds[0])
    yte = branin(Xte)
    pred = gp.predict(Xte)
    rmse = np.sqrt(np.mean((pred - yte) ** 2))
    scale = np.std(yte)
    assert rmse / scale < 0.1, f"RMSE normalisée trop élevée: {rmse / scale:.4f}"


def test_fit_returns_self_and_stores():
    """fit() renvoie self et expose noise/kernel après optimisation."""
    X, y = _toy_data()
    gp = GPRegressor(SquaredExponentialARD([0.5, 0.5]), noise=1e-4)
    out = gp.fit(X, y, optimize=True)
    assert out is gp
    assert np.all(gp.kernel.lengthscales > 0)
    assert gp.kernel.variance > 0
    assert gp.noise > 0
    assert np.isfinite(gp.log_marginal_likelihood())
