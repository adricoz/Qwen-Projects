"""Tests pytest du paquet `gp_lite`."""
import numpy as np

from gp_lite import GPRegressor, Matern32ARD, Matern52ARD, SquaredExponentialARD


# --------------------------------------------------------------------- fixtures
def test_kernel_symmetry_and_diagonal():
    """Symetrie K(X1, X2) = K(X2, X1).T et diagonale = variance."""
    rng = np.random.default_rng(1)
    X1 = rng.random((7, 3))
    X2 = rng.random((5, 3))
    l, v = [0.4, 0.8, 1.5], 2.0
    for cls in (SquaredExponentialARD, Matern32ARD, Matern52ARD):
        k = cls(l, v)
        K12 = k.K(X1, X2)
        K21 = k.K(X2, X1)
        assert np.allclose(K12, K21.T, rtol=1e-12, atol=1e-12)
        # diagonale de K(X1, X1) = variance
        assert np.allclose(np.diag(k.K(X1, X1)), v, rtol=1e-12, atol=1e-14)
        # diag()
        assert np.allclose(k.diag(X2), np.full(5, v))
        # positivement definie
        w = rng.random((7,))
        assert w @ k.K(X1, X1) @ w > 0


def test_matern52_zero_distance():
    """Materne 5/2 vaut sigma^2 a r = 0 (id. 3/2 et SE)."""
    k52 = Matern52ARD([0.5, 2.0], variance=3.0)
    k32 = Matern32ARD([0.5, 2.0], variance=3.0)
    se = SquaredExponentialARD([0.5, 2.0], variance=3.0)
    for k in (k52, k32, se):
        x = np.array([[0.1, 0.7]])
        assert np.isclose(k.K(x, x)[0, 0], 3.0, rtol=0, atol=1e-12)


def test_matern52_formula_value():
    """Valeur explicite de Matern 5/2 en une distance connue (1D)."""
    k = Matern52ARD([1.0], variance=2.0)
    x, xp = np.array([[0.0]]), np.array([[0.3]]).astype(float)
    r = 0.3  # l = 1 donc r = |x - x'|
    expected = 2.0 * (1.0 + np.sqrt(5) * r + 5 * r**2 / 3) * np.exp(-np.sqrt(5) * r)
    assert np.isclose(k.K(x, xp)[0, 0], expected, rtol=1e-12)


# ----------------------------------------------------------------------- predict
def test_predict_interpolates_at_zero_noise():
    """A bruit quasi nul, predict interpole les points d'apprentissage."""
    rng = np.random.default_rng(2)
    X = rng.random((12, 2))
    y = np.exp(X[:, 0]) * (1.0 + X[:, 1])
    kernel = Matern52ARD([0.25, 0.35], variance=1.0)
    gp = GPRegressor(kernel, noise=1e-9, normalize_y=False)
    gp.condition(X, y)  # hyperparametres fixes, bruit quasi nul
    mean = gp.predict(X)
    assert np.max(np.abs(mean - y)) < 1e-6


def test_predict_shape_and_std_positive():
    """predict renvoie la bonne forme et un ecart-type >= 0."""
    rng = np.random.default_rng(3)
    X = rng.random((10, 2))
    y = rng.random(10)
    gp = GPRegressor(Matern52ARD([0.5, 0.5], 1.0), noise=1e-4).fit(X, y)
    Xt = rng.random((5, 2))
    mean, std = gp.predict(Xt, return_std=True)
    assert mean.shape == (5,)
    assert std.shape == (5,)
    assert np.all(std >= 0)


# ------------------------------------------------------------------------- loo
def test_loo_matches_n_refits():
    """loo() en forme fermee = n GP conditionnes sans le point i (hyperparam. fixes)."""
    rng = np.random.default_rng(4)
    X = rng.random((15, 2))
    y = np.sin(4 * X[:, 0]) + X[:, 1]
    # hyperparametres fixes (pas de refit ni de re-optimisation)
    kernel = Matern52ARD([0.18, 0.22], variance=1.0)
    gp = GPRegressor(kernel, noise=1e-2, normalize_y=False)
    gp.condition(X, y)

    loo_mean, loo_var = gp.loo()
    for i in range(X.shape[0]):
        mask = np.arange(X.shape[0]) != i
        # GP sans le point i, memes hyperparametres
        sub = GPRegressor(Matern52ARD([0.18, 0.22], variance=1.0), noise=1e-2,
                          normalize_y=False)
        sub.condition(X[mask], y[mask])
        m_i, s_i = sub.predict(X[i:i + 1], return_std=True)
        # tolerees absolues un peu plus grandes: K est mal conditionnee (points
        # proches, bruit faible), les deux calculs sont corrects a ~1e-10 pres
        assert np.isclose(loo_mean[i], m_i[0], atol=1e-5, rtol=1e-5), (i, loo_mean[i], m_i[0])
        # loo renvoie la variance POSTERIEURE LATENTE (sans bruit); c'est
        # exactement celle que renvoie predict(return_std=True)
        assert np.isclose(loo_var[i], s_i[0]**2, atol=1e-5, rtol=1e-5)


# ------------------------------------------------------------- vs scikit-learn
def test_against_sklearn_fixed_hyperparameters():
    """Moyennes predict() = sk-learn GPR (RBF ARD + White, hyperparam. FIXES)."""
    import sklearn.gaussian_process as skgp
    from sklearn.gaussian_process.kernels import RBF, WhiteKernel

    rng = np.random.default_rng(5)
    X = rng.uniform(0.0, 1.0, size=(25, 2))
    y = 1.5 * np.sin(2 * X[:, 0]) * X[:, 1] + 0.3 * X[:, 0]

    # hyperparametres identiques, fixes, normalisation des sorties desactivee
    lengthscales = np.array([0.25, 0.45])
    variance = 1.7
    noise = 1e-3

    gpl = GPRegressor(SquaredExponentialARD(lengthscales, variance), noise=noise,
                      normalize_y=False)
    gpl.condition(X, y)

    sk = skgp.GaussianProcessRegressor(
        # sklearn: la variance du signal est le COEFFICIENT de la RBF
        kernel=(variance * RBF(length_scale=lengthscales,
                               length_scale_bounds=(lengthscales, lengthscales))
                + WhiteKernel(noise_level=noise,
                              noise_level_bounds=(noise, noise))),
        normalize_y=False, optimizer=None, alpha=0.0)
    sk.fit(X, y)

    Xt = rng.uniform(0.0, 1.0, size=(40, 2))
    mean_gpl = gpl.predict(Xt)
    mean_sk, std_sk = sk.predict(Xt, return_std=True)
    assert np.max(np.abs(mean_gpl - mean_sk)) < 1e-6
    # et les ecart-types a posteriori aussi
    _, std_gpl = gpl.predict(Xt, return_std=True)
    assert np.max(np.abs(std_gpl - std_sk)) < 1e-6


# ------------------------------------------------------- end-to-end (Branin)
def branin(x):
    """Fonction de Branin 2D."""
    a, b = x[:, 0], x[:, 1]
    return (b - 5.1 / (4 * np.pi**2) * a**2 + 5 / np.pi * a - 6)**2 \
        + 10 * (1 - 1 / (8 * np.pi)) * np.cos(a) + 10


def lhs_points(n, d, seed=0):
    """n points LHS dans [0, 1]^d (un par cellule par dimension)."""
    rng = np.random.default_rng(seed)
    P = np.empty((n, d))
    for j in range(d):
        base = (rng.permutation(n) + rng.random(n)) / n
        P[:, j] = base
    return P


def test_branin_end_to_end():
    """Bout en bout sur Branin: RMSE normalisee < 0.1."""
    X = lhs_points(30, 2, seed=7)
    y = branin(X)
    kernel = Matern52ARD([0.35, 0.35], variance=1.0)
    gp = GPRegressor(kernel, noise=1e-6)
    gp.fit(X, y)

    Xt = lhs_points(200, 2, seed=11)
    yt = branin(Xt)
    mean = gp.predict(Xt)
    rmse = np.sqrt(np.mean((mean - yt)**2))
    scale = np.std(yt)
    assert rmse / scale < 0.1, f"RMSE normalisee = {rmse / scale:.4f}"
