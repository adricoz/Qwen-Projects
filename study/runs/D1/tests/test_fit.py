"""Verification du fit lui-meme : vraissemblance optimisee vs sklearn optimise.

Methode : sur Branin avec 3 plans aleatoires,
- la LML du paquet aux hyperparametres ajustes doit egalre la reference naive ;
- la LML optimisee du paquet ne doit pas etre nettement inferieure a celle de
  scikit-learn optimise (modele identique : RBF ARD x constante + WhiteKernel,
  normalize_y=True, optimizeurs activees), les deux calculees avec la reference
  naive (meme echelle y normalisee, ddof=0).
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gp_lite import GPRegressor, SquaredExponentialARD
from tests.reference_naive import naive_log_marginal_likelihood


def branin(x):
    """Fonction de Branin 2D sur [0, 1]^2."""
    x1, x2 = x[:, 0], x[:, 1]
    return ((x2 - 5.0 * x1 ** 2 + 5.0 * x1 - 5.0) ** 2
            + 10.0 * (np.sin(6.0 * x1) + np.sin(2.0 * x2)) + 10.0)


def random_plan(n, d, seed):
    r = np.random.default_rng(seed)
    return r.uniform(0, 1, (n, d))


@pytest.mark.parametrize("seed", [101, 202, 303])
def test_fit_lml_matches_naive_reference_and_beats_sklearn_gap(seed):
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF
    from sklearn.gaussian_process.kernels import WhiteKernel

    X = random_plan(30, 2, seed)
    y = branin(X)
    y_mean, y_std = float(y.mean()), float(y.std())
    y_n = (y - y_mean) / y_std

    # --- gp_lite optimise
    gp = GPRegressor(SquaredExponentialARD([0.3, 0.3])).fit(X, y)
    ll_pkg = gp.log_marginal_likelihood()

    # reference naive aux hyperparametres ajustes.
    # Tolerance absolue : la LML est sensible au bruit quand noise est proche
    # de sa borne basse (conditionnement ~ 1e9) : 1e-6 abs est du bruit flottant.
    ll_naive = naive_log_marginal_likelihood(
        X, y_n, gp.kernel.lengthscales, gp.kernel.variance, gp.noise, "se")
    assert abs(ll_pkg - ll_naive) < 1e-5

    # --- sklearn optimise (modele equivalent)
    sk = GaussianProcessRegressor(
        kernel=RBF([0.3, 0.3]) * 1.0 + WhiteKernel(1e-6),
        normalize_y=True,
        n_restarts_optimizer=3,
        random_state=0,
    ).fit(X, y)
    # structure sklearn : (RBF * Constant) + WhiteKernel
    rbf, const, white = sk.kernel_.k1.k1, sk.kernel_.k1.k2, sk.kernel_.k2
    ll_sk = naive_log_marginal_likelihood(
        X, y_n,
        np.asarray(rbf.length_scale, float),
        float(const.constant_value),
        float(white.noise_level),
        "se")

    # le paquet ne doit pas etre nettement plus bas
    assert ll_pkg > ll_sk - 5.0, f"gp_lite LML {ll_pkg} vs sklearn LML {ll_sk}"

    # sanity : la LML doit etre superieure a celle de l'initialisation naive
    ll_init = naive_log_marginal_likelihood(X, y_n, [0.3, 0.3], 1.0, 1e-6, "se")
    assert ll_pkg > ll_init
