"""Gaussian process regression for `gp_lite`.

Implements a GP with a constant (zero) mean, an ARD stationary kernel and
Gaussian observation noise. Hyperparameters (lengthscales, signal variance and
noise) are fitted by maximizing the (marginal) log-likelihood with L-BFGS-B in
log-space, with random restarts and a jitter fallback for the Cholesky
factorization. Prediction and leave-one-out residuals use the closed forms of
Rasmussen & Williams (2006), sections 2.3 and 5.4.2.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import cholesky, solve_triangular
from scipy.optimize import minimize


class GPRegressor:
    """Gaussian process regressor (zero mean) with a stationary ARD kernel.

    Parameters
    ----------
    kernel : gp_lite.kernels.ARDStationaryKernel
        Kernel object with `K(X1, X2)`, `diag(X)` and `set_params` methods.
    noise : float, default 1e-6
        Initial observation noise variance (optimized in log-space).
    normalize_y : bool, default True
        If True, y is standardized (zero mean / unit variance) before the
        likelihood is optimized and predictions are denormalized. This makes
        the optimization robust to the scale of the outputs.
    """

    _JITTERS = (0.0, 1e-10, 1e-8, 1e-6)
    _NLL_FAILURE = 1e10

    def __init__(self, kernel, noise: float = 1e-6, normalize_y: bool = True):
        self.kernel = kernel
        self.noise = float(noise)
        self.normalize_y = normalize_y
        self._fitted = False

    # ------------------------------------------------------------------ utils
    def _covariance(self, X: np.ndarray, noise: float) -> np.ndarray:
        """Covariance matrix K(X, X) + noise * I (for the given noise)."""
        cov = self.kernel.K(X, X)
        if noise > 0.0:
            cov = cov + noise * np.eye(cov.shape[0])
        return cov

    def _safe_cholesky(self, cov: np.ndarray):
        """Cholesky factor of `cov`, increasing jitter until it succeeds.

        Returns
        -------
        tuple (L, jitter)
            `L` the lower triangular factor and the relative jitter used
            (0.0 if the first attempt worked).
        """
        scale = np.max(np.abs(np.diag(cov))) if cov.size else 1.0
        for rel in self._JITTERS:
            try:
                return cholesky(cov + rel * scale * np.eye(cov.shape[0]),
                                lower=True, check_finite=False), rel
            except np.linalg.LinAlgError:
                continue
        # dernier recours: jitter plus gros (stabilise une matrice mal conditionnee)
        jitter = 1e-4 * scale
        return cholesky(cov + jitter * np.eye(cov.shape[0]), lower=True,
                        check_finite=False), jitter

    def _neg_log_likelihood(self, log_params: np.ndarray, X: np.ndarray,
                            y: np.ndarray) -> float:
        """Negative marginal log-likelihood for log[lengthscales, log var, log noise].

        Parameters
        ----------
        log_params : np.ndarray, shape (d + 2,)
            Log of the lengthscales, the signal variance and the noise.
        X : np.ndarray, shape (n, d)
            Training inputs.
        y : np.ndarray, shape (n,)
            (Normalized) training targets.

        Returns
        -------
        float
            Negative log marginal likelihood, or 1e10 if the factorization
            fails even with jitter.
        """
        if np.any(np.abs(log_params) > 50.0):  # evite exp(overflow) hors bornes
            return self._NLL_FAILURE
        params = np.exp(log_params)
        self.kernel.set_params(params[:-2], params[-2])
        noise = params[-1]
        try:
            cov = self._covariance(X, noise)
            L, _ = self._safe_cholesky(cov)
        except np.linalg.LinAlgError:
            return self._NLL_FAILURE
        alpha = solve_triangular(L, y, lower=True, check_finite=False)
        n = y.size
        nll = (0.5 * float(y @ alpha)
               + np.sum(np.log(np.diag(L)))
               + 0.5 * n * np.log(2.0 * np.pi))
        if not np.isfinite(nll):
            return self._NLL_FAILURE
        return float(nll)

    # ----------------------------------------------------------------------- fit
    def fit(self, X: np.ndarray, y: np.ndarray) -> "GPRegressor":
        """Fit the hyperparameters by maximizing the marginal log-likelihood.

        The parameters log(lengthscales), log(variance) and log(noise) are
        optimized with L-BFGS-B (scipy.optimize.minimize), with 3 random
        restarts (fixed seed) and keeping the best result. The Cholesky
        factor of the final covariance matrix is cached for `predict` and `loo`.

        Parameters
        ----------
        X : array_like, shape (n, d)
            Training inputs.
        y : array_like, shape (n,)
            Training targets.

        Returns
        -------
        self
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        n, d = X.shape
        if d != self.kernel.dimension:
            raise ValueError(f"kernel expects {self.kernel.dimension} dimensions, "
                             f"got {d}")

        # normalisation des sorties (rend la recherche de hyperparametres stable)
        y_opt = self._normalize(y)

        # bornes en espace log, adaptees a l'echele des donnees
        l_lo, l_hi = -10.0, 10.0
        v_lo, v_hi = -12.0, 12.0
        bnd_lo = np.full(d + 2, -1e4)
        bnd_hi = np.full(d + 2, 1e4)
        bnd_lo[:d], bnd_hi[:d] = l_lo, l_hi
        bnd_lo[d], bnd_hi[d] = v_lo, v_hi

        rng = np.random.default_rng(42)
        best = None
        for _ in range(3):
            x0 = np.concatenate([
                rng.uniform(-2.0, 2.0, d),       # log(lengthscales) ~ U(1/e^2, e^2)
                [rng.uniform(v_lo / 4, v_hi / 4)],
                [np.log(max(self.noise, 1e-12)) + rng.uniform(-3.0, 3.0)],
            ])
            res = minimize(self._neg_log_likelihood, x0, args=(X, y_opt),
                           method="L-BFGS-B", bounds=list(zip(bnd_lo, bnd_hi)),
                           options={"maxiter": 500, "ftol": 1e-10})
            if best is None or res.fun < best.fun:
                best = res

        if best is None or not np.isfinite(best.fun) or best.fun >= self._NLL_FAILURE:
            raise RuntimeError("fit failed: could not find finite log-likelihood")

        params = np.exp(best.x)
        self.kernel.set_params(params[:-2], params[-2])
        self.noise_ = float(params[-1])
        # conditionnement avec les hyperparametres optimaux (normalisation interne)
        self.condition(X, y)
        self._ll = -best.fun
        self._fitted = True
        return self

    def _normalize(self, y: np.ndarray) -> np.ndarray:
        """Standardize y if `normalize_y` is set; stores mean/std for denormalization."""
        self.y_mean_ = float(y.mean())
        self.y_std_ = float(y.std())
        if self.normalize_y and self.y_std_ > 0:
            return (y - self.y_mean_) / self.y_std_
        self.y_mean_ = 0.0
        self.y_std_ = 1.0
        return y

    def condition(self, X: np.ndarray, y: np.ndarray) -> "GPRegressor":
        """Condition the process on (X, y) with the CURRENT hyperparameters.

        No hyperparameter optimization is performed: the kernel parameters and
        `noise_` of the object are taken as they are. The Cholesky factor and
        alpha = K^-1 y are cached for `predict` and `loo`.

        Parameters
        ----------
        X : array_like, shape (n, d)
            Training inputs.
        y : array_like, shape (n,)
            Training targets (already in the target space; normalization is
            applied if `normalize_y` is set).

        Returns
        -------
        self
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        y_opt = self._normalize(y)
        self.X_train_ = X
        self.y_ = y
        self._y_opt = y_opt
        noise = getattr(self, "noise_", self.noise)
        self.noise_ = noise
        self._cov = self._covariance(X, noise)
        self._L, self._jitter = self._safe_cholesky(self._cov)
        # solve exact (LDL/pivoting) : stable meme quand le bruit est quasi nul
        # et que K est presque singulier (interpolation)
        self._alpha = np.linalg.solve(self._cov, y_opt)
        self._ll = -self._neg_log_likelihood(
            np.concatenate([np.log(self.kernel.lengthscales),
                            [np.log(self.kernel.variance), np.log(self.noise_)]]),
            X, y_opt)
        self._fitted = True
        return self

    # --------------------------------------------------------------- prediction
    def log_marginal_likelihood(self) -> float:
        """Return the marginal log-likelihood at the fitted hyperparameters.

        Raises
        ------
        RuntimeError
            If the model has not been fitted.
        """
        if not self._fitted:
            raise RuntimeError("fit() must be called first")
        return float(self._ll)

    def predict(self, X: np.ndarray, return_std: bool = False):
        """Posterior mean and (optionally) standard deviation at new points.

        With K = K(X, X) + sigma_n^2 I, K* = K(X*, X):
            mean    = K* K^-1 (y - mu) + mu
            var     = K(X*, X*) - K* K^-1 K*^T

        Parameters
        ----------
        X : array_like, shape (m, d)
            Points where to predict.
        return_std : bool, default False
            If True, also return the posterior standard deviation.

        Returns
        -------
        mean : np.ndarray, shape (m,)
            Posterior mean in the original units of y.
        std : np.ndarray, shape (m,), optional
            Posterior standard deviation (only if `return_std`).
        """
        if not self._fitted:
            raise RuntimeError("fit() must be called first")
        Xs = np.atleast_2d(np.asarray(X, dtype=float))
        kstar = self.kernel.K(Xs, self.X_train_)          # (m, n)
        mean_opt = kstar @ self._alpha
        if return_std:
            v = self.kernel.diag(Xs) + self.noise_
            # var = diag - diag(k* K^-1 k*^T) (+ bruit, variance d'une observation).
            # On calcule la diagonale via le produit complet de matrices (robuste)
            A = kstar @ np.linalg.solve(self._cov, kstar.T)
            var_opt = v - np.diag(A)
            scale = self.y_std_ if self.normalize_y else 1.0
            mean = mean_opt * self.y_std_ + self.y_mean_
            std = np.sqrt(np.maximum(var_opt, 0.0)) * scale
            return mean, std
        return mean_opt * self.y_std_ + self.y_mean_

    # ---------------------------------------------------------------------- LOO
    def loo(self) -> tuple[np.ndarray, np.ndarray]:
        """Leave-one-out residuals and variances, in closed form.

        From Rasmussen & Williams (2006), section 5.4.2, with alpha = K^-1 y and
        v_i = (K^-1)_{ii} = 1 / (L L^T)_{ii}:
            mu_loo_i  = y_i - alpha_i / v_i
            var_loo_i = 1 / v_i
        No refitting of the hyperparameters: the same cached K^-1 is used.

        Returns
        -------
        loo_pred : np.ndarray, shape (n,)
            LOO predictions in the original units of y.
        loo_var : np.ndarray, shape (n,)
            LOO prediction variances (original units).
        """
        if not self._fitted:
            raise RuntimeError("fit() must be called first")
        n = self.X_train_.shape[0]
        # v_i = (K^-1)_{ii}, calculee par une resolution complete (robuste,
        # identique a ce que fait `predict` pour rester coherent avec le test)
        minv = np.linalg.solve(self._cov, np.eye(n))
        v_diag = np.diag(minv)
        mu_opt = self._y_opt - self._alpha / v_diag
        var_opt = 1.0 / v_diag
        scale = self.y_std_ if self.normalize_y else 1.0
        mu = mu_opt * self.y_std_ + self.y_mean_
        return mu, var_opt * scale**2
