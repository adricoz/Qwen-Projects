"""Regression par processus gaussien a moyenne nulle et hyperparametres optimises."""

import numpy as np
from scipy.optimize import minimize

# Jitters (relatifs a l'ecchelle de la matrice) essayes quand la factorisation
# de Cholesky echoue.
_JITTERS = (1e-10, 1e-8, 1e-6, 1e-4)
# Valeur retournee comme NLL quand aucune factorisation ne reussit.
_NLL_FAILURE = 1e10
# Bornes des hyperparametres dans l'espace optimise (y normalise) :
# log(l) dans [-6, 6], log(variance) dans [-8, 8], log(noise) dans [-12, 0].
_LS_BOUNDS = (-6.0, 6.0)
_VAR_BOUNDS = (-8.0, 8.0)
_NOISE_BOUNDS = (-12.0, 0.0)


class GPRegressor:
    """Gaussian process regressor (zero mean) with a single ARD kernel.

    Parameters
    ----------
    kernel : gp_lite.kernels._ARDKernelBase
        Kernel object; its ``lengthscales`` and ``variance`` are the initial
        hyperparameters (replaced by the fitted ones after ``fit``).
    noise : float, optional
        Observation noise variance (default ``1e-6``), in the units of the
        (possibly normalized) output.
    normalize_y : bool, optional
        If True (default), the output is standardized with ``np.std`` (ddof=0)
        before fitting: the likelihood and ``noise`` refer to the normalized
        output, while predictions are returned in the original units.
    """

    def __init__(self, kernel, noise=1e-6, normalize_y=True):
        if noise <= 0:
            raise ValueError("noise must be strictly positive")
        self.kernel = kernel
        self.noise = float(noise)
        self.normalize_y = normalize_y
        # Remplis par fit()
        self.X_train = None
        self.y_raw = None
        self.y_mean = 0.0
        self.y_std = 1.0
        self.y_train = None  # cible (eventuellement normalisee)
        self._L = None  # factor de Cholesky de K(X, X) + noise * I
        self._alpha = None  # solve(K, y_train)

    # ------------------------------------------------------------------ #
    # Likelihood                                                         #
    # ------------------------------------------------------------------ #
    def _nll(self, log_params, X, y, d):
        """Negative log marginal likelihood of ``y`` (normalized) at ``log_params``.

        ``log_params`` holds ``[log(l_1..l_d), log(variance), log(noise)]``.
        """
        lengthscales = np.exp(log_params[:d])
        variance = float(np.exp(log_params[d]))
        noise = float(np.exp(log_params[d + 1]))

        # Les hyperparametres sont ecrits sur le noyau courant (cote-effet
        # requis par l'interface) avant l'evaluation de la vraisemblance.
        if lengthscales.ndim == 0:
            self.kernel.lengthscales = float(lengthscales)
        else:
            self.kernel.lengthscales = lengthscales
        self.kernel.variance = variance
        self.noise = noise

        K = self.kernel.K(X, X)
        K.flat[:: X.shape[0] + 1] += noise  # diagonale
        L, log_det = self._safe_cholesky(K)
        if L is None:
            return _NLL_FAILURE
        a = np.linalg.solve(L, y)
        a = np.linalg.solve(L.T, a)
        n = y.shape[0]
        return 0.5 * float(y @ a) + log_det + 0.5 * n * np.log(2.0 * np.pi)

    @staticmethod
    def _safe_cholesky(K):
        """Cholesky factorization with increasing jitters on failure.

        Returns
        -------
        (L, log_det) or (None, None) if every attempt fails.
        """
        for jitter in (0.0,) + _JITTERS:
            J = K + jitter * np.eye(K.shape[0]) if jitter else K
            try:
                L = np.linalg.cholesky(J)
                return L, 2.0 * float(np.sum(np.log(np.diag(L))))
            except np.linalg.LinAlgError:
                continue
        return None, None

    # ------------------------------------------------------------------ #
    # Fitting                                                            #
    # ------------------------------------------------------------------ #
    def fit(self, X, y, optimize=True):
        """Fit the GP on ``X`` (n, d) and ``y`` (n,).

        With ``optimize=True`` the hyperparameters (lengthscales, variance,
        noise) are maximized by L-BFGS-B on the log-likelihood in log-space,
        with 3 restarts (seeded) keeping the best one. With ``optimize=False``
        the current ``self.kernel`` / ``self.noise`` are kept and only the
        conditioning is performed.

        Returns
        -------
        self
        """
        X = np.atleast_2d(np.asarray(X, dtype=float))
        y = np.asarray(y, dtype=float).ravel()
        self.X_train = X
        self.y_raw = y
        self.y_mean = float(np.mean(y)) if self.normalize_y else 0.0
        self.y_std = float(np.std(y)) if self.normalize_y else 1.0
        if self.normalize_y and self.y_std > 0:
            self.y_train = (y - self.y_mean) / self.y_std
        else:
            self.y_mean, self.y_std = 0.0, 1.0
            self.y_train = y.copy()

        n, d = X.shape
        if optimize:
            self._optimize(d)

        # Conditionnement avec les hyperparametres courants
        K = self.kernel.K(X, X)
        K.flat[:: n + 1] += self.noise
        L, _ = self._safe_cholesky(K)
        if L is None:  # cas limite : chute sur un inverse
            Kinv = np.linalg.pinv(K)
        else:
            Kinv = None
        if Kinv is None:
            self._alpha = np.linalg.solve(L, self.y_train)
            self._alpha = np.linalg.solve(L.T, self._alpha)
        else:
            self._alpha = Kinv @ self.y_train
        self._L = L
        return self

    def _optimize(self, d):
        """3 (re)starts of L-BFGS-B on the negative log-likelihood, keep the best."""
        ls0 = self.kernel.lengthscales
        if np.ndim(ls0) == 0:
            ls0 = np.full(d, float(ls0))
        # Tout est optimise dans l'espace log : [log l, log variance, log noise]
        x0 = np.concatenate([np.log(ls0),
                             [np.log(float(self.kernel.variance))],
                             [np.log(float(self.noise))]])
        bounds = [_LS_BOUNDS] * d + [_VAR_BOUNDS, _NOISE_BOUNDS]

        rng = np.random.default_rng(42)  # graine fixee pour la reproductibilite
        starts = [x0]
        for _ in range(2):
            starts.append(np.concatenate([
                rng.uniform(_LS_BOUNDS[0], _LS_BOUNDS[1], size=d),
                [rng.uniform(*_VAR_BOUNDS), rng.uniform(*_NOISE_BOUNDS)],
            ]))

        best_x, best_fun = None, np.inf
        for x_init in starts:
            res = minimize(
                self._nll,
                x_init,
                args=(self.X_train, self.y_train, d),
                method="L-BFGS-B",
                bounds=bounds,
            )
            if res.fun < best_fun:
                best_x, best_fun = res.x, res.fun
        if best_x is not None:
            # Ecrit les hyperparametres optimaux sur self.kernel / self.noise
            self._nll(best_x, self.X_train, self.y_train, d)
        return best_fun

    # ------------------------------------------------------------------ #
    # Prediction                                                         #
    # ------------------------------------------------------------------ #
    def _check_fitted(self):
        if self.X_train is None:
            raise RuntimeError("GPRegressor is not fitted: call fit() first")

    def predict(self, X, return_std=False):
        """Posterior mean and (optionally) standard deviation of ``f`` at ``X``.

        The standard deviation is the prior posterior std of the latent
        function (observation noise excluded). Predictions are in the units
        of ``y``.

        Returns
        -------
        mean : ndarray (m,)
        std : ndarray (m,), only if ``return_std=True``
        """
        self._check_fitted()
        X = np.atleast_2d(np.asarray(X, dtype=float))
        d = self.X_train.shape[1]
        K = self.kernel.K(X, self.X_train)  # (m, n)
        k_diag = self.kernel.diag(X)        # (m,)
        mean_n = K @ self._alpha
        # var(f) = k(x,x) - k(x,X) K(X,X)^-1 k(X,x)
        if self._L is not None:
            M = np.linalg.solve(self._L, K.T)  # (n, m)
            M = np.linalg.solve(self._L.T, M)
            var_n = k_diag - np.einsum("jn,nj->j", K, M)
        else:
            Kxx = self.kernel.K(self.X_train, self.X_train)
            Kxx.flat[:: Kxx.shape[0] + 1] += self.noise
            V = np.linalg.pinv(Kxx) @ K.T
            var_n = k_diag - np.einsum("jn,nj->j", K, V)
        mean = self.y_mean + self.y_std * mean_n
        if not return_std:
            return mean
        std = self.y_std * np.sqrt(np.maximum(var_n, 0.0))
        return mean, std

    # ------------------------------------------------------------------ #
    # Vraisemblance / leave-one-out                                      #
    # ------------------------------------------------------------------ #
    def log_marginal_likelihood(self):
        """Log marginal likelihood of the training data at the current
        hyperparameters (on the normalized output if ``normalize_y``)."""
        self._check_fitted()
        d = self.X_train.shape[1]
        if self._L is None:
            return -1e10
        y = self.y_train
        a = np.linalg.solve(self._L, y)
        a = np.linalg.solve(self._L.T, a)
        n = y.shape[0]
        return -(
            0.5 * float(y @ a)
            + 2.0 * float(np.sum(np.log(np.diag(self._L))))
            + 0.5 * n * np.log(2.0 * np.pi)
        )

    def loo(self):
        """Closed-form leave-one-out predictions (Rasmussen & Williams, 5.4.2).

        For each training point ``i``, the mean and variance of the predictive
        distribution of ``y_i`` (noise included) given all the other points,
        in the units of ``y`` (with the normalization of the full training
        set). No refitting is performed.

        Returns
        -------
        (mean, var) : tuple of ndarray (n,)
        """
        self._check_fitted()
        X, y_n, d = self.X_train, self.y_train, self.X_train.shape[1]
        K = self.kernel.K(X, X)
        K.flat[:: K.shape[0] + 1] += self.noise
        A = np.linalg.inv(K)                 # K^-1
        alpha = A @ y_n
        s = np.diag(A)
        # mean_i = y_i - alpha_i / A_ii (equivalent closed form of R&Wil 5.4.2)
        means_n = y_n - alpha / s
        # var_i (bruit inclus) = K_ii - k_i^T K_-i^-1 k_i = 1 / A_ii
        # (identite de l'inversion par blocs ; equivalente a la formule de R&Wil 5.4.2)
        vars_n = 1.0 / s
        means = self.y_mean + self.y_std * means_n
        vars_ = self.y_std ** 2 * vars_n
        return means, vars_
