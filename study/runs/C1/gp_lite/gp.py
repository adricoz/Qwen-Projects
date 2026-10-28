"""Gaussian process regressor (zero-mean, homoscedastic Gaussian noise)."""

import numpy as np
from scipy.linalg import cholesky, solve_triangular

# Jitters absolus tries en cascade quand la factorisation de Cholesky echoue.
JITTERS = (0.0, 1e-10, 1e-8, 1e-6, 1e-4)
# Valeur de -log-likelihood renvoyee quand la covariance n'est pas factorisable.
NLL_FAILURE = 1e10
# Bornes (en espace logarithmique) de l'optimisation des hyperparametres.
_BOUNDS = {
    "lengthscale": (np.log(1e-4), np.log(1e4)),
    "variance": (np.log(1e-8), np.log(1e8)),
    "noise": (np.log(1e-12), np.log(1e-2)),
}
N_RESTARTS = 3  # redemarrages aleatoires supplementaires (graine fixee)
SEED = 0


def _safe_cholesky(matrix):
    """Cholesky decomposition with an increasing diagonal jitter.

    Returns
    -------
    tuple
        ``(L, jitter)`` with L lower triangular (L @ L.T = matrix).
    """
    n = matrix.shape[0]
    for jitter in JITTERS:
        try:
            L = cholesky(matrix + jitter * np.eye(n), lower=True)
            # NB : selon la version de SciPy, cholesky(lower=True) peut renvoyer
            # le facteur SUPERIEUR U ; on normalise ici un L inferieur (L @ L.T = K).
            if np.count_nonzero(np.tril(L, -1)) == 0 and np.count_nonzero(np.triu(L, 1)) > 0:
                L = L.T
            return L, jitter
        except np.linalg.LinAlgError:
            continue
    raise np.linalg.LinAlgError("Cholesky failed even with the largest jitter")


class GPRegressor:
    """Zero-mean Gaussian process regressor with an ARD stationary kernel.

    Parameters
    ----------
    kernel : gp_lite.kernels.StationaryKernel
        Kernel object; its public attributes ``lengthscales`` and ``variance``
        are the hyperparameters optimized by :meth:`fit`.
    noise : float, optional
        Observation noise variance, by default 1e-6.
    normalize_y : bool, optional
        If True (default) the outputs are standardized (mean / ``np.std``)
        before conditioning; predictions are returned in the original units.
    """

    def __init__(self, kernel, noise=1e-6, normalize_y=True):
        self.kernel = kernel
        self.noise = float(noise)
        self.normalize_y = bool(normalize_y)
        self.X_train = None
        self.y_train = None
        self._y_mean = 0.0
        self._y_std = 1.0
        self._y_n = None
        self._L = None
        self._alpha = None

    # ------------------------------------------------------------------ #
    # normalisation                                                       #
    # ------------------------------------------------------------------ #
    def _standardize(self, y):
        """Standardize y (if requested) and store mean / std of the set."""
        if self.normalize_y:
            self._y_mean = float(np.mean(y))
            self._y_std = float(np.std(y))
            if self._y_std == 0.0:
                self._y_std = 1.0
            return (y - self._y_mean) / self._y_std
        self._y_mean = 0.0
        self._y_std = 1.0
        return y

    # ------------------------------------------------------------------ #
    # vraisemblance                                                       #
    # ------------------------------------------------------------------ #
    def _negative_log_mll(self, theta, y_n):
        """Negative log marginal likelihood of y_n at (log) parameters theta.

        theta = (log l_1..log l_d, log sigma^2, log sigma_n^2).
        """
        d = self.kernel.dim
        self.kernel.lengthscales = np.exp(theta[:d])
        self.kernel.variance = float(np.exp(theta[d]))
        noise = float(np.exp(theta[d + 1]))
        n = len(y_n)
        try:
            K = self.kernel.K(self.X_train, self.X_train) + noise * np.eye(n)
            L, _ = _safe_cholesky(K)
            beta = np.linalg.solve(L, y_n)
        except (np.linalg.LinAlgError, OverflowError, FloatingPointError):
            return NLL_FAILURE
        nll = (0.5 * beta @ beta
               + np.sum(np.log(np.diag(L)))
               + 0.5 * n * np.log(2.0 * np.pi))
        return float(nll)

    def log_marginal_likelihood(self):
        """Log marginal likelihood of the training data at the current
        hyperparameters (on standardized y if ``normalize_y`` was used).

        Returns
        -------
        float
            log p(y_n | X)
        """
        if self.X_train is None:
            raise RuntimeError("fit() must be called before "
                               "log_marginal_likelihood()")
        d = self.kernel.dim
        theta = np.array([
            *np.log(np.maximum(self.kernel.lengthscales, 1e-12)),
            np.log(max(float(self.kernel.variance), 1e-12)),
            np.log(max(float(self.noise), 1e-12)),
        ])
        return -self._negative_log_mll(theta, self._y_n)

    # ------------------------------------------------------------------ #
    # apprentissage                                                       #
    # ------------------------------------------------------------------ #
    def _optimize(self, y_n):
        """Maximize the log marginal likelihood with random restarts.

        Returns
        -------
        tuple
            ``(lengthscales, variance, noise)`` of the best solution found.
        """
        from scipy.optimize import minimize
        d = self.kernel.dim
        bounds = [_BOUNDS["lengthscale"]] * d \
            + [_BOUNDS["variance"], _BOUNDS["noise"]]
        theta0 = np.array([
            *np.log(np.maximum(self.kernel.lengthscales, 1e-12)),
            np.log(max(float(self.kernel.variance), 1e-12)),
            np.log(max(float(self.noise), 1e-12)),
        ])
        rng = np.random.default_rng(SEED)
        starts = [theta0]
        for _ in range(N_RESTARTS):
            starts.append(theta0 + rng.normal(0.0, 0.5, size=len(theta0)))

        best_nll, best_theta = np.inf, theta0
        for start in starts:
            res = minimize(
                self._negative_log_mll, start,
                args=(y_n,), method="L-BFGS-B", bounds=bounds,
            )
            if res.fun < best_nll:
                best_nll, best_theta = res.fun, res.x
        ls = np.exp(best_theta[:d])
        return ls, float(np.exp(best_theta[d])), float(np.exp(best_theta[d + 1]))

    def _condition(self):
        """Store the Cholesky factor L and alpha = K^-1 y_n of the model."""
        n = len(self._y_n)
        K = self.kernel.K(self.X_train, self.X_train) \
            + self.noise * np.eye(n)
        self._L, _ = _safe_cholesky(K)
        # alpha = K^-1 y_n, K = L L^T  =>  alpha = (L^T)^-1 L^-1 y_n
        self._alpha = np.linalg.solve(self._L.T,
                                      np.linalg.solve(self._L, self._y_n))

    def fit(self, X, y, optimize=True):
        """Fit the Gaussian process on (X, y).

        Parameters
        ----------
        X : array_like
            Training inputs, shape (n, d).
        y : array_like
            Training outputs, shape (n,).
        optimize : bool, optional
            If True (default), optimize the kernel hyperparameters and the
            noise by maximizing the log marginal likelihood (L-BFGS-B in
            log-space, 3 seeded random restarts, best kept). If False, the
            current ``self.kernel`` / ``self.noise`` are used as-is.

        Returns
        -------
        GPRegressor
            ``self``.
        """
        self.X_train = np.atleast_2d(np.asarray(X, dtype=float))
        self.y_train = np.asarray(y, dtype=float).ravel()
        self._y_n = self._standardize(self.y_train)

        if optimize:
            ls, var, noise = self._optimize(self._y_n)
            self.kernel.lengthscales = ls
            self.kernel.variance = var
            self.noise = noise
        self._condition()
        return self

    # ------------------------------------------------------------------ #
    # prediction                                                          #
    # ------------------------------------------------------------------ #
    def predict(self, X, return_std=False):
        """Posterior mean (and std of f, noise excluded) at X.

        Parameters
        ----------
        X : array_like
            Test inputs, shape (m, d).
        return_std : bool, optional
            If True, also return the posterior standard deviation.

        Returns
        -------
        numpy.ndarray or tuple of numpy.ndarray
            Mean of shape (m,) in the original units of y; plus the std if
            ``return_std``.
        """
        if self.X_train is None:
            raise RuntimeError("fit() must be called before predict()")
        X = np.atleast_2d(np.asarray(X, dtype=float))
        k_star = self.kernel.K(self.X_train, X)  # (n, m)
        mean_n = k_star.T @ self._alpha
        mean = mean_n * self._y_std + self._y_mean
        if not return_std:
            return mean
        # v = L^-1 k* (une seule resolution triangulaire) ; var = k(x*,x*) - v.v
        v = solve_triangular(self._L, k_star, lower=True)
        var_n = self.kernel.diag(X) - np.sum(v ** 2, axis=0)
        std = np.sqrt(np.maximum(var_n, 0.0)) * self._y_std
        return mean, std

    # ------------------------------------------------------------------ #
    # leave-one-out                                                       #
    # ------------------------------------------------------------------ #
    def loo(self):
        """Closed-form leave-one-out predictive moments (R&W eq. 5.12).

        For each training point i, the predictive mean and variance of y_i
        (noise included) are computed from the diagonal of K^-1, without
        refitting n models.

        Returns
        -------
        tuple of numpy.ndarray
            ``(mean, var)`` of shape (n,), in the original units of y.
        """
        if self.X_train is None:
            raise RuntimeError("fit() must be called before loo()")
        n = len(self._y_n)
        # [K^-1]_ii : diagonale de l'inverse de K (n points d'entrainement).
        K = self._L @ self._L.T
        k_inv_diag = np.diag(np.linalg.inv(K))
        mean_n = self._y_n - self._alpha / k_inv_diag
        var_n = 1.0 / k_inv_diag
        mean = mean_n * self._y_std + self._y_mean
        var = var_n * self._y_std ** 2
        return mean, var
