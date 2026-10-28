"""Gaussian process regression for gp_lite (NumPy + SciPy).

Model : f ~ GP(0, k), y = f + eps, eps ~ N(0, sigma_n^2 I).
All hyperparameters are optimised in log-space on the negative log marginal
likelihood, with a jitter fallback when the Cholesky factorisation fails.
Only ``scipy.optimize.minimize`` is used from SciPy; linear algebra is done
with NumPy (np.linalg.cholesky / np.linalg.solve) for robustness.
"""
import numpy as np
from scipy.optimize import minimize

# Jitters (additifs sur la diagonale) tries when Cholesky fails, then a last big one.
_JITTERS = (0.0, 1e-10, 1e-8, 1e-6, 1e-4)
# NLL returned when the covariance matrix cannot be factorised at all.
_NLL_FAILURE = 1e10
# Bounds of the log hyperparameters (lengthscales, variance, noise in [1e-3, 1e3]).
_LOG_BOUNDS = (-7.0, 7.0)


def _solve_LL(L, b):
    """Solve (L L^T) x = b from the lower Cholesky factor L of the system."""
    return np.linalg.solve(L.T, np.linalg.solve(L, b))


class GPRegressor:
    """Gaussian process regression with a zero mean function.

    Parameters
    ----------
    kernel : gp_lite.kernels._ARDKernel
        Stationary ARD kernel; its ``lengthscales`` and ``variance`` are the
        initial hyperparameters (and carry the optimised values after ``fit``).
    noise : float, optional
        Observation noise variance (default 1e-6).
    normalize_y : bool, optional
        If True, the target is standardised (ddof = 0) before conditioning;
        the likelihood and ``noise`` are expressed on the standardised target
        and predictions are returned in the units of ``y`` (default True).

    Attributes
    ----------
    X_train, y_train : ndarray
        Training data (raw units) stored after ``fit``.
    y_mean, y_std : float
        Target mean / standard deviation used by the normalisation.
    """

    def __init__(self, kernel, noise=1e-6, normalize_y=True):
        self.kernel = kernel
        self.noise = float(noise)
        self.normalize_y = bool(normalize_y)
        self.X_train = None
        self.y_train = None
        self.y_n = None
        self.y_mean = 0.0
        self.y_std = 1.0

    # ------------------------------------------------------------------ #
    # internal helpers
    # ------------------------------------------------------------------ #
    def _standardize(self, y):
        """Return (y, mean, std) with the target standardised when enabled."""
        if not self.normalize_y:
            return y, 0.0, 1.0
        mean = float(y.mean())
        std = float(y.std())
        if std == 0.0:
            std = 1.0
        return (y - mean) / std, mean, std

    def _factor(self):
        """Lower Cholesky factor of K + noise*I, with a jitter fallback."""
        n = self.X_train.shape[0]
        K = self.kernel.K(self.X_train, self.X_train)
        for jitter in _JITTERS:
            A = K + (self.noise + jitter) * np.eye(n)
            try:
                return np.linalg.cholesky(A)
            except np.linalg.LinAlgError:
                continue
        # dernier recours : grosse regularisation pour rester utilisable
        A = K + (self.noise + 1.0) * np.eye(n)
        return np.linalg.cholesky(A)

    def _nll(self, X, y):
        """Negative log marginal likelihood, with a jitter fallback.

        Returns (nll, factorisation) where factorisation is None on failure.
        """
        n = len(y)
        K = self.kernel.K(X, X)
        for jitter in _JITTERS:
            A = K + (self.noise + jitter) * np.eye(n)
            try:
                L = np.linalg.cholesky(A)
            except np.linalg.LinAlgError:
                continue
            alpha = _solve_LL(L, y)
            # logdet = 2 * sum(log(diag(L)))
            logdet = 2.0 * np.sum(np.log(np.diag(L)))
            nll = (0.5 * (y @ alpha + logdet)
                   + 0.5 * n * np.log(2.0 * np.pi))
            return nll, L
        return _NLL_FAILURE, None

    # ------------------------------------------------------------------ #
    # public API
    # ------------------------------------------------------------------ #
    def fit(self, X, y, optimize=True):
        """Condition on the data, optionally optimising the hyperparameters.

        Parameters
        ----------
        X : ndarray, shape (n, d)
            Training inputs.
        y : ndarray, shape (n,)
            Training targets.
        optimize : bool, optional
            If True (default), maximise the log marginal likelihood of
            [lengthscales, variance, noise] with L-BFGS-B in log-space,
            3 seeded random restarts, keeping the best. If False, keep the
            current ``self.kernel`` / ``self.noise`` and only condition.

        Returns
        -------
        GPRegressor
            ``self``, so the call can be chained.
        """
        X = np.atleast_2d(np.asarray(X, dtype=float))
        y = np.asarray(y, dtype=float).ravel()
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"X has {X.shape[0]} rows but y has {y.shape[0]}")
        self.X_train = X
        self.y_train = y
        self.y_n, self.y_mean, self.y_std = self._standardize(y)

        if optimize:
            self._optimize()
        return self

    def _optimize(self):
        """Maximum-likelihood fit in log-space, 3 seeded restarts, keep best."""
        d = self.X_train.shape[1]
        l0 = np.log(np.asarray(self.kernel.lengthscales, dtype=float))
        v0 = np.log(float(self.kernel.variance))
        n0 = np.log(float(self.noise))
        theta0 = np.concatenate([l0, [v0, n0]])
        lo, hi = _LOG_BOUNDS
        bounds = [(lo, hi)] * (d + 2)
        X, yn = self.X_train, self.y_n

        def nll(theta):
            self.kernel.lengthscales = np.exp(theta[:d])
            self.kernel.variance = float(np.exp(theta[d]))
            self.noise = float(np.exp(theta[d + 1]))
            val, _ = self._nll(X, yn)
            return val

        # Redemarrages : l'initialisation est conservee, puis des points reparties
        # en espace log (les echelles de Branin sont tres differentes des initiales).
        # Graine fixee pour la reproductibilite.
        rng = np.random.default_rng(0)
        starts = [theta0]
        for _ in range(9):
            theta = lo + (hi - lo) * rng.random(theta0.shape)
            starts.append(np.clip(theta, lo, hi))
        best_theta, best_val = None, np.inf
        for theta in starts:
            res = minimize(nll, theta, method="L-BFGS-B", bounds=bounds,
                           options={"maxiter": 600, "ftol": 1e-14, "gtol": 1e-8})
            val = res.fun if np.isfinite(res.fun) else np.inf
            if val < best_val:
                best_val, best_theta = val, np.asarray(res.x)

        # Re-appliquer le meilleur ensemble de parametres
        self.kernel.lengthscales = np.exp(best_theta[:d])
        self.kernel.variance = float(np.exp(best_theta[d]))
        self.noise = float(np.exp(best_theta[d + 1]))

    def log_marginal_likelihood(self):
        """Log marginal likelihood of the data at the current hyperparameters.

        Returns
        -------
        float
            Log marginal likelihood (on the standardised target when
            ``normalize_y`` is True).
        """
        if self.X_train is None:
            raise RuntimeError("Call fit() before log_marginal_likelihood().")
        val, _ = self._nll(self.X_train, self.y_n)
        return float(-val)

    def predict(self, X, return_std=False):
        """Posterior mean (and optionally standard deviation) at X.

        Parameters
        ----------
        X : ndarray, shape (m, d)
            Query points.
        return_std : bool, optional
            If True, also return the standard deviation of f (noise-free).

        Returns
        -------
        mean : ndarray, shape (m,)
            Posterior mean in the units of the training target.
        std : ndarray, shape (m,)
            Posterior standard deviation of f (only if ``return_std``).
        """
        if self.X_train is None:
            raise RuntimeError("Call fit() before predict().")
        Xs = np.atleast_2d(np.asarray(X, dtype=float))
        L = self._factor()
        alpha = _solve_LL(L, self.y_n)
        Ks = self.kernel.K(self.X_train, Xs)  # (n, m)
        mean = (Ks.T @ alpha) * self.y_std + self.y_mean
        if not return_std:
            return mean
        v = np.sum(Ks * _solve_LL(L, Ks), axis=0)
        std = np.sqrt(np.maximum(self.kernel.variance - v, 0.0)) * self.y_std
        return mean, std

    def loo(self):
        """Closed-form leave-one-out predictive means and variances (RW 5.4.2).

        Derived by removing the i-th diagonal of A = K + sigma_n^2 I and
        inverting the resulting block matrix, everything is expressed with
        the inverse of the full system: with A^-1 = (gamma_ii on the
        diagonal) and alpha = A^-1 y,

        - mean_i = y_i - alpha_i / gamma_ii
        - var_i  = 1 / gamma_ii  (predictive variance of y_i, noise included).

        No re-fitting, no n separate conditionings.

        Returns
        -------
        mean : ndarray, shape (n,)
            Leave-one-out predictive means of y_i, in the units of y.
        var : ndarray, shape (n,)
            Leave-one-out predictive variances of y_i (noise included),
            in the units of y.
        """
        if self.X_train is None:
            raise RuntimeError("Call fit() before loo().")
        L = self._factor()
        alpha = _solve_LL(L, self.y_n)
        n = len(self.y_n)
        Linv = np.linalg.solve(L, np.eye(n))
        gamma = np.sum(Linv * Linv, axis=0)  # (A^-1)_ii = ||col_i(L^-1)||^2
        v = 1.0 / gamma
        mu = self.y_n - v * alpha
        mean = mu * self.y_std + self.y_mean
        var = v * self.y_std**2
        return mean, var
