"""Gaussian process regression with hyperparameter optimization.

Only NumPy and SciPy are used. The negative log marginal likelihood is
minimized with L-BFGS-B over the logarithms of the hyperparameters
(lengthscales, variance, noise), with a fixed number of random restarts.
"""
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize


# Relatif jitter sur la diagonale, essayés dans l'ordre quand la factorisation échoue
_JITTERS = (0.0, 1e-10, 1e-8, 1e-6)
# Valeur renvoyée par l'NLL quand la matrice ne peut pas être factorisée
_NLL_FAILURE = 1e10


def _cholesky_with_jitter(A):
    """Factorise ``A`` par Cholesky, en ajoutant un jitter si nécessaire.

    Parameters
    ----------
    A : ndarray, shape (n, n)
        Symmetric positive (semi-)definite matrix.

    Returns
    -------
    tuple of (ndarray factor, float jitter)
        The factor from :func:`scipy.linalg.cho_factor` (lower,
        check_finite=False) and the jitter actually applied to the diagonal.
    """
    A = np.asarray(A, dtype=float)
    for jitter in _JITTERS:
        if jitter > 0:
            A = A + jitter * np.eye(A.shape[0])
        try:
            return cho_factor(A, lower=True, check_finite=False), jitter
        except np.linalg.LinAlgError:
            continue
    raise np.linalg.LinAlgError("Cholesky factorization failed even with jitter")


class GPRegressor:
    """Gaussian process regressor with an arbitrary ARD kernel.

    Parameters
    ----------
    kernel : object
        Kernel object with ``K(X1, X2)`` and ``lengthscales`` / ``variance``
        attributes (see :mod:`gp_lite.kernels`).
    noise : float, optional
        Initial observation *variance* (variance of the Gaussian noise).
    normalize_y : bool, optional
        If ``True`` (default) the training targets are standardized
        (zero mean / unit variance) before fitting and predictions are
        mapped back to the original scale. This makes the likelihood
        optimization considerably better conditioned.
    random_state : int, optional
        Seed used for the random restarts of :meth:`fit`.
    """

    def __init__(self, kernel, noise=1e-6, normalize_y=True, random_state=0):
        if not (noise > 0):
            raise ValueError("noise must be strictly positive")
        self.kernel = kernel
        self.noise = float(noise)
        self.normalize_y = bool(normalize_y)
        self.random_state = random_state

        # Attributes filled by fit() / condition()
        self.X_train_ = None
        self.y_train_ = None          # targets as passed (raw scale)
        self.alpha_ = None            # (K + noise I)^-1 y, normalized scale
        self.L_ = None                # cho_factor result (lower, check_finite=False)
        self._y_mean = 0.0
        self._y_std = 1.0
        self._jitter = 0.0

    # ------------------------------------------------------------------ #
    # Internal helpers (all in normalized target space)
    # ------------------------------------------------------------------ #
    def _set_params_from_log(self, log_params, d):
        """Set kernel lengthscales / variance and noise from their logs."""
        self.kernel.lengthscales = np.exp(log_params[:d])
        self.kernel.variance = float(np.exp(log_params[d]))
        self.noise = float(np.exp(log_params[d + 1]))

    def _neg_log_likelihood(self, log_params, X, y, d):
        """Negative log marginal likelihood (NLL), with jitter on failure.

        Parameters
        ----------
        log_params : ndarray, shape (d + 2,)
            ``log([l_1..l_d, variance, noise])`` (noise = variance).
        X : ndarray, shape (n, d)
        y : ndarray, shape (n,), normalized targets
        d : int
            Number of input dimensions.

        Returns
        -------
        float
            The NLL, or a large penalty if the matrix cannot be factorized.
        """
        n = y.shape[0]
        self._set_params_from_log(log_params, d)
        K = self.kernel.K(X, X) + self.noise * np.eye(n)
        try:
            chol, _ = _cholesky_with_jitter(K)
        except np.linalg.LinAlgError:
            return _NLL_FAILURE
        alpha = cho_solve(chol, y, check_finite=False)
        log_det = 2.0 * np.sum(np.log(np.diag(chol[0])))
        nll = (0.5 * float(np.dot(y, alpha)) + 0.5 * log_det
               + 0.5 * n * np.log(2.0 * np.pi))
        return float(nll)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def condition(self, X, y):
        """Condition the GP on the data with the CURRENT hyperparameters.

        No optimization is performed: the kernel lengthscale(s)/variance and
        ``self.noise`` are used as they are (mainly for tests and fixed
        hyperparameters).

        Parameters
        ----------
        X : ndarray, shape (n, d)
        y : ndarray, shape (n,)

        Returns
        -------
        self
        """
        X = np.atleast_2d(np.asarray(X, dtype=float))
        y = np.asarray(y, dtype=float).ravel()
        n = X.shape[0]
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y have inconsistent numbers of points")

        # [NORM] standardize the targets: the optimization is far better conditioned
        self._y_mean = float(np.mean(y))
        self._y_std = float(np.std(y)) if y.size > 1 else 1.0
        if self.normalize_y and self._y_std > 0:
            y_norm = (y - self._y_mean) / self._y_std
        else:
            self._y_std, self._y_mean = 1.0, 0.0
            y_norm = y

        K = self.kernel.K(X, X)
        K_n = K + self.noise * np.eye(n)
        self.L_, self._jitter = _cholesky_with_jitter(K_n)
        self.alpha_ = cho_solve(self.L_, y_norm, check_finite=False)
        self.X_train_ = X
        self.y_train_ = y
        return self

    def fit(self, X, y):
        """Optimize the hyperparameters by maximizing the marginal likelihood.

        L-BFGS-B is run from 3 random restarts (seeded) over the logarithms
        of the lengthscale(s), the variance and the noise; the best result is
        kept. After optimization the model is conditioned on the data.

        Parameters
        ----------
        X : ndarray, shape (n, d)
        y : ndarray, shape (n,)

        Returns
        -------
        self
        """
        X = np.atleast_2d(np.asarray(X, dtype=float))
        y = np.asarray(y, dtype=float).ravel()
        n, d = X.shape

        self._y_mean = float(np.mean(y))
        self._y_std = float(np.std(y)) if y.size > 1 else 1.0
        if self.normalize_y and self._y_std > 0:
            y_norm = (y - self._y_mean) / self._y_std
        else:
            self._y_std, self._y_mean = 1.0, 0.0
            y_norm = y

        d_kernel = self.kernel.lengthscales.shape[0]
        if d_kernel != d:
            raise ValueError(
                f"kernel expects {d_kernel} dimensions, data has {d}")

        def objective(log_params):
            return self._neg_log_likelihood(log_params, X, y_norm, d)

        rng = np.random.default_rng(self.random_state)
        # Bornes (en espace log) des hyperparamètres
        lo = np.array([-5.0] * d_kernel + [-5.0, -20.0])
        hi = np.array([5.0] * d_kernel + [5.0, 0.0])

        best_params, best_nll = None, np.inf
        for i in range(3):
            if i == 0:
                # premier départ : valeurs courantes du kernel / du bruit
                x0 = np.log(np.concatenate(
                    [self.kernel.lengthscales,
                     [self.kernel.variance, self.noise]]))
            else:
                x0 = rng.uniform(lo, hi)
            res = minimize(objective, x0, method="L-BFGS-B",
                           bounds=list(zip(lo, hi)),
                           options={"maxiter": 300, "gtol": 1e-8})
            if res.fun < best_nll:
                best_nll, best_params = float(res.fun), res.x

        self._set_params_from_log(best_params, d)
        return self.condition(X, y)

    def _predict_normalized(self, X_star):
        """Posterior mean/std of the normalized process at new points.

        Parameters
        ----------
        X_star : ndarray, shape (m, d)

        Returns
        -------
        tuple of (ndarray (m,), ndarray (m,))
            Posterior mean and standard deviation in normalized scale.
        """
        X_star = np.atleast_2d(np.asarray(X_star, dtype=float))
        n = self.X_train_.shape[0]
        m = X_star.shape[0]
        K_s = self.kernel.K(self.X_train_, X_star)          # (n, m)
        K_ss_diag = np.diag(self.kernel.K(X_star, X_star))  # (m,)
        # K_s A^-1 : résolution par Cholesky (A X = K_s)
        A_inv_Ks = cho_solve(self.L_, K_s, check_finite=False)   # (n, m)
        mu_norm = A_inv_Ks.T @ self.alpha_                         # (m,)
        v_diag = np.einsum("ij,ij->i", A_inv_Ks, K_s)             # (m,)
        var_norm = np.maximum(K_ss_diag - v_diag, 0.0)
        return mu_norm, np.sqrt(var_norm)

    def predict(self, X, return_std=False):
        """Predict the targets at new points (posterior mean and std).

        Parameters
        ----------
        X : ndarray, shape (m, d) or (d,)
        return_std : bool, optional
            If ``True`` also return the posterior standard deviation
            (in the same scale as the mean).

        Returns
        -------
        ndarray, shape (m,)
            Posterior mean.
        ndarray, shape (m,), optional
            Posterior standard deviation, if ``return_std=True``.
        """
        if self.X_train_ is None:
            raise RuntimeError("call fit() (or condition()) before predict()")
        mu_n, std_n = self._predict_normalized(X)
        if self.normalize_y and self._y_std != 1.0:
            mu = self._y_mean + self._y_std * mu_n
            std = abs(self._y_std) * std_n
        else:
            mu, std = mu_n, std_n
        if return_std:
            return mu, std
        return mu

    def log_marginal_likelihood(self):
        """Log marginal likelihood of the training data (raw scale).

        Returns
        -------
        float
            ``-0.5 y^T (K + noise I)^-1 y - 0.5 log det(K + noise I)
            - n/2 log(2 pi)``, expressed in the original target scale.
        """
        if self.X_train_ is None:
            raise RuntimeError("call fit() (or condition()) first")
        n = self.X_train_.shape[0]
        log_det = 2.0 * np.sum(np.log(np.diag(self.L_[0])))
        # On travaille en espace normalisé puis on reporte log(std_y)
        y_norm = ((self.y_train_ - self._y_mean) / self._y_std
                  if (self.normalize_y and self._y_std != 1.0)
                  else self.y_train_)
        lll = (-0.5 * float(np.dot(y_norm, self.alpha_))
               - 0.5 * log_det - 0.5 * n * np.log(2.0 * np.pi))
        if self.normalize_y and self._y_std != 1.0:
            lll -= n * np.log(self._y_std)
        return float(lll)

    def loo(self):
        """Leave-one-out residuals in closed form (no re-fitting).

        With ``A = K + noise I``, ``alpha = A^-1 y`` and
        ``(A^-1)_ii = 1 / (k_ii + noise - v_i)`` with
        ``v_i = k_-i^T A_-ii^-1 k_-i`` (Schur complement / block inversion),
        the Rasmussen & Williams (section 5.4.2) leave-one-out residual of
        point ``i`` is  ``r_i = alpha_i / (A^-1)_ii``.
        In the zero-noise limit ``v_i = 0`` and ``(A^-1)_ii = 1/k_ii``, so
        ``r_i = k_ii alpha_i = y_i`` and the residuals vanish exactly
        (the GP interpolates its training points).

        Returns
        -------
        ndarray, shape (n,)
            Leave-one-out residuals (in the original target scale).
        """
        if self.X_train_ is None:
            raise RuntimeError("call fit() (or condition()) first")
        n = self.X_train_.shape[0]
        X = self.X_train_
        # (A^-1)_ii : P = A^-1 (résolution A P = I), la diagonale de P suffit
        P = cho_solve(self.L_, np.eye(n), check_finite=False)
        ainv_ii = np.diag(P)
        r_norm = self.alpha_ / ainv_ii
        if self.normalize_y and self._y_std != 1.0:
            return self._y_std * r_norm
        return r_norm
