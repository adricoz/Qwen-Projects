"""Gaussian process regression with a constant mean (NumPy + SciPy only)."""
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize

# Jitters (absolus) tries when the Cholesky factorization fails (fixe numerique).
JITTERS = (1e-10, 1e-8, 1e-6, 1e-4)
# Valeur renvoyee par le NLL quand la factorisation echoue (peine forte).
NLL_FAILURE = 1e10


class GPRegressor:
    """Gaussian process regressor with an ARD kernel and a constant mean.

    Parameters
    ----------
    kernel : object with K(X1, X2), lengthscales, variance
        ARD kernel instance whose hyperparameters are updated in place by :meth:`fit`.
    noise : float, optional
        Observation noise variance (default 1e-6).
    normalize_y : bool, optional
        If True (default), train on ``y_n = (y - mean) / std`` (ddof=0); the
        likelihood and ``noise`` are expressed in ``y_n`` units and predictions
        are returned in the original ``y`` units.

    Attributes
    ----------
    noise : float
        Noise variance (updated by :meth:`fit` when ``optimize=True``).
    """

    def __init__(self, kernel, noise=1e-6, normalize_y=True):
        self.kernel = kernel
        self.noise = float(noise)
        self.normalize_y = bool(normalize_y)
        self.x_train = None
        self.y_train = None
        self._y_mean = None
        self._y_std = None
        self._L = None
        self._alpha = None
        self._seed = 0

    # ------------------------------------------------------------------ #
    # hyperparameter bookkeeping
    # ------------------------------------------------------------------ #
    def _param_vector(self):
        """Log-space hyperparameters: [log l_1..l_d, log variance, log noise]."""
        return np.concatenate([
            np.log(self.kernel.lengthscales),
            [np.log(self.kernel.variance), np.log(self.noise)],
        ])

    def _set_params(self, log_params):
        """Write a log-space parameter vector back into ``self.kernel`` and ``self.noise``."""
        d = self.kernel.lengthscales.size
        self.kernel.lengthscales = np.exp(log_params[:d])
        self.kernel.variance = float(np.exp(log_params[d]))
        self.noise = float(np.exp(log_params[d + 1]))

    # ------------------------------------------------------------------ #
    # likelihood
    # ------------------------------------------------------------------ #
    def _nll(self, y_n):
        """
        Negative log marginal likelihood of ``y_n`` at the current hyperparameters.

        Uses a Cholesky factorization; a diagonal jitter is added if it fails.
        """
        X = self.x_train
        n = y_n.size
        K = self.kernel.K(X, X) + self.noise * np.eye(n)
        for jitter in (0.0,) + JITTERS:
            if jitter:
                K = K + jitter * np.eye(n)
            try:
                factor = cho_factor(K, lower=True, check_finite=False)
            except Exception:
                continue
            break
        else:
            return NLL_FAILURE
        # cho_factor renvoie (lower, c) ; cho_solve s'attend a cette paire.
        Lmat = factor[0]
        alpha = cho_solve(factor, y_n)
        logdet = 2.0 * np.sum(np.log(np.diag(Lmat)))
        nll = 0.5 * y_n @ alpha + 0.5 * logdet + 0.5 * n * np.log(2.0 * np.pi)
        if not np.isfinite(nll):
            return NLL_FAILURE
        self._L = factor
        self._alpha = alpha
        return float(nll)

    def _negative_log_likelihood(self, log_params, y_n):
        """NLL as a function of the log-space parameters (side effect: sets them)."""
        self._set_params(log_params)
        return self._nll(y_n)

    # ------------------------------------------------------------------ #
    # fit / condition
    # ------------------------------------------------------------------ #
    def fit(self, X, y, optimize=True):
        """
        Fit the GP on training data.

        Parameters
        ----------
        X : array_like, shape (n, d)
        y : array_like, shape (n,)
        optimize : bool, optional
            If True, optimize ``log(lengthscales)``, ``log(variance)`` and
            ``log(noise)`` with L-BFGS-B (3 seeded random restarts, best kept).
            If False, keep the current hyperparameters and only condition
            on the data.

        Returns
        -------
        self
        """
        self.x_train = np.atleast_2d(np.asarray(X, dtype=float))
        y = np.asarray(y, dtype=float).ravel()
        self.y_train = y
        if self.normalize_y:
            self._y_mean = float(np.mean(y))
            self._y_std = float(np.std(y))
            y_n = (y - self._y_mean) / self._y_std
        else:
            self._y_mean = 0.0
            self._y_std = 1.0
            y_n = y

        if optimize:
            d = self.kernel.lengthscales.size
            # Bornes en espace log, calquees sur celles de scikit-learn
            # (longueurs d'onde et variance dans [1e-3, 1e3], bruit dans
            # [1e-10, 1e1]) : la meme surface de log-vraisemblance doit conduire
            # au meme optimum.
            lo, hi = np.log(1e-3), np.log(1e3)
            bounds = [(lo, hi)] * d + [(lo, hi)] + [(np.log(1e-10), np.log(1e1))]
            best_nll, best_params = NLL_FAILURE, None
            # Redemarrages au schema de scikit-learn (graine fixee) : en plus du
            # point courant, des tirages log-uniformes 0.9*log(U) (biases vers
            # des valeurs < 1, qui couvrent le bassin de l'optimum reel sur des
            # donnees 2D) et un point initial au bruit 1e-4.
            rng = np.random.RandomState(0)
            starts = [self._param_vector()]
            starts.append(np.concatenate([np.zeros(d), [np.log(1.0), np.log(1e-4)]]))
            for i in range(4):
                if i > 0:
                    starts.append(0.9 * np.log(rng.uniform(size=d + 2)))
                else:
                    starts.append(np.concatenate([np.zeros(d), [np.log(1.0), np.log(1e-4)]]))

            def objective(p):
                self._set_params(p)
                return self._nll(y_n)

            for x0 in starts:
                res = minimize(
                    objective, x0, method="L-BFGS-B", bounds=bounds,
                    options={"maxiter": 2000, "ftol": 1e-14, "gtol": 1e-10},
                )
                f = objective(res.x)
                if f < best_nll:
                    best_nll = f
                    best_params = res.x
            self._set_params(best_params)
            self._nll(y_n)
        else:
            self._nll(y_n)
        return self

    # ------------------------------------------------------------------ #
    # predictions
    # ------------------------------------------------------------------ #
    def _normalized_y(self):
        y = self.y_train
        if self.normalize_y:
            return (y - self._y_mean) / self._y_std
        return y

    def predict(self, X, return_std=False):
        """
        Posterior mean (and std of f, without noise) at ``X``.

        Parameters
        ----------
        X : array_like, shape (m, d)
        return_std : bool, optional
            If True, also return the posterior standard deviation.

        Returns
        -------
        mean : ndarray, shape (m,)
        std : ndarray, shape (m,), optional
            Posterior standard deviation of f (noise excluded).
        """
        X = np.atleast_2d(np.asarray(X, dtype=float))
        m = X.shape[0]
        Ks = self.kernel.K(X, self.x_train)
        # mean = Ks @ alpha (alpha = K_train^{-1} y)
        f_mean_n = Ks @ self._alpha
        # variance de f (sans bruit) : var_i = k_ii - Ks_i @ K_train^{-1} @ Ks_i^T
        v = cho_solve(self._L, Ks.T)  # v = K_train^{-1} Ks^T
        diag_kss = np.full(m, self.kernel.variance)
        cross = np.einsum("ij,ji->i", Ks, v)  # diag(Ks @ K_train^{-1} @ Ks^T)
        f_var_n = np.maximum(diag_kss - cross, 0.0)
        # retour en unite de y
        f_mean = self._y_mean + self._y_std * f_mean_n
        if not return_std:
            return f_mean
        f_std = self._y_std * np.sqrt(f_var_n)
        return f_mean, f_std

    # ------------------------------------------------------------------ #
    # likelihood / leave-one-out
    # ------------------------------------------------------------------ #
    def log_marginal_likelihood(self):
        """
        Log marginal likelihood of the training data at the current hyperparameters.

        Expressed in the normalized ``y_n`` units when ``normalize_y=True``.
        """
        return -self._nll(self._normalized_y())

    def loo(self):
        """
        Leave-one-out predictive moments of ``y_i`` (noise included), closed form.

        Closed form equivalent to Rasmussen & Williams section 5.4.2, with no
        re-fitting: with ``A = K + s I``, ``alpha = A^{-1} y``, ``a = K alpha``
        and ``S_i = 1 / (A^{-1})_ii``,
        ``mu_(i) = y_i - (S_i / s) (y_i - a_i)`` and ``var_(i) = S_i`` (noise
        included), all computed in ``y_n`` units when normalized.

        Returns
        -------
        mean : ndarray, shape (n,)
            LOO predictive mean of y_i, in y units.
        var : ndarray, shape (n,)
            LOO predictive variance of y_i, in y**2 units.
        """
        # Formule fermee verifiee (equivalente a Rasmussen & Williams 5.4.2) :
        # avec A = K + s I, alpha = A^{-1} y, a = K alpha, S_i = 1 / (A^{-1})_ii :
        #   mu_loo_i = y_i - (S_i / s) (y_i - a_i)
        #   var_loo_i = S_i   (variance de y_i incluant le bruit)
        y_n = self._normalized_y()
        n = y_n.size
        A_inv = cho_solve(self._L, np.eye(n))
        S = 1.0 / A_inv.diagonal()
        a = self.kernel.K(self.x_train, self.x_train) @ self._alpha
        mu_loo = y_n - (S / self.noise) * (y_n - a)
        var_loo = S
        # retour en unite de y
        y_mean = self._y_mean if self.normalize_y else 0.0
        y_std = self._y_std if self.normalize_y else 1.0
        return y_mean + y_std * mu_loo, y_std ** 2 * var_loo
