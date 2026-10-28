"""Regresseur par processus gaussien : ajustement, prediction, LOO."""
import numpy as np
from scipy.optimize import minimize

from .kernels import _ARDKernel


# Jitters relatifs tries quand la factorisation de Cholesky echoue
_JITTERS = (0.0, 1e-10, 1e-8, 1e-6)
# Valeur retournee par la NLL quand la matrice de covariance n'est pas factorisable
_NLL_FAILURE = 1e10
# Bornes des hyperparametres en espace log (l et variance dans [e^-6, e^6],
# noise dans [e^-10, e^5]) -- coherents avec une sortie normalisee.
_LL_SCALE = (-6.0, 6.0)   # log(lengthscales), log(variance)
_LL_NOISE = (-18.0, 5.0)  # log(noise) : borne basse ~1.5e-8 pour interpoleer les donnees propres
_N_RESTARTS = 3
_RNG_SEED = 0


class GPRegressor:
    """Processus gaussien a moyenne nulle avec bruit blanc additif.

    L'ajustement maximise la vraisemblance marginale (log-space, L-BFGS-B)
    des hyperparametres ``[lengthscales, variance, noise]`` avec redemarrages
    aleatoires. La sortie est normalisee par defaut (moyenne nulle, ecart-type 1)
    avant de former la vraisemblance ; les predictions sont renvoyees dans les
    unites de ``y``.
    """

    def __init__(self, kernel, noise=1e-6, normalize_y=True):
        """
        Parameters
        ----------
        kernel : object
            Noyau ARD de ce paquet (``lengthscales`` en (d,), ``variance``).
        noise : float, default 1e-6
            Variance du bruit blanc (sur y normalise si ``normalize_y``).
        normalize_y : bool, default True
            Si vrai, travaille sur y_n = (y - mean(y)) / std(y) (std, ddof = 0).
        """
        if not isinstance(kernel, _ARDKernel):
            raise TypeError("kernel must be a gp_lite kernel (ARD)")
        if noise < 0 or not np.isfinite(noise):
            raise ValueError(f"noise must be a finite non-negative number, got {noise}")
        self.kernel = kernel
        self.noise = float(noise)
        self.normalize_y = bool(normalize_y)
        # Etat appris (remplis par fit)
        self.X_train = None
        self.y_train = None
        self.y_mean = 0.0
        self.y_std = 1.0
        self._y_n = None
        self._L = None
        self._alpha = None
        self._b = None
        self._jitter = 0.0

    # ------------------------------------------------------------------ #
    def fit(self, X, y, optimize=True):
        """Ajuste le GP (optionnellement sur la vraisemblance) et conditionne.

        Parameters
        ----------
        X : array-like, shape (n, d)
            Points d'apprentissage.
        y : array-like, shape (n,)
            Observations.
        optimize : bool, default True
            Si vrai, optimise ``log(lengthscales), log(variance), log(noise)``
            par L-BFGS-B (3 redemarrages aleatoires, graine fixee, meilleur garde).
            Si faux, conserve les hyperparametres actuels et ne fait que conditionner.

        Returns
        -------
        self
        """
        X = np.atleast_2d(np.asarray(X, dtype=float))
        y = np.asarray(y, dtype=float).ravel()
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"X has {X.shape[0]} rows but y has {y.shape[0]} values")
        if X.shape[1] != self.kernel.n_dims:
            raise ValueError(
                f"X has {X.shape[1]} dims but the kernel expects {self.kernel.n_dims}"
            )
        self.X_train = X
        self.y_train = y
        # Normalisation (ddof = 0) ; si std = 0 on ne divise pas
        self.y_mean = float(np.mean(y)) if self.normalize_y else 0.0
        self.y_std = float(np.std(y)) if self.normalize_y else 1.0
        if self.normalize_y and self.y_std > 0.0:
            self._y_n = (y - self.y_mean) / self.y_std
        else:
            self._y_n = y
        if optimize:
            self._optimize(self._y_n)
        self._condition()
        return self

    # ------------------------------------------------------------------ #
    def predict(self, X, return_std=False):
        """Prediction a posteriori moyenne (et ecart-type de f, sans bruit).

        Parameters
        ----------
        X : array-like, shape (m, d)
            Points de prediction.
        return_std : bool, default False
            Si vrai, renvoie aussi l'ecart-type (m,).

        Returns
        -------
        mean : ndarray, shape (m,)
            Moyenne predite dans les unites de ``y``.
        std : ndarray, shape (m,), optionnel
            Ecart-type predit de ``f`` (sans le bruit), unites de ``y``.
        """
        if self._L is None:
            raise RuntimeError("call fit() before predict()")
        X = self.kernel._check_input(X)
        k_s = self.kernel.K(X, self.X_train)
        mean_n = k_s @ self._alpha
        # Var(f_* | y) = K(X, X) - k_s^T L^{-1} L^{-T} k_s ; on ne garde que la diagonale
        L_solve = np.linalg.solve(self._L, k_s.T)      # (n, m)
        diag = self.kernel.diag(X)
        var = diag - np.sum(L_solve ** 2, axis=0)
        var = np.maximum(var, 0.0)
        mean = self.y_mean + self.y_std * mean_n
        std = self.y_std * np.sqrt(var)
        if return_std:
            return mean, std
        return mean

    # ------------------------------------------------------------------ #
    def log_marginal_likelihood(self):
        """Log-vraissemblance marginale aux hyperparametres courants (sur y_n).

        Returns
        -------
        float
            ``-0.5 y^T A^{-1} y - sum log diag(L) - 0.5 n log(2 pi)``.
        """
        if self._L is None:
            raise RuntimeError("call fit() before log_marginal_likelihood()")
        n = self.X_train.shape[0]
        ll = (
            -0.5 * float(self._y_n @ self._alpha)
            - float(np.sum(np.log(np.diag(self._L))))
            - 0.5 * n * np.log(2.0 * np.pi)
        )
        return float(ll)

    # ------------------------------------------------------------------ #
    def loo(self):
        """Residus leave-one-out en forme fermee (Rasmussen & Williams 5.4.2).

        Formules utilisees (``A = K + noise I``, ``alpha = A^{-1} y_n``,
        ``b_i = (A^{-1})_{ii}``) :

        - moyenne : ``y_n_i - alpha_i / b_i``
        - variance : ``1 / b_i``

        sans refaire n ajustements. Renvoye dans les unites de ``y``.

        Returns
        -------
        mean : ndarray, shape (n,)
        var : ndarray, shape (n,)
            Moyenne et variance predictives (bruit inclus) de chaque y_i.
        """
        if self._b is None:
            raise RuntimeError("call fit() before loo()")
        y_n = self._y_n
        mean_n = y_n - self._alpha / self._b
        var_n = 1.0 / self._b
        return self.y_mean + self.y_std * mean_n, self.y_std ** 2 * var_n

    # ------------------------------------------------------------------ #
    def _optimize(self, y_n):
        """Maximise la vraissemblance sur log(l_1..l_d, log(var), log(noise))."""
        d = self.kernel.n_dims
        rng = np.random.default_rng(_RNG_SEED)
        # Points de depart : l'etat courant + _N_RESTARTS redemarrages aleatoires
        starts = [self._theta_from_state()]
        lo, hi = self._bounds()
        for _ in range(_N_RESTARTS):
            starts.append(lo + (hi - lo) * rng.random(d + 2))
        best_theta, best_nll = None, np.inf
        for x0 in starts:
            res = minimize(
                self._negative_ll,
                np.clip(x0, lo, hi),
                method="L-BFGS-B",
                bounds=list(zip(lo, hi)),
                options={"maxiter": 500, "maxfun": 5000},
            )
            nll = float(res.fun)
            if nll < best_nll:
                best_nll, best_theta = nll, res.x
        self._set_theta(best_theta)

    def _bounds(self):
        """Bornes (lo, hi) de dimension d + 2, dans l'espace log."""
        d = self.kernel.n_dims
        lo = np.concatenate([np.full(d, _LL_SCALE[0]), [_LL_SCALE[0], _LL_NOISE[0]]])
        hi = np.concatenate([np.full(d, _LL_SCALE[1]), [_LL_SCALE[1], _LL_NOISE[1]]])
        return lo, hi

    def _theta_from_state(self):
        d = self.kernel.n_dims
        return np.concatenate([
            np.log(self.kernel.lengthscales),
            [np.log(max(self.kernel.variance, 1e-12))],
            [np.log(max(self.noise, 1e-12))],
        ])

    def _set_theta(self, theta):
        d = self.kernel.n_dims
        self.kernel.lengthscales = np.exp(theta[:d]).copy()
        self.kernel.variance = float(np.exp(theta[d]))
        self.noise = float(np.exp(theta[d + 1]))

    def _negative_ll(self, theta, y_n=None):
        """NLL aux hyperparametres (espace log), avec jitter si besoin."""
        d = self.kernel.n_dims
        l = np.exp(theta[:d])
        var = float(np.exp(theta[d]))
        noise = float(np.exp(theta[d + 1]))
        self.kernel.lengthscales = l
        self.kernel.variance = var
        self.noise = noise
        if y_n is None:
            y_n = self._y_n
        A = self.kernel.K(self.X_train, self.X_train) + noise * np.eye(len(y_n))
        L, ok = _safe_cholesky(A)
        if not ok:
            return _NLL_FAILURE
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_n))
        nll = (
            0.5 * float(y_n @ alpha)
            + float(np.sum(np.log(np.diag(L))))
            + 0.5 * len(y_n) * np.log(2.0 * np.pi)
        )
        return nll

    def _condition(self):
        """Factorise la covariance aux hyperparametres courants et stocke alpha, b."""
        X, y_n = self.X_train, self._y_n
        A = self.kernel.K(X, X) + self.noise * np.eye(len(y_n))
        L, ok = _safe_cholesky(A)
        if not ok:
            raise RuntimeError("could not factorize the covariance matrix even with jitter")
        self._L = L
        self._alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_n))
        # b_i = (A^{-1})_{ii} = somme des carres de la COLONNE i de L^{-1} (A = L L^T)
        L_inv = np.linalg.solve(L, np.eye(len(y_n)))
        self._b = np.sum(L_inv ** 2, axis=0)


def _safe_cholesky(A):
    """Cholesky avec escalade de jitter relatif a l'echelle de la diagonale.

    Returns
    -------
    (L, ok) : L facteur inferieur si ok, sinon (None, False).
    """
    scale = max(1.0, float(np.max(np.abs(np.diag(A)))))
    for j in _JITTERS:
        if j > 0:
            A = A + j * scale * np.eye(len(A))
        try:
            L = np.linalg.cholesky(A)
            return L, True
        except np.linalg.LinAlgError:
            continue
    return None, False
