"""Régression par processus gaussien (GP) sur données bruitées."""
import numpy as np
from scipy.linalg import cho_factor, cho_solve

_JITTER_STEPS = (0.0, 1e-8, 1e-6, 1e-4)
_LOG_BOUNDS = (-6.0, 4.0)  # bornes pour log(lengthscales), log(variance), log(noise)


class GPRegressor:
    """Régression gaussienne avec bruit additif et conditionnement exact.

    Les hyperparamètres optimisés sont ``kernel.lengthscales``,
    ``kernel.variance`` et ``self.noise``. Si ``normalize_y=True``, les
    données sont normalisées (moyenne/écart-type, ddof=0) avant la
    vraisemblance ; les prédictions sont renvoyées dans les unités de y.

    Parameters
    ----------
    kernel : gp_lite.kernels._ARDKernel
        Noyau possédant les attributs ``lengthscales`` et ``variance``.
    noise : float, optional
        Variance du bruit additif (défaut 1e-6).
    normalize_y : bool, optional
        Si True, y est normalisé avant conditionnement (défaut True).
    """

    def __init__(self, kernel, noise=1e-6, normalize_y=True):
        self.kernel = kernel
        self.noise = float(noise)
        self.normalize_y = bool(normalize_y)
        self._X = None
        self._y_n = None  # y normalisé
        self._y_mean = 0.0
        self._y_std = 1.0
        self._L = None  # facteur de Cholesky de K + noise*I (espace y_n)
        self._alpha = None
        self._lml = None

    # ------------------------------------------------------------------
    # log-vraisemblance marginale (espace normalisé)
    # ------------------------------------------------------------------
    def _log_marginal_likelihood_raw(self, jitter=0.0):
        """Log-vraisemblance marginale par Cholesky (y normalisé)."""
        n = self._y_n.shape[0]
        K = self.kernel.K(self._X, self._X)
        K = K + (self.noise + jitter) * np.eye(n)
        y = self._y_n
        last_err = None
        for j in (jitter, *[_ for _ in _JITTER_STEPS if _ > jitter]):
            try:
                L, low = cho_factor(K, lower=True, check_finite=False)
                break
            except np.linalg.LinAlgError as err:  # Cholesky échouée: on augmente le jitter
                last_err = err
        else:
            raise np.linalg.LinAlgError("Cholesky failed even with jitter") from last_err
        logdet = np.sum(np.log(np.diag(L)))  # = 0.5 * log det(K)
        alpha = cho_solve((L, low), y, check_finite=False)
        lml = (
            -0.5 * float(y @ alpha)
            - logdet
            - 0.5 * n * np.log(2.0 * np.pi)
        )
        return float(lml), L, low, alpha

    def log_marginal_likelihood(self):
        """Log-vraisemblance marginale aux hyperparamètres courants.

        Returns
        -------
        float
            Valeur de la log-vraisemblance marginale (espace normalisé si
            ``normalize_y=True``).
        """
        if self._lml is None:
            lml, self._L, self._low, self._alpha = self._log_marginal_likelihood_raw()
            self._lml = lml
        return self._lml

    # ------------------------------------------------------------------
    # optimisation
    # ------------------------------------------------------------------
    def _set_params_from_theta(self, theta):
        """Copie les hyperparamètres (en log) de theta vers kernel/noise."""
        d = self.kernel.lengthscales.size
        self.kernel.lengthscales = np.exp(theta[:d])
        self.kernel.variance = float(np.exp(theta[d]))
        self.noise = float(np.exp(theta[d + 1]))

    def _negative_lml(self, theta):
        """Négative de la log-vraisemblance (fonction à minimiser)."""
        self._set_params_from_theta(theta)
        lml, _, _, _ = self._log_marginal_likelihood_raw()
        return -lml

    def _optimize(self):
        """Optimise log(l), log(variance), log(noise) par L-BFGS-B.

        Trois redémarrages aléatoires (graine fixée); le meilleur point est
        conservé.
        """
        from scipy.optimize import minimize  # import local pour garder les imports légers

        d = self.kernel.lengthscales.size
        rng = np.random.default_rng(0)  # graine fixée pour la reproductibilité
        lower = np.full(d + 2, _LOG_BOUNDS[0])
        upper = np.full(d + 2, _LOG_BOUNDS[1])

        # premier point de départ: hyperparamètres actuels
        starts = [
            np.concatenate([
                np.log(np.maximum(self.kernel.lengthscales, 1e-12)),
                np.array([
                    np.log(max(self.kernel.variance, 1e-12)),
                    np.log(max(self.noise, 1e-12)),
                ]),
            ])
        ]
        # deux redémarrages aléatoires bornés
        for _ in range(2):
            starts.append(
                rng.uniform(lower + 1.0, upper - 1.0, size=d + 2)
            )

        best_theta, best_val, best_state = None, np.inf, None
        for x0 in starts:
            res = minimize(
                self._negative_lml, x0, method="L-BFGS-B",
                bounds=list(zip(lower, upper)),
                options={"maxiter": 200, "ftol": 1e-10, "gtol": 1e-6},
            )
            if res.fun < best_val:
                best_val, best_theta, best_state = res.fun, res.x, res

        self._set_params_from_theta(best_theta)
        self._opt_result = best_state

    # ------------------------------------------------------------------
    # API publique
    # ------------------------------------------------------------------
    def fit(self, X, y, optimize=True):
        """Ajuste le GP sur (X, y).

        Parameters
        ----------
        X : array-like, shape (n, d)
            Points d'apprentissage.
        y : array-like, shape (n,)
            Observations (bruitées).
        optimize : bool, optional
            Si True, optimise les hyperparamètres par log-vraisemblance
            marginale; sinon les hyperparamètres actuels sont conservés.

        Returns
        -------
        self
        """
        self._X = np.atleast_2d(np.asarray(X, dtype=float))
        y = np.asarray(y, dtype=float).ravel()
        if self.normalize_y:
            self._y_mean = float(np.mean(y))
            self._y_std = float(np.std(y))
            if self._y_std == 0.0:  # dégeneration: pas de normalisation possible
                self._y_std = 1.0
        else:
            self._y_mean = 0.0
            self._y_std = 1.0
        self._y_n = (y - self._y_mean) / self._y_std

        if optimize:
            self._optimize()
        # conditionnement final (Cholesky + alpha), hyperparamètres définitifs
        self._lml = None
        self.log_marginal_likelihood()
        return self

    def predict(self, X, return_std=False):
        """Prédiction a posteriori moyenne et écart-type de f (sans bruit).

        Parameters
        ----------
        X : array-like, shape (m, d)
            Points de test.
        return_std : bool, optional
            Si True, renvoie aussi l'écart-type a posteriori de f.

        Returns
        -------
        mean : numpy.ndarray, shape (m,)
            Moyenne prédictive, dans les unités de y.
        std : numpy.ndarray, shape (m,), optional
            Écart-type de f (sans bruit), dans les unités de y.
        """
        if self._alpha is None:
            raise RuntimeError("Appelez d'abord fit(X, y).")
        X = np.atleast_2d(np.asarray(X, dtype=float))
        Ks = self.kernel.K(X, self._X)  # (m, n)
        mean_n = Ks @ self._alpha
        mean = mean_n * self._y_std + self._y_mean
        if not return_std:
            return mean
        v = cho_solve((self._L, self._low), Ks.T, check_finite=False)  # (n, m)
        var_n = self.kernel.diag(X) - np.sum(Ks.T * v, axis=0)  # (m,)
        std = np.sqrt(np.clip(var_n, 0.0, None)) * self._y_std
        return mean, std

    def loo(self):
        """Résidus leave-one-out en forme fermée (Rasmussen & Williams 5.4.2).

        Sans refaire n ajustements: on utilise A = (K + noise*I)^{-1} et
        alpha = A y_n, avec ``a_ii = A[i, i]``.

        Returns
        -------
        mean : numpy.ndarray, shape (n,)
            Moyenne prédictive leave-one-out de y_i (bruit inclus), dans les
            unités de y.
        var : numpy.ndarray, shape (n,)
            Variance prédictive leave-one-out de y_i (bruit inclus), dans les
            unités de y.
        """
        if self._alpha is None:
            raise RuntimeError("Appelez d'abord fit(X, y).")
        n = self._y_n.shape[0]
        A = cho_solve((self._L, self._low), np.eye(n), check_finite=False)
        a_ii = np.diag(A)
        # MOG (5.4.2), en espace normalisé, bruit inclus:
        #   mean_n(i) = y_n[i] - alpha[i] / A[i, i]
        #   var_n(i)  = 1 / A[i, i]   (complement de Schur, bruit compris)
        mean_n = self._y_n - self._alpha / a_ii
        var_n = 1.0 / a_ii
        return (
            mean_n * self._y_std + self._y_mean,
            var_n * self._y_std ** 2,
        )
