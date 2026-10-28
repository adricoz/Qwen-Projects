ll) else 1e10

    # ------------------------------------------------------------------
    def fit(self, X, y, optimize=True):
        """Estimate the hyperparameters by maximizing the marginal likelihood.

        The log of [lengthscales, variance, noise] is optimized with L-BFGS-B
        (3 random restarts, fixed seed), keeping the best result.

        Parameters
        ----------
        X : array_like
            Training points, shape ``(n, d)``.
        y : array_like
            Training targets, shape ``(n,)``.
        optimize : bool
            If False, only condition on (X, y) with the current hyperparameters
            (no optimization) -- useful for testing against a reference.

        Returns
        -------
        self
        """
        X = np.atleast_2d(np.asarray(X, dtype=float))
        y = np.asarray(y, dtype=float).ravel()
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of rows")
        self.X_train, self.y_train = X, y

        if self.normalize_y:
            self.y_mean = float(y.mean())
            self.y_std = float(max(y.std(), 1e-12))
        else:
            self.y_mean, self.y_std = 0.0, 1.0
        y_n = (y - self.y_mean) / self.y_std
        self._y_train_n = y_n

        if not optimize:
            self._condition(X, y_n)
            return self

        d = X.shape[1]
        if self.kernel.lengthscales.size == 1 and d > 1:
            self.kernel.lengthscales = np.full(d, self.kernel.lengthscales[0])
        n_len = self.kernel.lengthscales.size

        rng = np.random.default_rng(42)
        log_lo = np.array([np.log(0.01)] * n_len + [np.log(1e-3), np.log(1e-10)])
        log_hi = np.array([np.log(10.0)] * n_len + [np.log(50.0), np.log(1.0)])
        best, best_x = np.inf, None
        for _ in range(3):
            x0 = rng.uniform(log_lo, log_hi)
            res = minimize(self._nll_only, x0, method="L-BFGS-B",
                           bounds=list(zip(log_lo, log_hi)),
                           options=dict(maxiter=300))
            if res.fun < best:
                best, best_x = res.fun, res.x
        self._set_from_log(best_x)
        self._nll_best = best
        self._condition(X, y_n)
        return self

    def refit(self, X, y):
        """Re-condition on (X, y) WITHOUT re-optimizing (same hyperparameters).

        Parameters
        ----------
        X : array_like
            Training points, shape ``(n, d)``.
        y : array_like
            Training targets, shape ``(n,)``.

        Returns
        -------
        self
        """
        return self.fit(X, y, optimize=False)

    def log_marginal_likelihood(self):
        """Return the log marginal likelihood of the training data.

        Returns
        -------
        float
            Value at the fitted hyperparameters.
        """
        if not self._fitted:
            raise RuntimeError("call fit() first")
        return -self._nll_best

    def predict(self, X, return_std=False):
        """Posterior mean and, optionally, standard deviation at ``X``.

        Parameters
        ----------
        X : array_like
            Points of shape ``(m, d)``.
        return_std : bool
            If True, also return the posterior standard deviation.

        Returns
        -------
        numpy.ndarray or (numpy.ndarray, numpy.ndarray)
            Posterior mean (raw units), plus std if requested.
        """
        if not self._fitted:
            raise RuntimeError("call fit() first")
        X = np.atleast_2d(np.asarray(X, dtype=float))
        f = self.kernel.K(X, self.X_train) @ self._alpha
        mean = self.y_mean + self.y_std * f
        if not return_std:
            return mean
        k = self.kernel.K(X, X).T
        v = self.kernel.diag(X) - np.einsum("ij,ij->i", k, f)
        std = self.y_std * np.sqrt(np.maximum(v, 0.0))
        return mean, std

    def loo(self):
        """Leave-one-out residuals in closed form (Rasmussen & Williams, 5.4.2).

        ``mu_-i = mu_i - k_i . alpha / (1 + k_i^T (K + sigma^2 I)^-1 k_i)``

        Returns
        -------
        numpy.ndarray
            Residuals ``y_i - mu_-i`` in raw units, shape ``(n,)``.
        """
        if not self._fitted:
            raise RuntimeError("call fit() first")
        n = self.X_train.shape[0]
        K = self.kernel.K(self.X_train, self.X_train)
        Xmat = cho_solve((self._L, True), K)
        v = np.einsum("ij,ij->i", K, Xmat)
        mu_n = K @ self._alpha
        residual_n = (self._y_train_n
                      - (mu_n - np.diag(K) * self._alpha / (1.0 + v)))
        return self.y_std * residual_n
