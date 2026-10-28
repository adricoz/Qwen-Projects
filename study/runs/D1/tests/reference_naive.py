"""References naives et independantes de gp_lite (Rasmussen & Williams, ch. 2 & 5).

Chaque fonction est ecrite a partir des formules matricielles elementaires
(``np.linalg.inv``, ``np.linalg.slogdet``, ``np.linalg.solve``, boucles)
SANS reutiliser le code du paquet, pour servir de comparaison dans les tests.
"""
import numpy as np


def naive_kernel_matrix(X1, X2, kind, lengthscales, variance):
    """Matrice (n1, n2) d'un noyau ARD, calculee point par point (boucles).

    Parameters
    ----------
    X1, X2 : ndarray, shape (n1, d) / (n2, d)
    kind : {'se', 'm32', 'm52'}
    lengthscales : ndarray, shape (d,)
    variance : float
    """
    X1 = np.atleast_2d(np.asarray(X1, float))
    X2 = np.atleast_2d(np.asarray(X2, float))
    n1, n2 = X1.shape[0], X2.shape[0]
    out = np.empty((n1, n2))
    for a in range(n1):
        for b in range(n2):
            r2 = np.sum(((X1[a] - X2[b]) / lengthscales) ** 2)
            if kind == "se":
                out[a, b] = variance * np.exp(-0.5 * r2)
            elif kind == "m32":
                r = np.sqrt(r2)
                out[a, b] = variance * (1.0 + np.sqrt(3.0) * r) * np.exp(-np.sqrt(3.0) * r)
            else:  # m52
                r = np.sqrt(r2)
                out[a, b] = variance * (1.0 + np.sqrt(5.0) * r + (5.0 / 3.0) * r2) * np.exp(-np.sqrt(5.0) * r)
    return out


def naive_log_marginal_likelihood(X, y, lengthscales, variance, noise, kind="se"):
    """log N(y | 0, K + noise I), calculee avec inv + slogdet."""
    K = naive_kernel_matrix(X, X, kind, lengthscales, variance)
    A = K + noise * np.eye(len(y))
    sign, logdet = np.linalg.slogdet(A)
    if sign <= 0:
        return -np.inf
    n = len(y)
    return -0.5 * float(y @ np.linalg.solve(A, y)) - 0.5 * logdet - 0.5 * n * np.log(2.0 * np.pi)


def naive_predict(X, X_new, y, lengthscales, variance, noise, kind="se",
                  return_std=False):
    """Prediction (moyenne, ecart-type de f sans bruit) par inv/cholesky."""
    K = naive_kernel_matrix(X, X, kind, lengthscales, variance)
    A = K + noise * np.eye(len(y))
    Ainv = np.linalg.inv(A)
    ks = naive_kernel_matrix(X, X_new, kind, lengthscales, variance)  # (n, m)
    mean = (ks.T @ Ainv @ y).ravel()
    var = variance - np.einsum("ij,ij->j", Ainv @ ks, ks)
    if return_std:
        return mean, np.sqrt(np.maximum(var, 0.0))
    return mean


def naive_loo(X, y, lengthscales, variance, noise, kind="se"):
    """LOO par n re-ajustements explicites (moyenne + variance predictives, bruit inclus).

    Formules : moy_i = k_s^T A_i^{-1} y_i ; var_i = K_ii + noise - k_s^T A_i^{-1} k_s,
    ou, equivalent (RW 5.4.2), var_i = 1 / (A^{-1})_{ii} avec A la matrice complete.
    """
    Kfull = naive_kernel_matrix(X, X, kind, lengthscales, variance)
    A = Kfull + noise * np.eye(len(y))
    beta = np.diag(np.linalg.inv(A))
    n = len(y)
    mean = np.empty(n)
    for i in range(n):
        Xi = np.delete(X, i, 0)
        Ai = naive_kernel_matrix(Xi, Xi, kind, lengthscales, variance) + noise * np.eye(n - 1)
        ks = naive_kernel_matrix(Xi, X[i:i + 1], kind, lengthscales, variance).ravel()
        mean[i] = float(ks @ np.linalg.solve(Ai, np.delete(y, i)))
    var = 1.0 / beta
    return mean, var
