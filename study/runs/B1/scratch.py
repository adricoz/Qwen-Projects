import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import minimize

def branin_design(n, seed):
    rng = np.random.default_rng(seed)
    u = np.stack([rng.permutation(n) / n for _ in range(2)], axis=1)
    u = (u + rng.random((n, 2))) / n
    return np.column_stack([u[:, 0] * 15 - 5, u[:, 1] * 15])

def branin(X):
    x1, x2 = X[:, 0], X[:, 1]
    return ((x2 - 5.1 / (4 * np.pi**2) * x1**2 + 5 / np.pi * x1 - 6)**2
            + 10 * (1 - 1 / (8 * np.pi)) * np.cos(x1) + 10)

X = branin_design(30, 0)
y = branin(X)
m, s = y.mean(), y.std()
yn = (y - m) / s

def lml(ls, var, noise):
    r = cdist(X / ls, X / ls)
    K = var * np.exp(-0.5 * r**2) + noise * np.eye(len(yn))
    L = np.linalg.cholesky(K)
    a = np.linalg.solve(L.T, np.linalg.solve(L, yn))
    return -0.5 * (yn @ a + 2 * np.log(np.diag(L)).sum()) - 15 * np.log(2 * np.pi)

def nll(t):
    v = lml(np.exp(t[:2]), np.exp(t[2]), np.exp(t[3]))
    return -v

lo, hi = -7.0, 7.0
rng = np.random.default_rng(0)
starts = [np.log([1.0, 1.0, 1.0, 1e-6])]
for _ in range(9):
    starts.append(np.clip(lo + (hi - lo) * rng.random(4), lo, hi))
for i, t0 in enumerate(starts):
    res = minimize(nll, t0, method="L-BFGS-B", bounds=[(lo, hi)] * 4,
                   options={"maxiter": 600, "ftol": 1e-14, "gtol": 1e-8})
    print(i, "start", np.round(t0, 2), "-> LML", -res.fun,
          " ls", np.round(np.exp(res.x[:2]), 3), "var", np.round(np.exp(res.x[2]), 3),
          "noise", np.exp(res.x[3]))
