import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from gp_lite import SquaredExponentialARD, GPRegressor

noise = 0.01
def kern():
    return SquaredExponentialARD(np.array([0.8, 1.2]), 1.0)

rng = np.random.default_rng(1)
X = rng.normal(size=(15, 2))
y = X @ np.array([1., -1.]) + 0.2
Xt = rng.normal(size=(7, 2))
n = len(y)

m = GPRegressor(kern(), noise=noise, normalize_y=False)
m.condition(X, y)
K = kern().K(X, X)
A = K + noise*np.eye(n)

mu_ours = m.predict(Xt)
mu_direct = kern().K(Xt, X) @ np.linalg.solve(A, y)   # K(X*,X) A^-1 y
print("predict vs direct:", np.abs(mu_ours - mu_direct).max())

# what does the code actually compute?
K_s = kern().K(X, Xt)   # n x m
from scipy.linalg import cho_solve
alpha_star = cho_solve(m.L_, K_s)
mu_code = alpha_star.T @ m.alpha_
print("code-formula:", np.abs(mu_code - mu_ours).max())
