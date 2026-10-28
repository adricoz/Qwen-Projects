import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scipy.linalg import cho_solve
from gp_lite import SquaredExponentialARD, GPRegressor

rng = np.random.default_rng(0)
X = rng.normal(size=(12, 2))
y = X @ np.array([1., -1.]) + 0.1
noise = 0.01

def fresh():
    return GPRegressor(SquaredExponentialARD(np.array([0.8, 1.2]), 1.0),
                       noise=noise, normalize_y=False)

m = fresh()
m.condition(X, y)
n = len(y)
K = m.kernel.K(X, X)
A = K + noise*np.eye(n)
ainv = np.linalg.inv(A)
alpha_ref = np.linalg.solve(A, y)

# ainv via cho_solve
P = cho_solve(m.L_, np.eye(n), check_finite=False)
ainv_ii = np.einsum("ij,ij->i", P, P)
print("cho ainv diag:", ainv_ii)
print("ref ainv diag:", np.diag(ainv))
print("alpha match:", np.abs(m.alpha_ - alpha_ref).max())

loo_model = m.loo()
r_ref = alpha_ref / np.diag(ainv)
print("model loo :", loo_model[:4])
print("ref loo   :", r_ref[:4])

# explicit refit
ref = np.array([
    y[i] - fresh().condition(np.delete(X,i,0), np.delete(y,i)).predict(X[i:i+1])[0]
    for i in range(len(y))
])
print("refit loo :", ref[:4])
print("err refit vs r_ref:", np.abs(ref - r_ref).max())
