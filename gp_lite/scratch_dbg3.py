import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scipy.linalg import cho_solve, cho_factor
from gp_lite import SquaredExponentialARD, GPRegressor

noise = 0.01
def kern():
    return SquaredExponentialARD(np.array([0.8, 1.2]), 1.0)

rng = np.random.default_rng(0)
X = rng.normal(size=(12, 2))
y = X @ np.array([1., -1.]) + 0.1
n = len(y)

m = GPRegressor(kern(), noise=noise, normalize_y=False)
m.condition(X, y)
K = kern().K(X, X)
A = K + noise*np.eye(n)
ainv_ref = np.linalg.inv(A)
alpha_ref = np.linalg.solve(A, y)

print("alpha match:", np.abs(m.alpha_ - alpha_ref).max())
P = cho_solve(m.L_, np.eye(n), check_finite=False)
print("P (A^-1) diag :", np.diag(P)[:4])
print("ainv_ref diag :", np.diag(ainv_ref)[:4])
print("P vs ainvref  :", np.abs(P - ainv_ref).max())

# manual cho factor of A to compare with m.L_
c_ref, lo_ref = cho_factor(A, lower=True, check_finite=False)
print("L_ match ref cho:", np.abs(np.tril(c_ref) - np.tril(m.L_[0])).max())
print("lower flags:", lo_ref, m.L_[1] if len(m.L_)>1 else m.L_[1])

loo = m.loo()
print("loo:", loo[:4])
print("alpha/ainv:", (alpha_ref/np.diag(ainv_ref))[:4])
