import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from gp_lite import SquaredExponentialARD, GPRegressor

noise = 0.01
def kern():
    return SquaredExponentialARD(np.array([0.8, 1.2]), 1.0)
def fresh():
    return GPRegressor(kern(), noise=noise, normalize_y=False)

rng = np.random.default_rng(0)
X = rng.normal(size=(12, 2))
y = X @ np.array([1., -1.]) + 0.1

i = 0
Xd = np.delete(X, i, 0); yd = np.delete(y, i)
m = fresh()
m.condition(Xd, yd)
pred = m.predict(X[i:i+1])[0]

A = kern().K(Xd, Xd) + noise*np.eye(len(Xd))
kxv = kern().K(np.array([X[i]]), Xd)[0]
pred_d = float(kxv @ np.linalg.solve(A, yd))
print("pred via GP :", pred)
print("pred direct :", pred_d)
print("y_i         :", y[i])
print("loo_i (y_i - pred) :", y[i]-pred_d)
print("m.noise:", m.noise, "lens:", m.kernel.lengthscales, "var:", m.kernel.variance)
print("m alpha_ len:", len(m.alpha_), "Xd rows:", Xd.shape[0])
print("A diag0:", A[0,0], "knn diag0:", kern().K(Xd,Xd)[0,0])
