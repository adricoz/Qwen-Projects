import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from gp_lite import SquaredExponentialARD

rng = np.random.default_rng(0)
X = rng.normal(size=(12, 2))
y = X @ np.array([1., -1.]) + 0.1
noise = 0.01

def fresh():
    return SquaredExponentialARD(np.array([0.8, 1.2]), 1.0)

n = len(y)
K = fresh().K(X, X)
A = K + noise*np.eye(n)
alpha = np.linalg.solve(A, y)
ainv = np.linalg.inv(A)
a_ii = np.diag(A)

ref = np.array([
    y[i] - (fresh().K(np.array([X[i]]), np.delete(X,i,0))[0] @ np.linalg.solve(fresh().K(np.delete(X,i,0), np.delete(X,i,0)) + noise*np.eye(n-1), np.delete(y,i)))
    for i in range(n)
])

cand = ainv.diagonal() * (y - a_ii * alpha)
print("err:", np.abs(cand - ref).max())
print("ref:", ref[:4])
print("cand:", cand[:4])

# zero-noise limit
K2 = fresh().K(X, X)
A2 = K2
alpha2 = np.linalg.solve(A2, y)
ainv2 = np.linalg.inv(A2)
cand2 = ainv2.diagonal() * (y - np.diag(A2) * alpha2)
print("zero-noise resid (should be ~0):", np.abs(cand2).max())
