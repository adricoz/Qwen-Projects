import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from gp_lite import SquaredExponentialARD

noise = 0.01

def Kfn(X1, X2):
    return SquaredExponentialARD(np.array([0.8, 1.2]), 1.0).K(X1, X2)

rng = np.random.default_rng(0)
X = rng.normal(size=(12, 2))
y = X @ np.array([1., -1.]) + 0.1
n = len(y)

K = Kfn(X, X)
A = K + noise*np.eye(n)
alpha = np.linalg.solve(A, y)
ainv = np.linalg.inv(A)

for i in range(3):
    Xd = np.delete(X, i, 0)
    yd = np.delete(y, i)
    B = Kfn(Xd, Xd) + noise*np.eye(n-1)
    kvec = Kfn(np.array([X[i]]), Xd)[0]
    pred = kvec @ np.linalg.solve(B, yd)
    r_refit = y[i] - pred
    s = A[i,i] - kvec @ np.linalg.solve(B - kvec[...,None]*0, kvec)  # k^T B^-1 k via solve
    s = A[i,i]
    kBT = np.linalg.solve(B, kvec)
    s = K[i,i] + noise - kvec @ kBT
    r_s = s * alpha[i]
    print(i, "refit:", r_refit, " s*alpha:", r_s, " alpha/(Ainv)_ii:", alpha[i]/ainv[i,i])
