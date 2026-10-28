import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scipy.linalg import cho_solve
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

Xh = np.array([X[i]])
n, mm = Xd.shape[0], 1
K_s = m.kernel.K(m.X_train_, Xh)
print("K_s shape", K_s.shape, "K_s[0]", K_s[0])
Ks2 = K_s.T
A_inv_Ks = cho_solve(m.L_, K_s)
mu = A_inv_Ks.T @ m.alpha_
print("A_inv_Ks[0]:", A_inv_Ks[:,0])
print("mu:", mu, " alpha_[:3]:", m.alpha_[:3])
# direct
A = kern().K(Xd,Xd)+noise*np.eye(n)
Ainv = np.linalg.inv(A)
print("Ainv@k [0]:", (Ainv @ K_s)[0])
print("alpha direct:", np.linalg.solve(A, yd)[:3])
print("diff A_inv_Ks:", np.abs(A_inv_Ks[:,0] - (Ainv@K_s)[0]).max())
print("diff alpha:", np.abs(m.alpha_-np.linalg.solve(A,yd)).max())
