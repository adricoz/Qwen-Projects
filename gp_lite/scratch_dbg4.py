import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scipy.linalg import cho_solve
from gp_lite import SquaredExponentialARD, GPRegressor

noise = 0.01
def kern():
    return SquaredExponentialARD(np.array([0.8, 1.2]), 1.0)

rng = np.random.default_rng(0)
X = rng.normal(size=(12, 2))
y = X @ np.array([1., -1.]) + 0.1

# condition on ALL X, predict X (should interpolate if low noise... check general)
m = GPRegressor(kern(), noise=noise, normalize_y=False)
m.condition(X, y)
mu = m.predict(X)

# direct formula
A = kern().K(X, X) + noise*np.eye(len(y))
alpha = np.linalg.solve(A, y)
Ks = kern().K(X, X).T  # K(X, X)
mu_direct = (Ks.T @ alpha)   # K(X*, X) A^-1 y with X*=X
print("predict :", mu[:4])
print("direct  :", mu_direct[:4])
print("diff    :", np.abs(mu - mu_direct).max())

# predict at a held-out point
Xh = np.array([[0.5, -0.3]])
mu_h = m.predict(Xh)
mu_hd = kern().K(Xh, X) @ alpha
print("heldout predict:", mu_h, "direct:", mu_hd, "diff", abs(mu_h[0]-mu_hd[0]))
