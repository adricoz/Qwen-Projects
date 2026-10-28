import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from gp_lite import GPRegressor, SquaredExponentialARD
from gp_lite.lhs import lhs

def branin2d(x):
    x1, x2 = x[..., 0], x[..., 1]
    return 1.0*(x2 - 5.1/(4*np.pi**2)*x1**2 + 5/np.pi*x1 - 6)**2 + 10*(1-1/(8*np.pi))*np.cos(x1) + 10

noise = 1e-3
ls, sv = [0.5, 0.3], 1.0
X = lhs(12, 2, seed=0); y = branin2d(X)
k = SquaredExponentialARD(lengthscales=ls, variance=sv)
n = len(y)
K0 = k.K(X, X); Kfull = K0 + noise*np.eye(n)
H = np.linalg.inv(Kfull)

ref_mu, ref_var = [], []
for i in range(n):
    gi = GPRegressor(SquaredExponentialARD(lengthscales=ls, variance=sv), noise=noise, normalize_y=False).fit(np.delete(X,i,0), np.delete(y,i), optimize=False)
    m, sd = gi.predict(X[i:i+1], return_std=True)
    ref_mu.append(m[0]); ref_var.append(sd[0]**2 + noise)
ref_var = np.array(ref_var)

# lin-algebra ground truth
gt_var, gt_mu = [], []
for i in range(n):
    idx = np.arange(n) != i
    Ktr = K0[idx][:, idx] + noise*np.eye(n-1)
    ki = K0[idx, i]
    yi = np.delete(y, i)
    gt_mu.append(ki @ np.linalg.solve(Ktr, yi))
    gt_var.append(1.0 - ki @ np.linalg.solve(Ktr, ki) + noise)
gt_var = np.array(gt_var)

print("1/H_ii   :", (1.0/np.diag(H))[:5])
print("gt linalg:", gt_var[:5])
print("package  :", ref_var[:5])
print("package vs gt mu err:", np.max(np.abs(np.array(ref_mu)-np.array(gt_mu))))
print("package vs gt var err:", np.max(np.abs(ref_var-gt_var)))
print("loo vs gt var err:", np.max(np.abs(1.0/np.diag(H)-gt_var)))
