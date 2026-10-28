import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from gp_lite import SquaredExponentialARD, GPRegressor

noise = 0.01
def fresh():
    return GPRegressor(SquaredExponentialARD(np.array([0.8, 1.2]), 1.0),
                       noise=noise, normalize_y=False)

rng = np.random.default_rng(0)
X = rng.normal(size=(12, 2))
y = X @ np.array([1., -1.]) + 0.1

model = fresh()
model.condition(X, y)
loo = model.loo()

ref = []
for i in range(len(y)):
    m2 = fresh()
    m2.condition(np.delete(X,i,0), np.delete(y,i))
    ref.append(y[i]-m2.predict(X[i:i+1])[0])
ref = np.array(ref)

print("loo  :", loo)
print("ref  :", ref)
print("max  :", np.abs(loo-ref).max())
print("model alpha_ :", model.alpha_[:3])
print("model noise   :", model.noise)
print("model lens    :", model.kernel.lengthscales, model.kernel.variance)
m2 = fresh(); m2.condition(X,y)
print("fresh alpha   :", m2.alpha_[:3], m2.noise, m2.kernel.lengthscales)
