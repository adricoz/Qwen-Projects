# gp_lite — petit paquet de régression par processus gaussien

`gp_lite` est un paquet autonome (NumPy + SciPy uniquement, **pas de scikit-learn
dans le paquet**) de régression par processus gaussien (GP) à noyau ARD, pensé
pour des surrogates aéro/hydro.

## Installation

Aucune installation nécessaire : le paquet n'utilise que NumPy, SciPy et
matplotlib (l'exemple). scikit-learn n'est requis que par une partie des tests.

```
gp_lite/
├── __init__.py
├── kernels.py     # SquaredExponentialARD, Matern32ARD, Matern52ARD
└── gp.py          # GPRegressor (fit, predict, log_marginal_likelihood, loo)
tests/test_gp_lite.py
examples/branin_demo.py
```

## API

### Noyaux

```python
from gp_lite import SquaredExponentialARD, Matern32ARD, Matern52ARD

k = Matern52ARD(lengthscales=[0.5, 0.7], variance=1.0)
K = k.K(X1, X2)     # (n1, n2)
d = k.diag(X)       # (n,)
```

Convention d'ARD : `r = sqrt(sum_j ((x_j - x'_j) / l_j)^2)`.

| Noyau | Formule |
|---|---|
| `SquaredExponentialARD` | σ² exp(−r²/2) |
| `Matern32ARD` | σ² (1 + ρ) e^{−ρ}, ρ = √3 r |
| `Matern52ARD` | σ² (1 + ρ + 5ρ²/3) e^{−ρ}, ρ = √5 r |

Attributs publics : `lengthscales` (ndarray (d,)) et `variance` (float).

### Régresseur

```python
from gp_lite import GPRegressor, SquaredExponentialARD

gp = GPRegressor(SquaredExponentialARD([0.5, 0.5]), noise=1e-6, normalize_y=True)
gp.fit(X, y)                       # optimise par défaut ; renvoie self
mean = gp.predict(Xtest)           # (m,), unités de y
mean, std = gp.predict(Xtest, return_std=True)   # std sans le bruit
ll = gp.log_marginal_likelihood()  # sur y_n si normalize_y
mean_loo, var_loo = gp.loo()       # (n,), (n,) : moyenne/var prédictives leave-one-out
```

Comportements clés :

- `fit(X, y, optimize=True)` : maximise la log-vraisemblance marginale en
  optimisant `log(lengthscales)`, `log(variance)`, `log(noise)` par
  L-BFGS-B (3 redémarrages, graine fixée, meilleur point conservé), avec un
  jitter progressif si la factorisation de Cholesky échoue.
- `fit(X, y, optimize=False)` : garde les hyperparamètres actuels et ne fait
  que conditionner sur les données.
- Après `fit`, `gp.kernel` porte les hyperparamètres ajustés et `gp.noise` la
  variance du bruit ajustée.
- `normalize_y=True` : y_n = (y − moy) / std (np.std, ddof = 0). La
  vraisemblance et `noise` sont exprimés sur y_n ; les prédictions et le LOO
  sont renvoyés dans les unités de y.

## Formules utilisées (Rasmussen & Williams, 2005)

Avec `C = K(X, X) + noise·I` et `y_n` la cible d'entraînement :

- Log-vraisemblance marginale :
  `log p(y|X) = −½ y_nᵀ C⁻¹ y_n − ½ log|C| − (n/2) log 2π`.
- Prédiction (moyenne et variance de f, sans le bruit) :
  `μ(x) = k(x)ᵀ C⁻¹ y_n`,  `var f(x) = k(x,x) − k(x)ᵀ C⁻¹ k(x)`.
- Leave-one-out en forme fermée (§ 5.4.2), avec `A = C⁻¹`, `b = A y_n` :
  `m_i = y_i − b_i / A_ii`,  `v_i = 1 / A_ii` (variance prédictive de y_i
  sachant y_{−i}, bruit inclus). Calculé sans refaire n ajustements.

## Tests

```
python -m pytest -q tests
```

La suite contient :

- noyaux : symétrie, diagonale = variance, Matérn 5/2 à r = 0 ;
- prédictions : interpolation à bruit nul (erreur < 1e-6) ;
- LOO : équivalence avec n réajustements explicites sans ré-optimisation ;
- comparaison avec scikit-learn (`GaussianProcessRegressor`, RBF ARD +
  WhiteKernel, hyperparamètres fixés identiques, `optimizer=None`) ;
- **vérifications contre des références naïves et indépendantes**
  (`np.linalg.solve`, `np.linalg.inv`, `np.linalg.slogdet`, boucles), sans
  réutiliser le code du paquet, pour plusieurs jeux d'hyperparamètres, avec et
  sans `normalize_y` : noyaux, log-vraisemblance, moyenne/écart-type, LOO ;
- bout en bout sur Branin : 30 points LHS, 200 points test, RMSE normalisée
  < 0.1 ; la log-vraisemblance obtenue par `fit` est vérifiée par la référence
  naïve et comparée à celle de scikit-learn optimisé, sur un LHS **et deux
  plans aléatoires**.

### Résultats obtenus

Sortie de `python -m pytest -q tests` :

```
22 passed, 4 warnings in 2.52s
```

Sortie de `python examples/branin_demo.py` :

```
hyperparamètres ajustés : lengthscales=[7.219 19.2452], variance=131.8, noise=8.315e-07
log-vraisemblance marginale : 56.081
RMSE = 0.081   RMSE normalisée = 0.0027
graphique écrit : examples/branin_demo.png
```

## Exemple

```
python examples/branin_demo.py
```

Entraîne le GP sur la fonction de Branin (2D) avec 30 points LHS et écrit
`examples/branin_demo.png` (backend Agg) : moyenne prédite en couleurs,
courbes de niveau, points d'entraînement en rouge.
