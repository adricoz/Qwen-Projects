# gp_lite

Petit paquet autonome de **régression par processus gaussien (GP)** écrit en
Python, pour des surrogates aéro/hydro. Il repose **uniquement sur NumPy et
SciPy** (scikit-learn n'est utilisé que par les tests de référence, jamais
dans le paquet).

Fonctionnalités :
- Noyaux ARD : RBF (Squared Exponential), Matérn 3/2, Matérn 5/2.
- `GPRegressor` : optimisation de la log-vraisemblance marginale par
  `scipy.optimize.minimize` (L-BFGS-B, 3 redémarrages, jitter auto),
  prédiction moyenne / écart-type a posteriori, log-vraisemblance marginale
  et **résidus leave-one-out en forme fermée** (Rasmussen & Williams, §5.4.2)
  sans refaire n ajustements.

## Installation

Aucune : le paquet utilise uniquement NumPy et SciPy (déjà disponibles).
Pour lancer les tests il faut pytest et scikit-learn (tests de référence).

## Structure

```
gp_lite/
  __init__.py
  kernels.py      # SquaredExponentialARD, Matern32ARD, Matern52ARD
  gp.py           # GPRegressor
tests/
  test_gp_lite.py
examples/
  branin_demo.py  # démo Branin -> branin_demo.png
```

## API

### Noyaux

```python
from gp_lite import SquaredExponentialARD, Matern32ARD, Matern52ARD

k = SquaredExponentialARD(lengthscales=[0.5, 1.0], variance=2.0)
k.K(X1, X2)   # matrice de covariance (n1, n2)
k.diag(X)     # diagonale K(X, X) (n,)
k.lengthscales  # ndarray (d,)
k.variance      # float
```

Convention des longueurs d'onde (ARD) :
`r = sqrt( sum_i ((x_i - x'_i) / l_i)^2 )`.

- **Squared Exponential** : `K = sigma^2 exp(-0.5 r^2)`.
- **Matérn 3/2** : `K = sigma^2 (1 + sqrt(3) r) exp(-sqrt(3) r)`.
- **Matérn 5/2** : `K = sigma^2 (1 + sqrt(5) r + 5 r^2 / 3) exp(-sqrt(5) r)`.

### GPRegressor

```python
from gp_lite import GPRegressor, SquaredExponentialARD

gp = GPRegressor(SquaredExponentialARD([1.0, 1.0]), noise=1e-6, normalize_y=True)
gp.fit(X, y, optimize=True)        # renvoie self
mean      = gp.predict(Xt)         # moyenne a posteriori
mean, std = gp.predict(Xt, return_std=True)   # std de f (sans bruit)
lml = gp.log_marginal_likelihood()
loo_mean, loo_var = gp.loo()       # résidus leave-one-out (bruit inclus)
gp.kernel.lengthscales; gp.kernel.variance; gp.noise
```

### Comportement d'`optimize`

- `fit(X, y, optimize=True)` : optimise `log(lengthscales)`, `log(variance)`,
  `log(noise)` par L-BFGS-B sur la log-vraisemblance marginale négative (3
  redémarrages aléatoires à graine fixée, le meilleur point est conservé).
  Après `fit`, `self.kernel` porte les hyperparamètres ajustés et `self.noise`
  la variance de bruit ajustée.
- `fit(X, y, optimize=False)` : conserve les hyperparamètres actuels de
  `self.kernel` et `self.noise` et ne fait que conditionner le GP sur les
  données (utile pour la comparaison LOO et les tests à hyperparamètres fixés).

### Normalisation de y

`normalize_y=True` (par défaut) : `y_n = (y - mean(y)) / std(y)` avec
`std = np.std` (ddof = 0). La log-vraisemblance marginale et `noise` sont
exprimées sur `y_n`. Les prédictions (moyenne et écart-type) sont renvoyées
dans les **unités de y**.

## Formules utilisées

Soit `K = kernel(K(X, X))`, `C = K + noise * I` (dans l'espace `y_n` si
`normalize_y=True`), et `alpha = C^{-1} y`.

- **Log-vraisemblance marginale** :
  `lml = -0.5 y^T C^{-1} y - log det(C) - (n/2) log(2 pi)`.
  Calculée par factorisation de Cholesky ; si la factorisation échoue un jitter
  est ajouté et on réessaie.

- **Prédiction a posteriori** (au point `*`) :
  - moyenne : `f_mean = k_*^T C^{-1} y` (avec `k_* = K(*, X)`)
  - variance (sans bruit) : `var(f) = kernel(*) - k_*^T C^{-1} k_*`
  - si `return_std=True`, on renvoie `std = sqrt(var(f)) * std_y + ...`
    (écart-type de `f` dans les unités de `y`, bruit non inclus).

- **Leave-one-out (Rasmussen & Williams §5.4.2)** :
  avec `A = C^{-1}`, `alpha = A y` et `a_ii = A[i, i]` :
  - `mean_loo(i) = y_i - alpha_i / a_ii`
  - `var_loo(i)  = 1 / a_ii`   (complément de Schur, bruit inclus)
  Ces quantités sont renvoyées dans les unités de `y`.

## Lancer les tests

```
python -m pytest -q tests
```

### Résultat de la suite de tests (vérifié)

```
$ python -m pytest -q tests
..........
10 passed, 1 warning in 1.43s
```

Les 10 tests couvrent :
- symétrie des noyaux et diagonale = variance ;
- Matérn 5/2 à `r = 0` vaut `sigma^2` ;
- interface du constructeur `Classe(lengthscales, variance=1.0)` ;
- bloc croisé `K(X1, X2)` avec `n1 != n2` ;
- `predict` interpole les points d'apprentissage à bruit quasi nul ;
- `loo()` coïncide avec le recalcul explicite par n ajustements (hyperparams
  fixés, `optimize=False`) à 1e-6 ;
- LML et unités de bruit ;
- comparaison avec scikit-learn (`GaussianProcessRegressor`, noyau RBF ARD +
  WhiteKernel, hyperparamètres FIXÉS identiques, `optimizer=None`) : moyennes
  prédites égales à 1e-6 ;
- test de bout en bout sur Branin : 30 points LHS → RMSE normalisée < 0.1 ;
- `fit()` renvoie `self` et expose hyperparamètres/`noise` après optimisation.

### Résultat de la démo Branin (`python examples/branin_demo.py`)

```
RMSE (Branin, 30 pts LHS): 34.650
RMSE normalisée (par std du test): 0.0285
Hyperparamètres finaux: l=[14.294 54.598], variance=22.705, noise=0.002
log-vraisemblance marginale: 24.947
Figure écrite dans: examples/branin_demo.png
```

La RMSE normalisée (0.0285) est bien inférieure au seuil de 0.1 exigé par le
test de bout en bout.
