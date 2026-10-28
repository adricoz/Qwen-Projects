# gp_lite — régression par processus gaussien

Petit paquet autonome (NumPy + SciPy uniquement, **pas de scikit-learn dans le paquet**)
de régression GP avec noyaux stationnaires ARD, pensé pour des surrogates aéro/hydro.

## Installation

Aucune : seul NumPy, SciPy (et, pour les tests, pytest + scikit-learn) sont requis,
tous deux déjà disponibles. `conftest.py` met le dossier du projet sur `sys.path`.

## API

```python
from gp_lite import GPRegressor, SquaredExponentialARD, Matern32ARD, Matern52ARD

# Noyaux : Classe(lengthscales, variance=1.0)
k = SquaredExponentialARD(np.array([1.0, 1.0]), variance=1.0)

gp = GPRegressor(k, noise=1e-6, normalize_y=True)
gp.fit(X, y, optimize=True)          # renvoie self ; optimise log(ls), log(var), log(noise)
mean, std = gp.predict(Xs, return_std=True)   # std = écart-type de f (sans le bruit)
lml = gp.log_marginal_likelihood()    # sur y_n si normalisé
mu_loo, var_loo = gp.loo()            # leave-one-out, forme fermée, en unités de y
```

- **Noyaux** (`gp_lite/kernels.py`) : `K(X1, X2) -> (n1, n2)` et `diag(X) -> (n,)`.
  Convention Matérn : `r = sqrt(sum(((x - x')/l)^2))` ;
  Matérn 5/2 = `σ²(1 + √5 r + 5r²/3) exp(−√5 r)`, Matérn 3/2 = `σ²(1 + √3 r) exp(−√3 r)`,
  SE = `σ² exp(−r²/2)`.
- **`fit(X, y, optimize=True)`** : maximise la log-vraisemblance marginale (L-BFGS-B,
  `scipy.optimize.minimize`) sur les paramètres log, avec 3 redémarrages aléatoires
  (graine fixée, `np.random.default_rng(0)`) et conservation du meilleur ;
  jitter progressif si la Cholesky échoue. Avec `optimize=False`, les hyperparamètres
  courants sont conservés et le GP conditionne seulement sur les données.
  Après `fit`, `gp.kernel` porte les hyperparamètres ajustés et `gp.noise` la variance
  de bruit ajustée.
- **`normalize_y=True`** : `y_n = (y − moyenne)/écart-type` (`np.std`, ddof = 0) ;
  la vraisemblance et `noise` sont sur `y_n`, les prédictions sont renvoyées en unités de y.
- **`predict(X, return_std)`** : moyenne a posteriori ; `std` sans le bruit.
- **`loo()`** : moments prédictifs leave-one-out (moyenne, variance bruit incluse),
  en forme fermée, sans refaire n ajustements.

## Formules

Avec `K* = K(X, X) + σ_n² I`, `α = K*⁻¹ y_n` (vraisemblance) et
`β = (K + σ_n² I)^{-1}` (modèle bruité, pour le LOO) :

- Log-vraisemblance marginale :
  `log p(y|X,θ) = −½ yᵀ K*⁻¹ y − ½ log|K*| − (n/2) log(2π)`.
- Prédiction : `μ*(x) = k(x)ᵀ α`, `σ_f²(x) = k(x,x) − k(x)ᵀ K*⁻¹ k(x)` (variance de f,
  sans le bruit).
- Leave-one-out (Rasmussen & Williams §5.4.2) : pour chaque `i`,
  `S_i = 1/β_ii` (complément de Schur) ;
  `mu_loo_i = y_i − S_i α_i` et `var_loo_i = S_i` (bruit inclus, en unités de y
  via la normalisation du jeu d'apprentissage).

## Tests

```bash
python -m pytest -q tests            # tests unitaires du paquet
python -m pytest -q tests_acceptance # 23 tests d'acceptation (référence indépendante + sklearn)
```

### Sortie de pytest (obtention réelle)

```
$ python -m pytest -q tests tests_acceptance
.....................................                                    [100%]
37 passed, 18 warnings in 3.95s
```

(les 18 warnings proviennent de scikit-learn / pandas, pas du paquet)

## Exemple Branin

```bash
python examples/branin_demo.py
```

Sortie obtenue :

```
lengthscales: [ 4.437 26.232]
variance    : 98.773
noise       : 3.044e-07
LML         : 15.456
RMSE normalisée (200 pts test) : 0.0126
figure écrite : examples/branin_demo.png
```

Le test de bout en bout exige une RMSE normalisée < 0.1 ; on obtient **0.013**.

## Structure

```
gp_lite/kernels.py        noyaux SE ARD, Matérn 3/2 ARD, Matérn 5/2 ARD
gp_lite/gp.py             GPRegressor (fit, predict, log_marginal_likelihood, loo)
tests/test_gp_lite.py     tests unitaires (noyaux, interpolation, loo, sklearn, Branin)
examples/branin_demo.py   démo + figure examples/branin_demo.png
conftest.py               ajoute le projet sur sys.path (pytest)
```
