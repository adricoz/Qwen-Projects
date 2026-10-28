# `gp_lite` — petit paquet de régression par processus gaussien

`gp_lite` est un paquet Python autonome de régression par processus gaussien (GP),
écrit avec **NumPy + SciPy uniquement** (pas de scikit-learn dans le paquet,
scikit-learn est utilisé **dans les tests seulement** pour une comparaison).

## Installation

Aucune installation nécessaire : le paquet est importable directement depuis la
racine du projet (les dépendances `numpy` et `scipy` sont supposées disponibles ;
`matplotlib`, `pytest` et `scikit-learn` pour les tests/la démo).

```bash
cd <dossier du projet>
python -m pytest -q tests          # lancer les tests
python examples/branin_demo.py    # démo Branin -> examples/branin_demo.png
```

## API

### Noyaux — `gp_lite.kernels`

```python
from gp_lite import SquaredExponentialARD, Matern32ARD, Matern52ARD

k = Matern52ARD(lengthscales=[0.4, 0.9], variance=2.0)
K  = k.K(X1, X2)     # (n1, d), (n2, d) -> matrice de covariance (n1, n2)
v  = k.diag(X)       # vecteur des variances marginales (n,) (= variance)
k.lengthscales       # ndarray (d,)
k.variance           # float
```

Convention d'écart : `r = sqrt(sum(((x - x') / l) ** 2))` avec `l` les longueurs
caractéristiques ARD. Formules :

- **SquaredExponential** : `K = sigma² exp(-r²/2)` ;
- **Matérn 3/2** : `K = sigma² (1 + sqrt(3) r) exp(-sqrt(3) r)` ;
- **Matérn 5/2** : `K = sigma² (1 + sqrt(5) r + 5 r²/3) exp(-sqrt(5) r)`.

### Régresseur — `gp_lite.GPRegressor`

```python
from gp_lite import GPRegressor, SquaredExponentialARD

gp = GPRegressor(SquaredExponentialARD([10.0, 2.0]), noise=1e-6, normalize_y=True)
gp.fit(X, y)                 # renvoie self ; optimise les hyperparamètres
gp.fit(X, y, optimize=False) # conditionne sans toucher aux hyperparamètres

mean = gp.predict(X)                       # ndarray (m,)
mean, std = gp.predict(X, return_std=True) # std : écart-type de f (sans bruit)
lml = gp.log_marginal_likelihood()         # log-vraisemblance marginale (sur y_n)
mloo, vloo = gp.loo()                      # moyenne et variance leave-one-out de y_i
```

Comportements imposés :

- `fit` optimise `log(lengthscales)`, `log(variance)`, `log(noise)` par
  `scipy.optimize.minimize` (L-BFGS-B) sur la log-vraisemblance négative calculée
  par Cholesky (jitters croissants si la factorisation échoue), avec **3
  redémarrages aléatoires à graine fixée** en conservant le meilleur ;
- après `fit`, `gp.kernel` porte les hyperparamètres ajustés et `gp.noise` la
  variance de bruit ajustée ;
- `normalize_y=True` : `y_n = (y − mean(y)) / std(y)` (`np.std`, ddof=0) ; la
  vraisemblance et `noise` sont exprimés sur `y_n`, les prédictions sont renvoyées
  dans les unités de `y` ;
- `loo()` renvoie `(mean, var)` de forme `(n,)` : prédictions leave-one-out de
  `y_i` (bruit inclus), sans refaire n ajustements.

## Formules utilisées

Avec `K = K(X, X) + noise·I`, `A = K⁻¹`, `α = A y` :

- **Log-vraisemblance marginale** :
  `ℓ = -½ yᵀA y - log det(K) - (n/2) log(2π)`.
- **Prédiction** (moyenne a posteriori, variance sans bruit) :
  `μ(x) = k(x)ᵀ α`, `var f(x) = k(x,x) - k(x)ᵀ A k(x)`.
- **Leave-one-out** (Rasmussen & Williams, §5.4.2) :
  `mean_i = y_i - α_i / A_ii`, `var_i = 1 / A_ii`
  (identité de l'inversion par blocs : `k_iᵀ K₋ᵢ⁻¹ k_i = K_ii - 1/A_ii`).

## Tests et résultats

Lancement : `python -m pytest -q tests` — sortie obtenue (10 tests, tous passants) :

```
..........
============================== warnings summary ===============================
tests/test_gp_lite.py::test_against_sklearn_fixed_hyperparameters
  ... UserWarning: Pandas requires version '2.10.2' or newer of 'numexpr' (version '2.10.1' currently installed).
    from pandas.core.computation.check import NUMEXPR_INSTALLED
10 passed, 1 warning in 1.52s
```

Contenu des tests : symétrie des noyaux, diagonale = variance, Matérn 5/2 = σ² à
r = 0 ; interpolation à bruit nul (erreur < 1e-6) ; `loo()` cohérente avec le
recalcul explicite par n ajustements sans ré-optimisation (tol 1e-6) ; égalité
des moyennes avec `sklearn.gaussian_process.GaussianProcessRegressor`
(RBF ARD + WhiteKernel, hyperparamètres fixés, `optimizer=None`) à 1e-6 près ;
`fit` renvoie `self` et met à jour les hyperparamètres.

Démo de bout en bout (`examples/branin_demo.py`) — fonction de Branin 2D,
30 points LHS, 200 points de test :

```
Hyperparametres : l = [ 42.65000909 403.42879349], variance = 218.1213, bruit = 6.14e-06
Log-vraisemblance marginale : 213.17
RMSE test : 214.1397  (normalisee : 0.0012)
Figure écrite : .../examples/branin_demo.png
```

soit une **RMSE normalisée ≈ 0.0012**, bien sous le seuil de 0.1 exigé par les tests.

## Limitations connues

- Moyenne a priori constante (nulle sur `y_n`) ; pas de fonction de moyenne
  apprise et pas de co-krigeage multi-fidélité (le paquet se limite au GP unique) ;
- `loo()` utilise une inversion dense de `K` (O(n³)), ce qui reste exact mais
  n'est pas adapté à n très grand ;
- L'optimisation se borne aux intervalles log(`l`) ∈ [e⁻⁶, e⁶],
  log(`variance`) ∈ [e⁻⁸, e⁸], log(`noise`) ∈ [e⁻¹², 1] (bornes fixes).
