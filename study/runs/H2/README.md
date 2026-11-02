# gp_lite

Petit paquet autonome de **régression par processus gaussien** (GP) en
Python, écrit avec **NumPy + SciPy uniquement** (pas de scikit-learn dans le
paquet ; scikit-learn n'est utilisé que par un test de comparaison).

Pensé pour servir de surrogat aéro/hydro : moyennes et incertitudes
a posteriori, vraisemblance marginale, leave-one-out en forme fermée.

## Installation

Aucune : le paquet est autonome et ne dépend que de NumPy et SciPy
(déjà présents). Pour tester, `pytest` et `scikit-learn` sont requis ;
`matplotlib` est requis pour l'exemple.

## API

### Noyaux (`gp_lite.kernels`)

```python
from gp_lite import SquaredExponentialARD, Matern32ARD, Matern52ARD

k = SquaredExponentialARD(lengthscales=[0.5, 1.0], variance=1.0)
K = k.K(X1, X2)   # (n1, d), (n2, d) -> (n1, n2)
d = k.diag(X)     # (n,) : diagonale du noyau (= variance en ARD)
```

Attributs publics : `k.lengthscales` (ndarray `(d,)`) et `k.variance`
(float). Convention Matérn : `r = sqrt(sum(((x - x')/l)^2))`, avec
Matérn 5/2 = σ²(1 + √5 r + 5 r²/3) e^{−√5 r} et Matérn 3/2 =
σ²(1 + √3 r) e^{−√3 r}.

### Régresseur (`gp_lite.gp`)

```python
from gp_lite import GPRegressor

gp = GPRegressor(SquaredExponentialARD([0.5, 0.5]), noise=1e-6,
                 normalize_y=True)
gp.fit(X, y, optimize=True)        # renvoie self

mean = gp.predict(Xt)                              # (m,)
mean, std = gp.predict(Xt, return_std=True)        # std de f (sans bruit)
ll = gp.log_marginal_likelihood()                  # sur y_n si normalisé
mean_loo, var_loo = gp.loo()                       # (n,), y_n, bruit inclus
```

Semantique :

- `fit(X, y, optimize=False)` : conserve les hyperparamètres courants de
  `self.kernel` et `self.noise`, ne fait que conditionner sur les données.
- Après `fit`, `gp.kernel` porte les hyperparamètres ajustés et `gp.noise`
  la variance de bruit ajustée.
- `normalize_y=True` : `y_n = (y − mean) / std` (`np.std`, ddof = 0).
  La vraisemblance et le bruit sont exprimés sur `y_n` ; les prédictions
  sont renvoyées dans les unités de `y`.

## Formules utilisées

Avec `M = K(X, X) + σ_n² I`, `y` normalisé si `normalize_y=True` :

- **Log-vraisemblance marginale** :
  `log p(y) = −½ yᵀM⁻¹y − ½ log|K + σ_n² I| − (n/2) log(2π)`
  (calculée par factorisation de Cholesky).
- **Ajustement** : minimisation de `−log p(y)` par L-BFGS-B sur
  `(log lengthscales, log variance, log noise)` ; points de départ :
  hyperparamètres courants + 12 redémarrages aléatoires log-uniformes
  (graine fixée à 42) + grille grossière quand la dimension est petite,
  puis affinement du voisinage des meilleurs points ; le meilleur minimum
  est retenu. Jitter croissant si la factorisation échoue.
- **Prédiction** (point hors échantillon) :
  `μ(x) = k(x)ᵀM⁻¹y`, `Var f(x) = k(x,x) − k(x)ᵀM⁻¹k(x)` ; le bruit n'est
  pas inclus dans `std`.
- **Leave-one-out** (Rasmussen & Williams, 5.4.2), via complément de
  Schur, sans refaire n ajustements :
  `μ_LOO(i) = y_i − (M⁻¹y)_i / (M⁻¹)_ii` et
  `Var_LOO(i) = 1 / (M⁻¹)_ii` ; résultats renvoyés dans les unités de `y`.

## Tests

```bash
python -m pytest -q tests
```

Résultat obtenu :

```
$ python -m pytest -q tests
............
12 passed, 1 warning in 1.55s
```

(Le warning unique est un `UserWarning` interne à pandas/numexpr, sans
rapport avec `gp_lite`.)

L'exemple de bout en bout sur la fonction de Branin (30 points LHS,
200 points de test) donne **RMSE = 0.0014** (RMSE/std ≈ 0.0001),
bien en dessous du seuil exigé de 0.1.

## Exemple

```bash
python examples/branin_demo.py
```

Entraine le GP sur la fonction de Branin (2D), affiche les
hyperparamètres ajustés et écrit `examples/branin_demo.png` (moyenne
prédite en cartographie + points d'apprentissage). Sortie obtenue :

```
Hyperparametres ajustes :
  lengthscales = [2.19520391 2.99478297]
  variance     = 12.133021480214024
  noise        = 5.711606925705767e-09
RMSE sur la grille : 0.0014 (std = 15.9658)
Figure ecrite dans .../examples/branin_demo.png
```

## Contenu du dépôt

```
gp_lite/
    __init__.py    # exports
    kernels.py     # SquaredExponentialARD, Matern32ARD, Matern52ARD
    gp.py          # GPRegressor (fit, predict, loo, log_marginal_likelihood)
tests/
    test_gp_lite.py
examples/
    branin_demo.py (+ branin_demo.png genere)
```
