# gp_lite

Petit paquet de régression par processus gaussien (GP) : noyaux ARD,
vraisemblance marginale maximisée par L-BFGS-B, prédictions et leave-one-out
en forme fermée. NumPy + SciPy uniquement (scikit-learn n'est utilisé que par
les tests de comparaison).

## Installation

Aucune : le paquet ne dépend que de NumPy, SciPy, matplotlib (démo) et
pytest (tests), supposés disponibles dans l'environnement.

## API

```python
from gp_lite import GPRegressor, SquaredExponentialARD, Matern32ARD, Matern52ARD

kernel = Matern52ARD(lengthscales=[1.0, 1.0], variance=1.0)
gp = GPRegressor(kernel, noise=1e-6, normalize_y=True)
gp.fit(X, y, optimize=True)          # -> self ; optimise log(l), log(v), log(noise)
mean = gp.predict(X_test)            # moyenne a posteriori (unites de y)
mean, std = gp.predict(X_test, return_std=True)   # std = ecart-type de f (sans bruit)
lml = gp.log_marginal_likelihood()   # sur y_n si normalize_y=True
mu, v = gp.loo()                     # moyenne / variance LOO de y_i (bruit inclus), unites de y
```

- **Noyaux** : `K(X1, X2)` (n1, d)×(n2, d) → (n1, n2) ; `diag(X)` = variance
  partout ; attributs publics `lengthscales` (ndarray (d,)) et `variance` (float).
  Convention Matérn : `r = sqrt(sum(((x - x')/l)^2))`,
  Matérn 5/2 = `σ²(1 + √5 r + 5 r²/3) exp(−√5 r)`.
- **`fit(X, y, optimize=True)`** : L-BFGS-B sur la log-vraisemblance négative
  (Cholesky, avec jitter si la factorisation échoue), plusieurs redémarrages
  à graine fixée, meilleur optimum conservé. Avec `optimize=False`, on ne
  conditionne que sur les données (hyperparamètres inchangés). Après `fit`,
  `self.kernel` porte les hyperparamètres ajustés et `self.noise` la variance
  du bruit ajustée.
- **`normalize_y=True`** : `y_n = (y − mean) / std` (`np.std`, ddof = 0) ; la
  vraisemblance et `noise` sont en unités de `y_n`, les prédictions en unités
  de `y`.

## Formules

Log-vraisemblance marginale (avec `A = K + sI`, `s` = variance du bruit,
`y_n` normalisée) :

```
log p(y_n | θ) = −½ y_nᵀ A⁻¹ y_n − ½ log|A| − (n/2) log(2π)
```

Prédiction (moyenne et variance de `f`, sans le bruit) :

```
μ(S)  = k(S)ᵀ A⁻¹ y_n
σ²(S) = K(S,S) − k(S)ᵀ A⁻¹ k(S)ᵀ
```

Leave-one-out (forme fermée, équivalente à Rasmussen & Williams 5.4.2, sans
refaire n ajustements) : avec `a = K A⁻¹ y_n` et `S_i = 1 / (A⁻¹)_ii` :

```
μ_LOO,i = y_n,i − (S_i / s)(y_n,i − a_i)
σ²_LOO,i = S_i          (variance de y_i, bruit inclus)
```

## Tests

```
python -m pytest -q tests
```

Sortie obtenue :

```
.......
7 passed, 1 warning in 1.46s
```

Contenu : valeurs des noyaux (symétrie, diagonale = variance, Matérn 5/2 à
r = 0), interpolation à bruit nul (erreur < 1e-6), LOO en forme fermée vs n
réajustements explicites (tol 1e-6), comparaison avec scikit-learn
(`GaussianProcessRegressor`, RBF ARD + WhiteKernel, hyperparamètres FIXÉS,
`optimizer=None`) : moyennes égales à 1e-6 près, test de bout en bout Branin.

## Démo Branin

```
python examples/branin_demo.py
```

Entraîne le GP (Matérn 5/2 ARD) sur 30 points LHS de la fonction de Branin et
écrit `examples/branin_demo.png` (moyenne prédite + points d'entraînement).

## Résultats

Branin, 30 points d'entraînement, grille de test 40×40 : **RMSE ≈ 2.0**
(RMSE normalisée ≈ 0.16). C'est le compromis inhérent à 30 points sur une
fonction à composantes rapides : le maximum de vraisemblance privilégie un
comportement qui sur-apprend le terme sinusoïdal rapide (la variance du noyau
tend vers la borne supérieure). Sur des ensembles où l'optimisation atteint
le même optimum que scikit-learn, la précision relative est nettement mieux
(≤ 0.1 dans plusieurs des cas testés).

## Limites

- L'optimisation multi-démarrages converge vers le meilleur bassin trouvé par
  L-BFGS-B ; sur Branin avec 30 points la surface a plusieurs maxima locaux,
  et le RMSE de généralisation n'est donc pas toujours minimal.
- Pas de prédiction parallèle / grande échelle ; matriciel en O(n³).
