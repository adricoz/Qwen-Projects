# gp_lite — regression par processus gaussien

Paquet Python autonome (NumPy + SciPy uniquement, scikit-learn n'est utilisé
que par les tests) implémentant un GP régresseur à moyenne nulle avec bruit
blanc, des noyaux ARD (Squred Exponential, Matérn 3/2 et 5/2), l'optimisation
des hyperparamètres par vraisemblance marginale, les prédictions a posteriori
et les résidus leave-one-out en forme fermée.

## Installation

Aucune : le paquet n'utilise que NumPy et SciPy (déjà présents).
Les tests utilisent en plus `pytest` et `scikit-learn`.

```bash
python -m pytest -q tests
```

## API

```python
from gp_lite import GPRegressor, Matern52ARD   # et SquaredExponentialARD, Matern32ARD

gp = GPRegressor(Matern52ARD(lengthscales=[0.3, 0.3], variance=1.0),
                 noise=1e-6, normalize_y=True)
gp.fit(X_train, y_train)                    # optimise (L-BFGS-B, 3 redémarrages)
mean = gp.predict(X_new)                    # moyenne (m,)
mean, std = gp.predict(X_new, return_std=True)   # std de f, sans le bruit
ll = gp.log_marginal_likelihood()           # LML sur y_n
loo_mean, loo_var = gp.loo()                # résidus leave-one-out (n,), unités de y
```

- Noyaux : `Classe(lengthscales, variance=1.0)` avec attributs publics
  `lengthscales` (ndarray `(d,)`) et `variance` (float), méthodes
  `K(X1, X2) -> (n1, n2)` et `diag(X)`.
- `fit(X, y, optimize=True)` renvoie `self`. Avec `optimize=False`, les
  hyperparamètres courants de `self.kernel` et `self.noise` sont conservés et
  le GP se contente de conditionner sur les données.
- Après `fit`, `gp.kernel` porte les hyperparamètres ajustés et `gp.noise` la
  variance de bruit ajustée.
- `normalize_y=True` : `y_n = (y − mean(y)) / std(y)` (`np.std`, ddof = 0) ;
  vraisemblance et bruit exprimés sur `y_n`, prédictions renvoyées en unités de `y`.
- `loo()` renvoie `(mean, var)` : moyenne et variance prédictives leave-one-out
  de chaque `y_i` (bruit inclus), en unités de `y`.

## Formules

Notations : `K` matrice de noyau, `A = K + σ_n² I`, `α = A⁻¹ y_n`,
`β_i = (A⁻¹)_{ii}`, `r = sqrt(Σ_j ((x_j − x'_j)/ℓ_j)²)`.

**Noyaux** : SE `σ² exp(−r²/2)` ; Matérn 3/2 `σ²(1+√3 r)e^{−√3 r}` ;
Matérn 5/2 `σ²(1+√5 r+5r²/3)e^{−√5 r}`.

**Log-vraisemblance marginale** :

```
log p(y_n | θ) = −½ y_nᵀ A⁻¹ y_n − log det(A)^{1/2} − n/2 log(2π)
```

calculée par Cholesky ; si la factorisation échoue, un jitter
(`jitter × max(1, max|diag(A)|) × I`, `jitter ∈ {0, 1e-10, 1e-8, 1e-6}`)
est ajouté. L'optimisation maximise la LML sur
`θ = [log ℓ_1 … log ℓ_d, log σ², log σ_n²]` (bornes
`ℓ, σ² ∈ [e⁻⁶, e⁶]`, `σ_n ∈ [e⁻¹⁸, e⁵]`) par L-BFGS-B depuis 4 points de
départ (l'état courant + 3 redémarrages aléatoires, graine fixée à 0), en
conservant le meilleur.

**Prédiction** (Rasmussen & Williams, algorithme 2.1) :

```
E[f*] = k_sᵀ A⁻¹ y_n          (k_s = K(X*, X))
Var[f*] = K(X*, X*)_ii − k_sᵀ A⁻¹ k_s      (sans le bruit)
```

**Leave-one-out** (section 5.4.2), sans refaire `n` ajustements :

```
mean_i = y_n_i − α_i / β_i
var_i  = 1 / β_i
```

(`var_i = 1/β_i` car `A = K + σ_n² I` et `K_ii = σ²`).

Les tests de référence (`tests/reference_naive.py`) recalculent le noyau
point par point, la LML avec `np.linalg.inv`/`slogdet`, la prédiction avec
`np.linalg.inv` et le LOO par `n` ré-ajustements explicites — sans aucun code
du paquet.

## Résultats obtenus

`python -m pytest -q tests` :

```
.................................................                        [100%]
49 passed, 4 warnings in 2.66s
```

(les warnings proviennent de l'environnement pandas/numexpr et de la
ConvergenceWarning de scikit-learn sur le bruit ; aucun ne concerne gp_lite)

Démonstration Branin (`python examples/branin_demo.py`, 30 points LHS,
200 points de test) :

```
RMSE        : 0.1239
RMSE normal.: 0.0176
figure      : examples/branin_demo.png
```

## Contenu

```
gp_lite/kernels.py    noyaux ARD (SE, Matérn 3/2, Matérn 5/2)
gp_lite/gp.py         GPRegressor (fit, predict, LML, LOO)
tests/test_gp_lite.py tests fonctionnels + parité scikit-learn (hyperparamètres fixés)
tests/test_fit.py     vérification du fit : LML vs référence naïve + sklearn optimisé
tests/reference_naive.py  références naïves indépendantes (inv/slogdet/boucles)
examples/branin_demo.py   démo Branin -> examples/branin_demo.png
conftest.py           ajoute le dossier courant à sys.path pour pytest
```
