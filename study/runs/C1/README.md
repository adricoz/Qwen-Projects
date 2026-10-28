# gp_lite

Petit paquet de **régression par processus gaussien** (GP) en Python,
autonome : uniquement **NumPy** et **SciPy** (pas de scikit-learn dans le
paquet). Pense pour des surrogates aéro/hydro : noyaux stationnaires ARD,
optimisation de la log-vraisemblance marginale, prédictions a posteriori et
résidus *leave-one-out* en forme fermée.

## Installation

Aucune : le paquet est autonome et n'utilise que des dépendances déjà
disponibles (NumPy, SciPy). Pour les tests : `pytest` et, pour le seul test
de comparaison, `scikit-learn`. Pour la démo : `matplotlib`.

## Structure

```
gp_lite/
    kernels.py   # SquaredExponentialARD, Matern32ARD, Matern52ARD
    gp.py        # GPRegressor
tests/
    test_gp_lite.py
examples/
    branin_demo.py   # démo sur Branin + branin_demo.png
```

## API

### Noyaux

```python
from gp_lite import SquaredExponentialARD, Matern32ARD, Matern52ARD

k = Matern52ARD(lengthscales=[1.0, 2.0], variance=1.0)
K  = k.K(X1, X2)   # (n1, d), (n2, d) -> (n1, n2)
d  = k.diag(X)     # (n,) : diagonale de la covariance (= variance)
```

- Attributs publics : `lengthscales` (ndarray `(d,)`), `variance` (float).
- Convention Matérn : `r = sqrt(sum(((x - x')/l)^2))` ;
  Matérn 5/2 : `σ² (1 + √5 r + 5r²/3) exp(−√5 r)` ;
  Matérn 3/2 : `σ² (1 + √3 r) exp(−√3 r)` ;
  SE/RBF : `σ² exp(−r²/2)`.

### GPRegressor

```python
from gp_lite import GPRegressor, Matern52ARD

gp = GPRegressor(Matern52ARD([1.0, 1.0]), noise=1e-6, normalize_y=True)
gp.fit(X, y, optimize=True)          # renvoie self
mean, std = gp.predict(X_test, return_std=True)
loglik    = gp.log_marginal_likelihood()
loo_mean, loo_var = gp.loo()
```

- `fit(X, y, optimize=True)` : maximise la log-vraisemblance marginale par
  `scipy.optimize.minimize` (L-BFGS-B) sur θ = (log l₁..log l_d, log σ²,
  log σ_n²), avec **3 redémarrages aléatoires** (graine fixée) et
  conservation du meilleur ; jitter croissant (1e-10 → 1e-4) si la
  factorisation de Cholesky échoue. Avec `optimize=False`, les
  hyperparamètres courants de `self.kernel` et `self.noise` sont conservés.
- Après `fit`, `self.kernel` porte les hyperparamètres ajustés et
  `self.noise` la variance de bruit ajustée.
- `normalize_y=True` : y est standardisé (`y_n = (y − ȳ)/s`, `s = np.std(y)`,
  ddof = 0) avant le conditionnement ; la vraisemblance et `noise` sont sur
  y_n ; les prédictions sont renvoyées dans les unités de y.
- `predict(X, return_std=True)` → `(mean, std)` : la std est celle de **f**
  (bruit exclu).
- `loo()` → `(mean, var)` : moments prédictifs leave-one-out de y_i
  (bruit inclus) en forme fermée, unités de y.

## Formules utilisées

Notations : n points, `K = k(X, X) + σ_n² I`, `L` = Cholesky de K
(triangulaire inférieure, `L Lᵀ = K`), y_n les sorties normalisées.

- **α** = K⁻¹ y_n = (Lᵀ)⁻¹ L⁻¹ y_n
- **Log-vraisemblance marginale** :
  `log p(y_n|X) = −½ βᵀβ − Σᵢ log Lᵢᵢ − (n/2) log 2π`,
  avec β = L⁻¹ y_n (donc βᵀβ = y_nᵀ K⁻¹ y_n).
- **Prédiction** en x* : k* = k(X, x*),
  - moyenne : `μ* = k*ᵀ α`
  - variance de f : `k(x*, x*) − k*ᵀ K⁻¹ k*` (v = L⁻¹ k*, `var = k*·k* − v·v`)
- **Leave-one-out** (Rasmussen & Williams, éq. 5.12), avec
  `[K⁻¹]ᵢᵢ` la diagonale de K⁻¹ :
  - `μ₋ᵢ = y_n,i − αᵢ / [K⁻¹]ᵢᵢ`
  - `σ²₋ᵢ = 1 / [K⁻¹]ᵢᵢ` (variance de y_i, bruit inclus)
- **Retour aux unités de y** : `μ = μ_n · s + ȳ`, `var × s²`, avec
  `ȳ` et `s = np.std(y)` du jeu d'apprentissage complet.
- **Optimisation** : minimisation de −log p(y_n|X) en θ =
  (log l₁..log l_d, log σ², log σ_n²), bornes log 1e-4..1e4 (longueurs),
  1e-8..1e8 (variance), 1e-12..1e-2 (bruit).

## Tests

```
python -m pytest -q tests
```

### Résultats

```
......
============================== warnings summary ===============================
tests/test_gp_lite.py::test_matches_sklearn_fixed_hyperparameters
  ... UserWarning: Pandas requires version '2.10.2' or newer of 'numexpr'
======================= 6 passed, 1 warning in 1.55s =========================
```

Les tests couvrent :
- noyaux : symétrie, diagonale = variance, valeur exacte du Matérn 5/2 ;
- interpolation exacte des points d'entraînement à bruit nul (< 1e-6) ;
- `loo` en forme fermée ≡ n ré-ajustements explicites (tol. 1e-6) ;
- concordance des moyennes prédites avec `sklearn.gaussian_process`
  (RBF ARD + WhiteKernel, hyperparamètres fixés identiques, `optimizer=None`)
  à 1e-6 près ;
- de bout en bout Branin 2D : 30 points LHS, 200 points test,
  RMSE normalisée < 0.1.

### Démo Branin

```
python examples/branin_demo.py
```

Sortie obtenue :

```
Hyperparametres ajustes :
  lengthscales = [ 47.90276593 143.71677001]
  variance     = 7517.6635552885555
  noise        = 1.6123521618136035e-06
log-vraisemblance marginale = 36.13895328762618
RMSE normalisee (200 points test) = 0.008789593806691398
Figure sauvegardee : examples/branin_demo.png
```

La figure `examples/branin_demo.png` montre la moyenne prédite (isovaleurs)
et les 30 points d'entraînement.

## Limites connues

- Bruit homoscédastique uniquement (pas de WhiteKernel appris séparément,
  mais la variance de bruit est optimisée).
- `loo()` utilise un inverse explicite de K (O(n³)) — adapté à de petits/
  moyens ensembles de points d'apprentissage.
