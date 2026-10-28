# `gp_lite` — petit paquet de régression par processus gaussien

Paquet Python **autonome** (NumPy + SciPy seulement, sans scikit-learn)
implementant la regression par processus gaussien (GP) pour construire des
surrogates aéro/hydro : noyaux stationnaires ARD, maximisation de la
log-vraisemblance marginale, prediction a posteriori et résidus
leave-one-out en forme fermée.

## Installation

Aucune : le paquet n'a besoin que de `numpy`, `scipy`, `matplotlib`
(exemple) et `pytest` (tests). scikit-learn n'est utilisé que par un test
de comparaison (non requis par le paquet lui-meme).

```
├── gp_lite/
│   ├── __init__.py
│   ├── kernels.py      # SquaredExponentialARD, Matern32ARD, Matern52ARD
│   └── gp.py           # GPRegressor (fit / predict / loo / log_marginal_likelihood)
├── tests/test_gp_lite.py
├── examples/branin_demo.py
└── README.md
```

## API

```python
from gp_lite import GPRegressor, Matern52ARD, Matern32ARD, SquaredExponentialARD

# noyaux : distances echellees r = sqrt(sum(((x - x')/l)^2))
k = Matern52ARD(lengthscales=[0.5, 0.8], variance=1.0)
K  = k.K(X1, X2)      # matrice (n1, n2)
d  = k.diag(X)        # diagonale (= variance)

# modele
gp = GPRegressor(k, noise=1e-6, normalize_y=True)
gp.fit(X, y)                      # L-BFGS-B sur log(lengthscales), log(variance), log(noise)
mean      = gp.predict(Xt)        # moyenne a posteriori
mean, std = gp.predict(Xt, return_std=True)
ll  = gp.log_marginal_likelihood()
mu, v = gp.loo()                  # residus leave-one-out, forme fermee

# conditionnement SANS re-optimisation (hyperparametres fixes)
gp = GPRegressor(k, noise=1e-3, normalize_y=False)
gp.condition(X, y)
```

- `GPRegressor(kernel, noise, normalize_y)` : moyenne nulle, bruit gaussien
  sur les observations, normalisation optionnelle de y pour la stabilite de
  l'optimisation.
- `fit` optimise log(lengthscales), log(variance), log(noise) avec
  `scipy.optimize.minimize` (L-BFGS-B), **3 redemarrages aleatoires** (graine
  fixée, `default_rng(42)`) et conservation du meilleur. Cholesky avec
  **jitter croissant** si la factorisation echoue.
- `predict(X, return_std)` : moyenne et ecart-type a posteriori.
- `loo()` : predictions et variances leave-one-out, **sans refaire n
  ajustements**.

## Formules utilisees

Noyaux (convention Matérn de l'enonce) :

- SquaredExponential ARD : `K(x, x') = sigma² exp(-r²/2)`
- Matérn 3/2 : `K = sigma² (1 + sqrt(3) r) exp(-sqrt(3) r)`
- Matérn 5/2 : `K = sigma² (1 + sqrt(5) r + 5 r²/3) exp(-sqrt(5) r)`

Vraisemblance marginale (y ~ N(0, K + sigma_n² I), n = taille echantillon) :

    ll = -0.5 y^T (K + sigma_n² I)^{-1} y - sum_i log L_ii - (n/2) log(2 pi)

avec `K + sigma_n² I = L L^T` (Cholesky, facteur triangulaire inferieur).

Prediction (Rasmussen & Williams 2006, sect. 2.3) :

    mean(X*) = K(X*, X) (K + sigma_n² I)^{-1} y
    var(X*)  = K(X*, X*) + sigma_n² - K(X*, X) (K + sigma_n² I)^{-1} K(X, X*)

Leave-one-out (Rasmussen & Williams 2006, sect. 5.4.2), avec
`alpha = K^{-1} y` et `v_i = (K^{-1})_ii` :

    mu_loo_i  = y_i - alpha_i / v_i
    var_loo_i = 1 / v_i   (variance du processus latent, sans bruit)

## Tests

```bash
python -m pytest -q tests
```

Sortie obtenue :

```
........
8 passed, 1 warning in 1.72s
```

Couverture des tests :

- noyaux : symétrie `K(X1, X2) = K(X2, X1)^T`, diagonale = variance,
  Matérn 5/2 (et 3/2, SE) = sigma² a r = 0, valeur explicite de Matérn 5/2 ;
- `predict` interpole les points d'entrainement a bruit quasi nul
  (erreur < 1e-6) ;
- `loo` coincide avec n GP conditionnes sans le point i, hyperparametres
  fixes, sans re-optimisation ;
- comparaison avec `sklearn.gaussian_process.GaussianProcessRegressor`
  (RBF ARD + WhiteKernel, hyperparametres FIXES identiques, `optimizer=None`)
  : moyennes ET ecart-types a 1e-6 pres ;
- bout en bout sur Branin : RMSE normalisee < 0.1.

## Résultats obtenus (exemple Branin)

Lancement :

```bash
python examples/branin_demo.py
```

Sortie :

```
hyperparametres estimes:
  lengthscales = [188.8829 905.8663]
  variance     = 26716.1449
  bruit        = 0.00000016
log-vraisemblance marginale = 91.433
RMSE (200 points test) = 1.068
RMSE normalisee        = 0.0241
figure ecrite : .../examples/branin_demo.png
```

Le test de bout en bout (Branin, 30 points LHS, 200 points test) donne une
**RMSE normalisee ≈ 0.024**, bien sous le seuil de 0.1. La figure
`examples/branin_demo.png` montre les contours de la moyenne predite et les
points d'entrainement.

## Limitations

- Moyenne nulle uniquement (pas de base de fonctions / tendance).
- Pas de gradients de la vraisemblance (L-BFGS-B en mode "quasi-Newton",
  pas BFGS analytique) : la convergence est correcte mais moins rapide
  qu'avec des gradients.
- Noyaux stationnaires 1D/2D seulement (pas de co-krigeage multi-fidelite
  dans ce paquet ; c'est le role du paquet de reference).
