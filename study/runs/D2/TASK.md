# Projet : `gp_lite`, un petit paquet de régression par processus gaussien

Contexte : nous utilisons des processus gaussiens (GP) et du co-krigeage pour des surrogates aéro/hydro
(paquet de référence, lisible via le tool `read_file` :
`C:/Users/SIM/Desktop/Code-Adri/Multifidelity-GP-Clean/mfego/src/surrogate_models.py`). Tu dois écrire, dans le
dossier courant, un paquet Python autonome `gp_lite` (NumPy + SciPy seulement, pas de scikit-learn dans le paquet),
testé avec pytest.

## Contenu attendu

1. `gp_lite/kernels.py` : classes `SquaredExponentialARD(lengthscales, variance)`, `Matern32ARD`, `Matern52ARD`
   avec une méthode `K(X1, X2)` (matrices (n1, d), (n2, d) -> (n1, n2)) et une méthode `diag(X)`.
   Convention Matérn : r = sqrt(sum(((x - x')/l)^2)), Matérn 5/2 = σ²(1 + √5 r + 5 r²/3) exp(−√5 r).
2. `gp_lite/gp.py` : classe `GPRegressor(kernel, noise=1e-6, normalize_y=True)` avec
   - `fit(X, y, optimize=True)` : optimise log(lengthscales), log(variance), log(noise) par `scipy.optimize.minimize`
     (L-BFGS-B) sur la log-vraisemblance négative calculée par Cholesky (avec un jitter si la factorisation échoue),
     3 redémarrages aléatoires (graine fixée) et conservation du meilleur ;
   - `predict(X, return_std=False)` : moyenne et écart-type a posteriori ;
   - `log_marginal_likelihood()` ;
   - `loo(self)` : résidus leave-one-out en forme fermée (formules de Rasmussen & Williams, section 5.4.2),
     sans refaire n ajustements.
3. `tests/test_gp_lite.py` (pytest) :
   - noyaux : symétrie, diagonale = variance, Matérn 5/2 à r = 0 vaut σ² ;
   - `predict` interpole les points d'apprentissage à bruit nul (erreur < 1e-6) ;
   - `loo` coïncide avec le calcul explicite par n ajustements sans ré-optimisation (hyperparamètres fixés), tol 1e-6 ;
   - comparaison avec scikit-learn (`sklearn.gaussian_process.GaussianProcessRegressor`, noyau RBF ARD + WhiteKernel,
     hyperparamètres FIXÉS identiques, `optimizer=None`) : moyennes prédites égales à 1e-6 près sur une fonction test 2D ;
   - un test de bout en bout : fonction de Branin (2D), 30 points LHS (écris un LHS simple avec `numpy`), 200 points
     test : RMSE normalisée < 0.1 après `fit`.
4. `examples/branin_demo.py` : entraîne le GP sur Branin et écrit `examples/branin_demo.png` (matplotlib, backend Agg)
   avec la moyenne prédite et les points d'apprentissage.
5. `README.md` en français : installation (aucune), API, formules utilisées, comment lancer les tests, résultats obtenus
   (copie la sortie de pytest et la RMSE).

## Interface imposée (d'autres programmes l'utiliseront)

- Noyaux : constructeur `Classe(lengthscales, variance=1.0)` ; attributs publics `lengthscales` (ndarray (d,)) et
  `variance` (float).
- `fit(X, y, optimize=True)` renvoie `self`. Avec `optimize=False`, il garde les hyperparamètres actuels de
  `self.kernel` et `self.noise` et ne fait que conditionner sur les données.
- Après `fit`, `self.kernel` porte les hyperparamètres ajustés et `self.noise` la variance du bruit ajustée.
- `normalize_y=True` : y_n = (y − moyenne) / écart-type (`np.std`, ddof = 0). La vraisemblance et `noise` sont
  exprimés sur y_n ; les prédictions sont renvoyées dans les unités de y.
- `predict(X, return_std=True)` renvoie `(mean, std)`, deux arrays (m,) ; `std` est l'écart-type de f (sans le bruit).
- `log_marginal_likelihood()` : log-vraisemblance marginale (sur y_n si normalisé) aux hyperparamètres courants.
- `loo()` renvoie `(mean, var)`, arrays (n,) : moyenne et variance prédictives leave-one-out de y_i (bruit inclus),
  dans les unités de y, la normalisation étant celle de tout le jeu d'apprentissage.

## Règles

- Vérifie chaque étape en exécutant le code (`run_python`). Les tests doivent passer : `python -m pytest -q tests`.
- Si un test échoue deux fois avec la même erreur, change d'approche plutôt que de réessayer.
- Pas de `pip install` : NumPy, SciPy, matplotlib, pytest et scikit-learn (pour les tests seulement) sont disponibles.
- Code en anglais avec docstrings NumPy ; commentaires et README en français.
- Termine avec l'outil `finish` : résumé de ce qui est fait, sortie de pytest, ce qui manque.

## Méthode de vérification obligatoire

Un paquet dont les tests passent n'est pas forcément juste : des tests qui réutilisent le code testé, ou qui
ne testent qu'à hyperparamètres fixés, laissent passer les erreurs de formule. Applique donc cette méthode :

1. Pour chaque quantité numérique (noyaux, log-vraisemblance, moyenne et variance a posteriori, LOO), écris dans
   `tests/` une version de référence **naïve et indépendante** (`np.linalg.solve`, `np.linalg.slogdet`,
   `np.linalg.inv`, boucles), sans réutiliser le code du paquet, et compare-la au paquet à hyperparamètres fixés,
   pour plusieurs jeux d'hyperparamètres, avec et sans `normalize_y`.
2. Teste aussi `fit` lui-même : sur Branin avec 3 plans **aléatoires** (pas seulement un LHS),
   - vérifie que `log_marginal_likelihood()` du paquet égale ta référence naïve aux hyperparamètres ajustés ;
   - compare la log-vraisemblance atteinte à celle de scikit-learn **optimisé** (optimiseur et redémarrages
     activés, même modèle : noyau × constante + WhiteKernel, `normalize_y=True`), toutes deux calculées avec ta
     référence naïve : la tienne ne doit pas être nettement plus basse.
3. Si une vérification échoue, la cause est dans ton code : cherche l'erreur mathématique. N'invoque pas
   l'environnement (BLAS, versions, non-déterminisme) sans l'avoir prouvé par un exemple minimal.
4. N'appelle `finish` que lorsque toutes ces vérifications passent ; sinon, liste précisément ce qui échoue.
