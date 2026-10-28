# Projet : `gp_lite`, un petit paquet de régression par processus gaussien

Contexte : nous utilisons des processus gaussiens (GP) et du co-krigeage pour des surrogates aéro/hydro
(paquet de référence : `Multifidelity-GP-Clean/mfego/src/surrogate_models.py`, lisible via le tool `read_file`
avec le chemin `../../Multifidelity-GP-Clean/mfego/src/surrogate_models.py`). Tu dois écrire, dans le dossier
courant, un paquet Python autonome `gp_lite` (NumPy + SciPy seulement, pas de scikit-learn dans le paquet),
testé avec pytest.

## Contenu attendu

1. `gp_lite/kernels.py` : classes `SquaredExponentialARD(lengthscales, variance)`, `Matern32ARD`, `Matern52ARD`
   avec une méthode `K(X1, X2)` (matrices (n1, d), (n2, d) -> (n1, n2)) et une méthode `diag(X)`.
   Convention Matérn : r = sqrt(sum(((x - x')/l)^2)), Matérn 5/2 = σ²(1 + √5 r + 5 r²/3) exp(−√5 r).
2. `gp_lite/gp.py` : classe `GPRegressor(kernel, noise=1e-6, normalize_y=True)` avec
   - `fit(X, y)` : optimise log(lengthscales), log(variance), log(noise) par `scipy.optimize.minimize`
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

## Règles

- Vérifie chaque étape en exécutant le code (`run_python`). Les tests doivent passer : `python -m pytest -q tests`.
- Si un test échoue deux fois avec la même erreur, change d'approche plutôt que de réessayer.
- Pas de `pip install` : NumPy, SciPy, matplotlib, pytest et scikit-learn (pour les tests seulement) sont disponibles.
- Code en anglais avec docstrings NumPy ; commentaires et README en français.
- Termine avec l'outil `finish` : résumé de ce qui est fait, sortie de pytest, ce qui manque.
