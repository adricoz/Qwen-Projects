
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
