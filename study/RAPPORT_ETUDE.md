# Comment prompter Qwen pour qu'un projet de programmation sorte juste

27 octobre 2026 · Qwen3.8-27B local (`qwen-science`, llama-server, 200k de contexte) piloté en agent par
`qwen_agent.py`, juge indépendant écrit par Claude.
Figures interactives : [`figures/etude_prompts.html`](figures/etude_prompts.html).

## En bref

- **Les 8 passes livrent un paquet utilisable, mais 3 seulement sont justes du premier coup.** Les 5 autres
  deviennent justes, 23/23, après une correction de **1 à 3 lignes** par Claude, guidée par les tests d'acceptance.
- **Aucune variante de prompt n'est fiable à elle seule.** Tests fournis (B), formules données (C) et méthode de
  vérification imposée (D) obtiennent chacune une passe parfaite et une ratée. L'énoncé seul (A) n'en obtient
  aucune. Avec 2 passes par variante, l'écart n'est pas significatif.
- **L'erreur la plus fréquente n'est pas une formule fausse, ce sont des bornes d'optimisation trop étroites.**
  5 passes sur 8 bornent le bruit ou la longueur de corrélation trop près de l'optimum. Leurs propres tests ne le
  voient pas.
- **Recette retenue : Claude écrit les tests d'acceptance, Qwen écrit le code, Claude corrige.** Les passes de Qwen
  ne coûtent que du temps de carte graphique, on peut donc en lancer deux et garder la meilleure.

## Protocole

- **Le projet** : le paquet `gp_lite`, déjà utilisé le 26 octobre (noyaux SE et Matérn ARD, `fit` par maximum de
  vraisemblance, prédiction, LOO, tests, exemple, README). L'énoncé de base ([`prompts/base.md`](prompts/base.md))
  reprend celui de la v5, avec en plus une interface imposée qui permet au juge de lire les hyperparamètres ajustés.
- **Les 4 variantes**, chacune ajoutant une section à l'énoncé de base :

| Variante | Ajout à l'énoncé |
|---|---|
| A. Énoncé seul | rien |
| B. Tests fournis | les 23 tests d'acceptance copiés dans le dossier ; le harnais les lance à chaque point de contrôle ([`prompts/B.md`](prompts/B.md)) |
| C. Formules données | NLL avec α = K⁻¹y, variance a posteriori, LOO (R&W 5.12), normalisation ([`prompts/C.md`](prompts/C.md)) |
| D. Vérification imposée | comparer chaque formule à une version naïve, comparer `fit` à scikit-learn optimisé sur des plans aléatoires, ne pas accuser l'environnement sans preuve ([`prompts/D.md`](prompts/D.md)) |

- **Les passes** : 2 par variante, dans l'ordre A1 B1 C1 D1 A2 B2 C2 D2. Chaque passe a droit à 90 étapes et
  60 min, avec une réflexion `medium`. Au total, 4 h de carte graphique.
- **Le juge** : [`../acceptance/test_acceptance.py`](../acceptance/test_acceptance.py), 23 tests écrits par Claude,
  toujours rejoués sur une copie intacte. Ils comparent :
  - les noyaux, la vraisemblance, la prédiction et le LOO à une référence indépendante (`slogdet` et `solve`, sans
    Cholesky) ;
  - `fit`, sur Branin avec 3 plans aléatoires et 2 noyaux, à l'optimum de scikit-learn : vraisemblance juste,
    optimum atteint à 2 nats près, NRMSE < 0,1.

  Le juge a d'abord été validé : une implémentation juste passe 23/23, la même avec le bug v5 (yᵀL⁻¹y) en rate 10.

## Résultats

| Passe | Tests (sur 23) | Arrêt | Étapes | Durée | Livrables | Défaut | Correction Claude |
|---|---|---|---|---|---|---|---|
| A1 | 13 | fini | 54 | 15 min | complets | log-déterminant compté deux fois ; borne du bruit ≥ e⁻¹² | 3 lignes → 23 |
| A2 | 17 | fini | 89 | 37 min | complets | bornes log (−6, 4) : bruit ≥ 2,5e-3 | 1 ligne → 23 |
| B1 | 17 | 90 étapes | 90 | 42 min | sans exemple ni README | bornes log (−7, 7) : bruit ≥ 9e-4 | 1 ligne → 23 |
| B2 | **23** | fini | 49 | 15 min | complets | — | — |
| C1 | **23** | fini | 48 | 15 min | complets | — | — |
| C2 | 17 | 90 étapes | 90 | 44 min | sans README | longueur ≤ 10, variance ≤ 100 | 3 lignes → 23 |
| D1 | **23** | fini | 87 | 35 min | complets | — | — |
| D2 | 14 | fini | 72 | 35 min | complets | Matérn 5/2 faux (5ρ²/3 avec ρ = √5 r) ; bornes variance et bruit | 3 lignes → 23 |

Les corrections sont dans [`fixes/`](fixes/), sous forme de diff. Les livraisons de Qwen ne sont pas modifiées.

### Ce qui marche

- **Quand la passe se passe bien, elle est rapide** : 15 min et environ 50 étapes pour B2 et C1, livrables
  complets et justes.
- **L'interface imposée est respectée** dans les 8 passes : aucun test n'échoue pour une question d'API.
- **Les tests fournis servent quand Qwen a le temps de les lire.** B2 a vu `test_fit_branin` échouer, a
  diagnostiqué « bornes trop restrictives » et les a retirées en 2 étapes.
- **La méthode imposée (D) a marché une fois sur deux.** D1 a trouvé seul sa borne de bruit trop haute et a
  démontré numériquement que l'écart-type de scikit-learn inclut le WhiteKernel.

### Ce qui ne marche pas

1. **Des bornes d'optimisation choisies sans regarder les données.** C'est le défaut le plus fréquent : 5 passes
   sur 8, dont deux cachées derrière une erreur de formule. Branin, déterministe, demande un bruit vers 1e-7 et une
   longueur de corrélation vers 20–30 sur x₂ ; des bornes de bruit à 1e-3 ou de longueur à 10 coûtent 15 à 20 nats
   de vraisemblance et une erreur 1,5 à 2,5 fois plus grande. Son propre test (NRMSE < 0,1 sur un LHS) laisse passer.
2. **L'algèbre linéaire, encore.** A1 compte le log-déterminant deux fois, l'erreur « facteur 2 » déjà vue au banc.
   Le terme quadratique, faux en v5, est juste dans les 8 passes : la consigne d'interface et les vérifications
   semblent avoir suffi.
3. **Les conventions du domaine.** D2 confond r et ρ = √5·r dans le Matérn 5/2, alors que l'énoncé donne la
   convention. Le noyau est faux avant comme après sa « correction » de l'étape 49, et le Cholesky échoue ensuite.
   C'est la même famille que le bug Matérn √3/√5 que Qwen ne voit jamais au banc d'essai.
4. **Une auto-vérification qui partage l'erreur.** La « référence naïve indépendante » de D2 contient la même
   formule de Matérn fausse : il l'a écrite lui-même avec la même compréhension, et ses 22 tests passent. La méthode
   D ne protège pas d'une erreur de compréhension.
5. **Le LOO est un gouffre à étapes.** B1 y passe environ 45 étapes et C2 environ 57, alors que C2 avait la
   formule. Ces deux passes atteignent la limite de 90 étapes sans finir les livrables.
6. **Les explications d'échec ne sont pas fiables.** B1 écrit « dans cet environnement `scipy.linalg.cho_factor`
   est cassé (facteur non triangulaire) ». C'est le comportement documenté de `cho_factor`, qui laisse des valeurs
   quelconques dans l'autre triangle. A2 conclut « aucune limitation fonctionnelle majeure » alors que son `fit` est
   bridé par ses bornes. Les rapports restent honnêtes sur ce qui manque (B1, C2).

### Ce que l'étude a corrigé dans l'outil

`run_python` lançait Python en mode isolé (`-I`), qui retire le dossier courant du chemin d'import. 7 passes sur 8
ont buté sur `No module named 'gp_lite'`, 2 à 5 étapes chacune, 19 au total, sans compter les contournements
(`conftest.py`, `sys.path.insert` dans chaque script). C'est corrigé dans `qwen_agent.py`, et un `import gp_lite`
passe maintenant directement.

Les résumés des 8 journaux sont dans [`summaries/`](summaries/). Qwen les a écrits : il a lu environ 535 000 tokens
de journaux et en a rendu environ 24 000. Les affirmations reprises ici ont été vérifiées dans les journaux ou le
code. Deux points étaient inexacts :

- pour D2, le résumé laisse croire à un noyau juste « cassé » à l'étape 49, alors qu'il était déjà faux avant ;
- les étapes perdues sur l'import sont surestimées : « 5 à 7 », contre 2 à 5 comptées dans les journaux.

## Recommandations

**Méthode de travail (coût minimal pour un résultat juste)**

1. Claude écrit les tests d'acceptance : une référence indépendante, plus une référence externe optimisée pour tout
   ce qui est ajusté. Ici, 192 lignes, réutilisées pour les 8 passes.
2. Qwen écrit le paquet en agent : 500 à 840 lignes de code et de tests, plus un README. Ça ne coûte que du temps
   de carte graphique.
3. Claude lance les tests d'acceptance, lit seulement les tests rouges et les lignes en cause, puis corrige.
   Ici, 1 à 3 lignes par passe ratée.
4. Si la carte graphique est libre, lancer deux passes et garder la meilleure. Avec 3 passes parfaites sur 8,
   on a environ 60 % de chances d'en obtenir au moins une juste, sans travail de Claude.

**Prompt**

- **Fournir les tests d'acceptance** (B) : Claude les écrit de toute façon.
- **Donner les formules et les conventions** (C), surtout celles où Qwen se trompe : Matérn, log-déterminant, LOO.
- **Imposer les choix numériques au lieu de les laisser deviner** : bornes larges (log bruit ≥ −23,
  longueurs et variance jusqu'à 1e3 à 1e4), départs des redémarrages.
- Garder la consigne « ne pas accuser l'environnement sans exemple minimal » (D), sans compter sur elle.
- Laisser 90 étapes à 60 min. Les passes ratées sont aussi les plus longues : au-delà, il tourne en rond.

**Prochain essai utile** : une variante B + C + bornes imposées, 2 passes, avec le harnais corrigé, pour savoir si
la combinaison rend la première passe juste de façon fiable.

## Relancer

```
cd C:\Users\SIM\Desktop\Code-Adri\Qwen-Projects\study
python run_study.py            # refait les passes absentes de results/
python run_study.py C1 C2      # seulement ces passes
python make_plots.py           # figures/etude_prompts.html
```

`runs/<passe>/` contient le paquet livré, `agent_run.log` et `.qwen_agent/log.md`, le journal complet.
`results/<passe>.json` contient les scores test par test et le rapport final de l'agent.
