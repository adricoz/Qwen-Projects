# Site VPP acceleration — lancer et suivre les optimisations de section

Petit site local (serveur Python standard, bibliothèques : numpy, scipy, plotly) pour :

- **lancer** une optimisation de section du C-foil (`vpp_acceleration/scripts/run_real_vpp.py`) en choisissant
  la zone de recherche, la distribution d'exploitation et la section de départ ;
- **suivre** le run en direct et l'**arrêter** ;
- **relire** les runs terminés : gain validé sur la section de départ, évolution, section, vitesses.

Il ne modifie jamais un run existant. Chaque run lancé est un dossier normal de
`vpp_acceleration/runs/<date>_<nom>/`.

## Démarrage

```
cd Qwen-Projects\site
..\..\Multifidelity-GP-Clean\.venv-mds\Scripts\python.exe app.py          # http://127.0.0.1:8321/
..\..\Multifidelity-GP-Clean\.venv-mds\Scripts\python.exe app.py --port 8400
```

Le site tourne aussi sous un autre Python 3.10+ qui a numpy, scipy et plotly. Les **runs**, eux, sont toujours lancés
avec l'interpréteur du projet (`.venv-mds`, qui a scikit-learn et numba) ; le chemin est affiché au démarrage.

Redémarrer ou quitter le site (Ctrl-C) **ne tue pas** les runs en cours : chaque run lancé est décrit dans
`logs/<clé>.json`, relu au démarrage, donc la page de suivi et le bouton d'arrêt sont retrouvés.

## Ce que le formulaire envoie vraiment au script

Chaque option du formulaire correspond à un flag réel de `run_real_vpp.py` (vérifié par les tests, qui lisent le
code du script).

**Contraintes fixées** (affichées verrouillées dans le formulaire) :

| Affichage | Ce que le script fait |
|---|---|
| Ixx / Ixx(section actuelle) ≥ 1 | `--box current` : toute section plus souple que la section actuelle est écartée (vérifié sur les sections décodées). Ixx actuelle = 6,717·10⁻⁵ (corde 1) |
| Boîte de recherche | `--box current` : largeurs du notebook Dim-Reduction × l'échelle, centrées sur la section actuelle |
| Géométrie | surfaces sans croisement, confiance NeuralFoil, bord de fuite fixé à 0,00197 c ; 16 poids de Kulfan libres |

La boîte « dimred » (premier run du 22/10) **n'est pas proposée** : elle n'a aucune contrainte d'Ixx, et toutes ses
sections sont plus souples que la section actuelle (Ixx/Ixx(actuelle) de 0,15 à 0,89). La page `/bounds/dimred/1.0`
la montre pour comparer.

**Réglages** (deux préréglages remplissent les champs, tout reste modifiable) :

| Préréglage | Valeurs |
|---|---|
| **Production** (défaut) | réglages des runs 7-8 : 140 min, échelle 3 (« ne pas restreindre la section », 29/10), plan de départ 256/64, lots de 16, 16 + 16 processus (la machine est partagée), recherche locale 0,5, convergence 10 × 0,01 nd, 2 redémarrages, `profile_v2` |
| **Test rapide** | 10 min, échelle 1, plan 120/30, lots de 4, 8 + 8 processus, **5 conditions** (pas de profil) : avec `profile_v2` (49 conditions) une seule mise à jour du modèle dure déjà plus de 10 min |

| Champ | Flag | Remarque |
|---|---|---|
| Distribution d'exploitation | `--search-profile` et `--profile` | défaut `profile_v2` ; « uniforme » = 5 conditions équipondérées |
| Section de départ | `--start-section` | meilleure section d'un run précédent. Hors boîte ou trop souple : la section admissible la plus proche, écrite dans `run.log`. Défaut : la section actuelle du C-foil |
| Comparer à la fin avec | `--compare` | sections de runs précédents validées en même temps que la nouvelle. **Ce n'est pas un départ** |
| Foil flexible | `--fsi` | coché par défaut ; ce n'est pas une contrainte (la contrainte d'Ixx vient de la boîte) |
| Avancé | `--k`, `--n-low`, `--n-high`, `--workers`, `--vpp-workers`, `--batch`, `--seed`, `--hf`, `--local-fraction`, `--converge`, `--restarts`, `--min-gain`, `--sensitivity-median`, `--smoke` | `--restarts` n'est émis que si la convergence est réglée (le script l'exige) |

`--start-section` est une option **ajoutée au script le 30/10/2026** (`scripts/run_real_vpp.py`) ; sans elle, le
script ne savait partir que de la section actuelle.

Les valeurs sont bornées (par exemple 32 processus au plus) ; un champ illisible est ignoré et le script garde son défaut.

## Pages

| URL | Contenu |
|---|---|
| `/` | runs en cours, formulaire de lancement, runs terminés avec leur gain validé |
| `/live/<clé>` | suivi d'un run lancé ici : état, journal, **⏹ Arrêter le run** (confirmation demandée) |
| `/run/<nom>` | run terminé : gain validé, évolution du meilleur candidat, section, vitesses par condition, figures complètes |
| `/profile/<nom>` | une distribution d'exploitation (TWA × TWS, taille = poids) |
| `/bounds/<current\|dimred>/<échelle>` | la zone de recherche réelle (voir ci-dessous) |

### Zone de recherche

La page construit **la même famille que le run** (mêmes fonctions : boîte centrée sur la section actuelle,
Ixx ≥ Ixx(actuelle), confiance NeuralFoil, absence de croisement) et montre :

- la part des tirages de la boîte qui passent chaque contrainte (échelle 1 : 54 % pour Ixx, 62 % pour la
  confiance NeuralFoil, 37,5 % pour toutes ; échelle 3 : 94 % sans croisement, 30 % pour toutes) ;
- l'enveloppe de 400 sections admissibles autour de la section actuelle ;
- les bornes des 16 poids de Kulfan, l'étendue des sections admissibles et la projection sur deux modes de forme.

Le calcul dure 4 à 5 s par échelle (puis en cache).

### Gain d'un run

Le gain affiché est **celui du run lui-même** : vitesse validée de la meilleure section − vitesse validée de la
section de départ, corrigée de l'écart du surrogate au code de référence, moyennée avec les poids de la distribution
d'exploitation sur les conditions résolues pour toutes les sections. C'est le chiffre de `summary.txt` (vérifié par un
test sur le run 8 : 13,566 contre 13,310 nd, **+0,256 nd**).

Le site donne ensuite le **temps gagné sur une route** (1 000 nm par défaut) par une règle de trois à vitesse
moyenne constante, et, si l'on saisit une valeur du temps en €/h, un montant. Ces deux conversions sont des
estimations indicatives. Elles remplacent l'ancien « gain financier » qui supposait un moteur diesel (60 L/h) et
comparait la vitesse prédite du premier candidat à celle du dernier, ce qui donnait un gain nul pour le run 8.

La courbe « évolution du meilleur candidat » montre la vitesse **prédite par le surrogate**, pas la vitesse validée.

## Garde-fous

- Un second run n'est lancé qu'après confirmation explicite si un run tourne déjà sur la machine (lancé d'ici ou à la
  main) : machine partagée, deux runs se ralentissent et faussent leur durée limite.
- Le site n'écoute que sur `127.0.0.1`. Les fichiers servis sous `/run/<nom>/static/` restent dans le dossier du run.
- Le JavaScript de plotly (4,8 Mo) est servi une fois (`/static/plotly.min.js`, mis en cache) : une page de run pèse
  100 ko au lieu de 19 Mo.

## Fichiers

| Fichier | Rôle |
|---|---|
| `app.py` | pages HTML et serveur |
| `launcher.py` | commande de lancement, suivi, arrêt, état persistant (`logs/`) |
| `data.py` | lecture des runs, gain validé, famille de sections et zone de recherche |
| `figures.py` | figures plotly |
| `tests/test_site.py` | 25 tests : `..\..\Multifidelity-GP-Clean\.venv-mds\Scripts\python.exe -m pytest -q tests` |
| `logs/` | une paire `<clé>.json` (état) et `<clé>.out` (console) par run lancé d'ici |

## Limites connues

- Le temps gagné suppose une vitesse moyenne constante sur la route ; ce n'est pas un calcul de routage.
- Un run qui n'a pas été lancé depuis ce site apparaît dans la liste des runs terminés, mais le site ne peut pas
  l'arrêter.
- La page d'un run en cours n'a pas de gain validé : il n'existe qu'à la fin du run (`validation.csv`).
