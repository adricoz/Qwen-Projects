# Qwen3.8-27B en agent de programmation : premier essai

26 octobre 2026 · modèle local `qwen-science` (Qwen3.8-27B Q4_K_M) servi par llama.cpp (200k de contexte, outils,
images), piloté par `C:\Users\SIM\.claude\tools\qwen_agent.py`.

## Le projet

Énoncé : [`gp_lite_v5/TASK.md`](gp_lite_v5/TASK.md). Écrire un petit paquet de régression par processus gaussien
(noyaux SE / Matérn 3/2 / Matérn 5/2 ARD, ajustement par maximum de vraisemblance avec redémarrages, prédiction,
leave-one-out en forme fermée), sa suite pytest (comparaison à scikit-learn, LOO contre n ajustements, Branin),
un exemple avec figure et un README. Environ 750 lignes au total. L'agent pouvait lire le code de référence de
`Multifidelity-GP-Clean`, écrire uniquement dans son dossier et exécuter Python.

## Résultat

**La 5e passe livre un projet complet : tous les livrables, 8 tests qui passent, rapport final honnête.**
Elle contient pourtant un bug mathématique que ses propres tests ne voient pas.

| Passe | Ce qui a changé | Résultat |
|---|---|---|
| 1 | harnais initial | 80 étapes perdues à déboguer la formule LOO, 9 fichiers scratch, ni tests ni README |
| 2 | contrôle de progression, mode finalisation, rapport forcé | tests écrits tôt, rapport honnête, mais fichiers tronqués et `python -c` muet |
| 3 | écritures tronquées jamais exécutées, `python -c` réparé | plantage du harnais sur une apostrophe non fermée (étape 10) |
| 4 | toute erreur d'outil renvoyée au modèle | écritures toujours tronquées à ~1 400 caractères |
| 5 | **sampler DRY retiré** | **projet complet, 8/8 tests, `finish` appelé seul** |

Causes des échecs 2 et 4 : le sampler anti-répétition DRY. Rejoué sur le contexte exact d'une écriture ratée,
il coupe le fichier à 1 460 caractères, contre 4 710 sans DRY. Il pénalise toute séquence déjà présente dans le
contexte, donc tout code qui ressemble à du code déjà lu. Il est retiré du harnais et du serveur.

Durée de la 5e passe : 81 min, dont 52 min de blocage. À 15h42, Ollama a chargé une seconde copie du modèle
(requête de chat de 27 min, probablement Continue via l'ancienne entrée Ollama) et llama-server est tombé à
1,5 tok/s. Hors incident : environ 26 min pour 73 étapes. Les deux outils signalent désormais ce conflit.

## Vérification indépendante (Claude)

| Contrôle | Résultat |
|---|---|
| Noyaux contre mes formules | exacts (1e-15) |
| Moyenne a posteriori contre scikit-learn | exacte (1e-15) |
| LOO contre n conditionnements | exact (2e-15) |
| Écart-type | inclut la variance du bruit (scikit-learn sans `WhiteKernel` donne la version latente) : convention, non documentée |
| `fit` sur Branin, plan aléatoire, 30 points | **faux** : log-vraisemblance 66 542 (scikit-learn : 15,7), longueur de corrélation 1 540 sur x₂, NRMSE 0,13 (SE) et 0,18 (Matérn 5/2) contre 0,010 et 0,028 pour scikit-learn |

**Bug :** `gp.py` ligne 101, le terme quadratique vaut `y @ alpha` avec `alpha = L⁻¹ y`, soit yᵀL⁻¹y au lieu de
yᵀK⁻¹y. C'est **la même erreur** que Qwen avait faite dans le banc d'essai du 19 octobre (question C8). Corrigé en
`alpha @ alpha` sur une copie, l'ajustement redevient correct (NRMSE 0,012 à 0,053 sur trois plans, au niveau de
scikit-learn) et ses 8 tests passent toujours. La livraison de l'agent n'est pas modifiée.

Ses tests ne l'ont pas vu car ils comparent à scikit-learn à hyperparamètres fixés et n'appellent `fit` que sur un
plan LHS favorable. Dans son rapport, l'agent attribue aussi une incohérence passagère à un « non-déterminisme
BLAS » : rien ne l'étaye, c'était très probablement un état partagé dans son propre code.

## Ce qu'on en retient

- **Qwen sait mener un projet de taille moyenne jusqu'au bout** quand le harnais le cadre : plan, tests tôt,
  correction sur erreur, finalisation, rapport.
- **Un projet « vert » n'est pas un projet juste.** Les erreurs d'algèbre linéaire du banc d'essai se retrouvent
  à l'identique, et ses propres tests ne les détectent pas. Il faut des tests d'acceptation écrits par quelqu'un
  d'autre : comparer `fit` à une référence optimisée, pas seulement à hyperparamètres fixés.
- **Ses explications d'un échec peuvent être inventées** (« bug d'environnement »).

## Relancer

```
cd C:\Users\SIM\Desktop\Code-Adri\Qwen-Projects\<dossier>
python C:/Users/SIM/.claude/tools/qwen_agent.py @TASK.md --workdir . --read-root C:/Users/SIM/Desktop/Code-Adri ^
  --effort medium --max-steps 80 --max-minutes 90 ^
  --deliverables gp_lite/kernels.py,gp_lite/gp.py,tests/test_gp_lite.py,examples/branin_demo.py,README.md ^
  --test-cmd "-m pytest -q tests"
```

Journal de chaque passe : `<dossier>/.qwen_agent/log.md` (pensée, appels d'outils, sorties) et `agent_run.log`.
Les dossiers `gp_lite` (passe 1) à `gp_lite_v4` sont conservés pour mémoire.
