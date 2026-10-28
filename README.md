# Qwen-Projects — évaluer un modèle Qwen local comme agent de programmation

Ce dépôt regroupe les essais et l'étude consacrés à l'utilisation d'un modèle local
**Qwen3.8-27B** (`qwen-science`, servi par llama.cpp, 200k de contexte, outils et images)
piloté en agent par un harnais (`qwen_agent.py`) pour écrire du code Python de taille moyenne
jusqu'au bout : paquet, tests, exemple, README.

L'objectif n'est pas le code livré lui-même, mais la **mesure de ce que le modèle sait faire en
autonome**, et de ce qu'il faut lui donner (tests, formules, méthode de vérification) pour que la
livraison soit *juste* — pas seulement « verte ». Un juge indépendant (tests d'acceptance écrits par
Claude, jamais montrés à l'agent sauf consigne contraire) arbitre chaque passe.

## Le projet d'évaluation : `gp_lite`

Toutes les passes reprennent le même énoncé ([`gp_lite_v5/TASK.md`](gp_lite_v5/TASK.md)) : écrire un
petit paquet autonome de régression par processus gaussien (NumPy + SciPy seulement, sans
scikit-learn) :

- noyaux stationnaires ARD — SquaredExponential, Matérn 3/2, Matérn 5/2 ;
- `fit` par maximum de vraisemblance marginale (L-BFGS-B, redémarrages, jitter) ;
- prédiction a posteriori ;
- résidus **leave-one-out en forme fermée** (Rasmussen & Williams, § 5.4.2) ;
- une suite pytest, un exemple sur la fonction de Branin avec figure, et un README.

Ce problème est volontairement sensible à l'algèbre linéaire (log-déterminant, terme quadratique,
LOO) et aux bornes d'optimisation : il distingue un paquet qui *se teste* d'un paquet qui est
*juste*. La référence de lecture pour l'agent est le dépôt `Multifidelity-GP-Clean`.

## Arborescence

```
├── acceptance/                    # tests d'acceptance écrits par Claude (23 tests) — le juge
│   └── test_acceptance.py
├── gp_lite/ … gp_lite_v5/         # les 5 passes successives du 1er essai (v1 à v5)
│   ├── gp_lite/                   #    chaque passe : paquet, tests, exemple, README, journal
│   ├── gp_lite_v2/ … gp_lite_v5/
├── study/                         # étude de prompts : 4 variantes (A-D) × 2 passes
│   ├── run_study.py               #    orchestration (serving, passes, scoring, résultats)
│   ├── make_plots.py              #    figures interactives
│   ├── prompts/                   #    énoncés de base + ajout de chaque variante
│   ├── runs/                      #    livraisons de Qwen, journaux (.qwen_agent/, agent_run.log)
│   ├── results/                   #    scores test par test + rapport final (JSON)
│   ├── fixes/                     #    corrections Claude (diff), les livraisons ne sont pas modifiées
│   ├── summaries/                 #    résumés des 8 journaux
│   └── RAPPORT_ETUDE.md           #    <— le résultat de l'étude
├── Bimodal-Sailing-Discretizer/   # tâche ponctuelle : éditeur web de profil de fonctionnement
├── qwen_task_stats/               # tâche ponctuelle : statistiques long terme des tâches Qwen
├── RAPPORT_AGENT.md               # <— compte-rendu du 1er essai (gp_lite, 5 passes)
└── README.md
```

## Deux temps forts

### 1. Premier essai — [`RAPPORT_AGENT.md`](RAPPORT_AGENT.md) (26 octobre)

Cinq passes successives, chaque harnais modifié en réaction aux échecs de la précédente. La 5e passe
livre un projet complet (8/8 tests, rapport honnête) — mais avec un **bug mathématique** que ses
propres tests ne voient pas : le terme quadratique de la log-vraisemblance vaut `yᵀL⁻¹y` au lieu de
`yᵀK⁻¹y`. Les dossiers `gp_lite/` à `gp_lite_v4/` sont conservés pour mémoire.

> À retenir : *un projet « vert » n'est pas un projet juste*. Ses explications d'échec peuvent être
> inventées (« bug d'environnement »).

### 2. Étude de prompts — [`study/RAPPORT_ETUDE.md`](study/RAPPORT_ETUDE.md) (27 octobre)

Quatre variantes d'énoncé, 2 passes chacune (8 passes, ~4 h de carte graphique) :

| Variante | Ajout à l'énoncé |
|---|---|
| A | l'énoncé seul |
| B | + les tests d'acceptance fournis |
| C | + les formules et conventions données |
| D | + une méthode de vérification imposée |

**Résultat :** les 8 passes livrent un paquet utilisable, mais **3 seulement sont justes du premier
coup** ; les 5 autres deviennent justes après 1 à 3 lignes corrigées par Claude, guidées par les tests
d'acceptance. Aucune variante n'est fiable à elle seule. Le défaut le plus fréquent n'est pas une
formule fausse : ce sont des **bornes d'optimisation trop étroites**, que les tests de l'agent ne
détectent pas.

> Recette retenue : **Claude écrit les tests d'acceptance, Qwen écrit le code, Claude corrige.** Les
> passes de Qwen ne coûtent que du temps de carte graphique — on peut en lancer deux et garder la
> meilleure.

## Relancer

> Chemins propres à cette machine (Windows, `SIM`) ; adapter `qwen_agent.py`, le dossier de lecture et
> le serveur llama.cpp avant d'exécuter.

**Premier essai**

```
cd C:\Users\SIM\Desktop\Code-Adri\Qwen-Projects\<dossier>
python C:/Users/SIM/.claude/tools/qwen_agent.py @TASK.md --workdir . ^
  --read-root C:/Users/SIM/Desktop/Code-Adri --effort medium --max-steps 80 --max-minutes 90 ^
  --deliverables gp_lite/kernels.py,gp_lite/gp.py,tests/test_gp_lite.py,examples/branin_demo.py,README.md ^
  --test-cmd "-m pytest -q tests"
```

**Étude de prompts**

```
cd C:\Users\SIM\Desktop\Code-Adri\Qwen-Projects\study
python run_study.py            # refait les passes absentes de results/
python run_study.py C1 C2      # seulement ces passes
python make_plots.py           # figures/etude_prompts.html
```

**Le juge** (toujours sur une copie intacte)

```
python -m pytest -q C:/Users/SIM/Desktop/Code-Adri/Qwen-Projects/acceptance/test_acceptance.py
# ou, en pointant GP_LITE_DIR sur la livraison à juger :
set GP_LITE_DIR=C:\...\Qwen-Projects\gp_lite_v5 && python -m pytest -q ..\acceptance\test_acceptance.py
```

## Fichiers de référence

- [`RAPPORT_AGENT.md`](RAPPORT_AGENT.md) — compte-rendu détaillé du 1er essai (passages, causes des
  échecs, vérification indépendante, bugs).
- [`study/RAPPORT_ETUDE.md`](study/RAPPORT_ETUDE.md) — protocole, tableau des 8 passes, ce qui marche /
  ce qui ne marche pas, recommandations.
- [`gp_lite_v5/README.md`](gp_lite_v5/README.md) — documentation du paquet livré (la meilleure passe).
- [`acceptance/test_acceptance.py`](acceptance/test_acceptance.py) — la suite de 23 tests du juge.
- Chaque passe conserve son journal : `<dossier>/.qwen_agent/log.md` (pensée, appels d'outils, sorties)
  et `agent_run.log`.
