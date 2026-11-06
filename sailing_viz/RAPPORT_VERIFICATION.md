# Visualiseur 3D du vocabulaire maritime — rapport de réalisation et de vérification

Date : 6 novembre 2026. Demande : un visualisateur 3D HTML d'un voilier qui bouge avec des sliders, pour apprendre
le vocabulaire (heel, yaw, cant, rake, leeway, noms des voiles…), réalisé avec Qwen et vérifié par Claude.

## Ce qui a été livré

| Fichier | Rôle |
|---|---|
| `index.html` | la page (Three.js r128 + GLTFLoader + OrbitControls en CDN, glossaire JSON intégré) |
| `sailing_viz_standalone.html` | même page avec le modèle 3D inclus en base64 (9,4 Mo) : s'ouvre par double-clic |
| `assets/sailing-yacht-8-m.glb` | modèle « Sailing yacht, 8 m » d'Innerscene, licence CC0 (`assets/LICENSE.txt`) |
| `glossary.json` | les 66 termes en 5 groupes, définitions en français (écrit par Claude, copié tel quel dans la page) |
| `serve.bat`, `build_standalone.py` | serveur local pour `index.html` ; construction de la version standalone |
| `tests/test_structure.py` | tests de Qwen (10) ; tests cachés de Claude dans `../sailing_viz.accept/` (12) |
| `check_visual.py` | contrôle visuel et numérique avec Playwright + Edge (optionnel, voir en tête du script) |

Le modèle est un croiseur à quille : les foils en C, le plan porteur de safran, le bout-dehors et les trois voiles
de portant (gennaker, spi, code 0) sont des primitives Three.js ajoutées. Aucun voilier à foils sous licence libre
n'a été trouvé en glTF.

## Conventions (repère bateau)

`+x` avant, `+y` haut, `+z` tribord. Gîte positive = mât vers tribord ; assiette positive = étrave haute ; lacet positif
= étrave vers tribord ; dérive positive = route à tribord du cap ; TWA positif = vent de tribord (tribord amures),
voiles et bôme sous le vent ; cant positif = extrémités des foils vers l'extérieur ; rake positif = bord d'attaque vers
le haut ; safran positif = bord de fuite vers tribord. Vent apparent : `AW = TW − V`, `VMG = V·cos(TWA)`.

## Déroulement

1. Claude : recherche d'un modèle 3D libre (Innerscene CC0, 170 mailles nommées : coque, pont, quille, bulbe, safran,
   mât, bôme, grand-voile, foc, haubans…), glossaire, énoncé `TASK.md`, 12 tests cachés.
2. Qwen (llama-server, 36 min, 40 étapes) : page, README, tests, script de build. Arrêté par Claude à l'étape 40 alors
   qu'il testait sa fonction de vent apparent : 11 tests cachés sur 12 passaient déjà.
3. Claude : exécution des tests cachés, capture d'écran et mesures dans un navigateur piloté (Playwright + Edge,
   rendu logiciel), lecture ciblée du code, 9 corrections, reconstruction du standalone.

## Défauts trouvés et corrigés par Claude

| # | Défaut | Effet visible | Correction |
|---|---|---|---|
| 1 | `computeApparentWind` arrondissait à 0,1 dans la fonction pure | test caché en échec | arrondi déplacé à l'affichage |
| 2 | re-parentage des mailles pendant `scene.traverse` | erreur « reading 'traverse' », voiles non attachées | collecte d'abord, re-parentage ensuite |
| 3 | flèches cherchées par `o.isArrowHelper` (inexistant en r128) | erreur « reading 'position' », scène bloquée | registre direct des 4 flèches |
| 4 | GLTFLoader remplace les espaces des noms par `_` | bôme, safran, mât jamais reconnus : ne bougeaient pas, surlignage muet | noms restaurés au chargement |
| 5 | signe de la gîte et de l'assiette (indication erronée de l'énoncé de Claude) | gîte à bâbord pour une valeur positive | `rotation.x = +heel`, `rotation.z = +trim` |
| 6 | signe du safran | bord de fuite à bâbord pour un angle positif | `rotation.y = +rudder` |
| 7 | plan porteur de safran décalé deux fois de −3,85 m | plaque flottante 4 m derrière le bateau | position locale (0, 0.33, 0) |
| 8 | voiles de portant en coordonnées absolues dans un groupe au point d'amure | spi/gennaker 5 m devant et au-dessus du mât | coordonnées relatives au point d'amure |
| 9 | extension des foils dans le mauvais sens ; voiles de portant jamais orientées | foils descendus de 2,4 m à extension 0 ; spi vu par la tranche | signe inversé ; rotation sous le vent comme le foc |

Vérifications finales (`check_visual.py`) : foils relevés à extension 0 (y du plan porteur −0,61 m contre −2,45 m
déployé), elevator en (−3,85 ; −1,17 ; 0), mât à z = +2,6 m pour 25° de gîte, bôme à z = −0,55 m pour TWA +60 et
+0,55 m pour TWA −60, safran et gennaker pivotants, glossaire cliquable, aucune erreur console. Tests : 12/12 cachés,
10/10 de Qwen.

## Leçon pour la délégation

Une scène 3D ne peut pas être validée par Qwen (pas de navigateur dans le harnais) ni par des tests de structure :
sur 9 défauts, 7 étaient géométriques (signes, repères, noms de nœuds) et invisibles aux tests. Le contrôle Playwright
(mesures de positions monde + captures) est indispensable et coûte 10 min. Voir `Qwen-Benchmark/REGLE_DELEGATION.md`.
