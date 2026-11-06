# Visualiseur 3D — vocabulaire de la voile (English / français)

Page web interactive qui charge un voilier 8 m (Three.js r128) et fait bouger
coque, voiles et foils en temps réel pour apprendre les termes maritimes anglais
(heel, trim, yaw, cant, rake, leeway, TWA, sail names...).

## Lancer
- Double-clic sur `serve.bat`, puis ouvrir `http://localhost:8765/`
  (le navigateur ne peut pas charger le GLB en `file://`).
- Ou générer le fichier unique : `python build_standalone.py` puis
  double-clic sur `sailing_viz_standalone.html` (modèle inline en base64).

## Commandes (colonne de droite)
- **Heel (gîte)** : inclinaison latérale, + = mât vers tribord.
- **Trim / pitch (assiette)** : + = étrave vers le haut.
- **Yaw / heading (lacet, cap)** : rotation autour de la verticale.
- **Leeway (dérive)** : dérive latérale (flèche d'orange vs verte).
- **Cant / Rake (foil)** : orientation des foils (cant = extrémités vers le haut/hors-bord ;
  rake = arête d'attaque vers le haut).
- **Foil extension** : foils remontés (0) à déployés (1).
- **Ride height (hauteur de vol)** : hauteur du bateau au-dessus de l'eau.
- **Rudder (safran)** : angle du safran (+ = bord de fuite tribord).
- **TWA / TWS** : angle et vitesse du vent réel ; **Boat speed** : vitesse du bateau.
  L'afficheur (AWA, AWS, VMG, cap, route) se met à jour en direct.
- **Sail config** : grand-voile + foc / génoa / gennaker / spinnaker / code 0.
- **Caméra** : 3/4 avant, profil, dessus, arrière, proue ; **Reset** remet les valeurs par défaut.

## Panneau glossaire
5 groupes, chaque terme « English (français) » ; clic = définition + surlignage
des pièces concernées pendant 2 s. Recherche insensible aux accents.

## Crédit modèle
« Sailing yacht, 8 m » — Innerscene Corp (innerscene.com), licence CC0 1.0, voir `assets/LICENSE.txt`.
Les foils en C, le plan porteur de safran, le bout-dehors et les voiles de portant sont des primitives ajoutées (le modèle est un croiseur à quille).
