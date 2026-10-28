
## Tests d'acceptation (fournis par l'équipe)

Le dossier `tests_acceptance/` contient 23 tests écrits par l'équipe. Ils comparent ton paquet à une implémentation
de référence indépendante et à scikit-learn optimisé. **Ils doivent tous passer** :
`python -m pytest -q tests_acceptance`. Ne les modifie pas : ils seront relancés sur une copie intacte. Le harnais
les lance à chaque point de contrôle. Si l'un d'eux échoue, l'erreur est dans ton paquet : lis le test pour
comprendre ce qu'il attend.
