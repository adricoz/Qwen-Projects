# conftest.py racine : permet a pytest d'ajouter le dossier courant a sys.path
# (import du paquet gp_lite sans installation).
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
