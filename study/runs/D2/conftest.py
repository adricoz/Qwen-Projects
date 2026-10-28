"""Permet d'importer gp_lite depuis la racine du projet (pytest ajoute déjà la racine)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
