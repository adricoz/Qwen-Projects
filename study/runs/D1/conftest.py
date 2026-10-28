"""Racine du projet : rend ``gp_lite`` importable par pytest (dossier courant sur sys.path)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
