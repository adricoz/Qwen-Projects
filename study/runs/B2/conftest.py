"""Ajoute le dossier du projet à sys.path pour que `gp_lite` soit importable."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
