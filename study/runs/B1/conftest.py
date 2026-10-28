"""Rend gp_lite importable depuis le racine du projet (et ses sous-dossiers de tests)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
