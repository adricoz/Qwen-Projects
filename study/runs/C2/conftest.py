"""Conftest : met le dossier du projet sur sys.path pour pytest."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
