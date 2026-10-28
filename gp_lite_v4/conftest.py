"""Point d'ancrage pytest: ajoute le projet courant au sys.path."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
