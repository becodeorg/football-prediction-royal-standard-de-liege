'''
Model package for football prediction.
Package de modèles pour la prédiction de football.
'''

# Import all predictor classes
from .base_predictor import BasePredictor
from .simple_predictor import SimplePredictor

# Define what gets imported with "from model import *"
__all__ = ['BasePredictor', 'SimplePredictor']

from .base_predictor import BasePredictor
from .simple_predictor import SimplePredictor

__all__ = ['BasePredictor', 'SimplePredictor']