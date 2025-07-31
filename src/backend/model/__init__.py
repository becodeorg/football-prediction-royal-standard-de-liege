'''
Model package for football prediction.
Package de modèles pour la prédiction de football.
'''

# Import all predictor classes
from .base_predictor import BasePredictor
from .simple_predictor_tree import SimplePredictorTree
from .simple_predictor_forest import SimplePredictorForest
from .simple_predictor_gradient import SimplePredictorGradient
from .simple_predictor_regression import SimplePredictorRegression
from .simple_predictor_poisson import SimplePredictorPoisson
from .simple_predictor_xgboost import SimplePredictorXGBoost

# Define what gets imported with "from model import *"
__all__ = ['BasePredictor', 'SimplePredictorTree', 'SimplePredictorForest', 'SimplePredictorGradient', 'SimplePredictorRegression', 'SimplePredictorPoisson', 'SimplePredictorXGBoost']