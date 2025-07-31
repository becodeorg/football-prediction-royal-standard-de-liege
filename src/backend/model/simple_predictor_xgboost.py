'''
Simple predictor module : This module implements a simple predictor using a Random Forest Classifier.
'''

# Import libraries (PEP8 convention)

# -- Import standard libraries --

# -- Import third-party libraries --
import pandas as pd # Pandas is a library for data manipulation and analysis.
from xgboost import XGBClassifier # XGBClassifier is an implementation of the XGBoost algorithm for classification tasks.
from sklearn.preprocessing import LabelEncoder # LabelEncoder is used to convert categorical labels into numerical format.


# -- Import local libraries --
from .base_predictor import BasePredictor # BasePredictor is a custom class that provides a base for creating predictors.

class SimplePredictorXGBoost(BasePredictor):
    '''
    Simple prediction model using XGBoost classifier.
    '''

    def __init__(self): # Initialize the SimplePredictor class.
        super().__init__("XGBoost Classifier") # Call the constructor of the BasePredictor class with a name for the predictor.
        self.label_encoders = {} # Initialize a dictionary to hold label encoders for categorical features.

    #----------------------------------------------------------------------------------------------------------------------------------------

    def build_model(self): # Build the XGBoost Classifier model.
        self.model = XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            use_label_encoder=False,
            eval_metric='mlogloss'
        )

        # 2. Print the model summary.
        print(f" XGBoost model built with {self.model.n_estimators} trees")

    #----------------------------------------------------------------------------------------------------------------------------------------
    
    def preprocess_features(self, data: pd.DataFrame) -> pd.DataFrame:
        print(f"Available columns in preprocess_features: {list(data.columns)}")
        processed_data = data.copy()

        # Exemple : supprimer uniquement les colonnes non utilisables pour la prédiction
        processed_data = processed_data.drop(
            columns=['home_team', 'away_team', 'home_goals', 'away_goals', 'result'], errors='ignore'
        )

        # Encodage des colonnes catégorielles si besoin
        for col in processed_data.columns:
            if processed_data[col].dtype == 'object':
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    processed_data[col] = self.label_encoders[col].fit_transform(processed_data[col].astype(str))
                else:
                    processed_data[col] = self.label_encoders[col].transform(processed_data[col].astype(str))

        processed_data = processed_data.fillna(0)
        print(f"Final processed features: {list(processed_data.columns)}")
        print(f"Shape: {processed_data.shape}")

        return processed_data

    #----------------------------------------------------------------------------------------------------------------------------------------
