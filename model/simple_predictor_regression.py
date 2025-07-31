'''
Simple predictor module : This module implements a Simple prediction model using Logistic Regression.
'''

# Import libraries (PEP8 convention)

# -- Import standard libraries --

# -- Import third-party libraries --
import pandas as pd # Pandas is a library for data manipulation and analysis.
from sklearn.linear_model import LogisticRegression # LogisticRegression is a machine learning model for classification tasks.
from sklearn.preprocessing import LabelEncoder # LabelEncoder is used to convert categorical labels into numerical format.


# -- Import local libraries --
from .base_predictor import BasePredictor # BasePredictor is a custom class that provides a base for creating predictors.

class SimplePredictorRegression(BasePredictor):
    '''
    Simple prediction model using Random Forest Regressor. 
    '''

    def __init__(self): # Initialize the SimplePredictor class.
        super().__init__("Logistic Regression") # Call the constructor of the BasePredictor class with a name for the predictor.
        self.label_encoders = {} # Initialize a dictionary to hold label encoders for categorical features.

    #----------------------------------------------------------------------------------------------------------------------------------------

    def build_model(self): # Build the Logistic Regression model.
        # 1. Create a Logistic Regression model.
        self.model = LogisticRegression(
            penalty='l2', # Use L2 regularization.
            C=1.0, # Inverse of regularization strength.
            solver='lbfgs', # Optimization algorithm.
            max_iter=1000, # Maximum number of iterations.
            random_state=42 # Random state for reproducibility.
        )

        # 2. Print the model summary.
        print(f" Logistic Regression model built")

    #----------------------------------------------------------------------------------------------------------------------------------------
    
    def preprocess_features (self, data: pd.DataFrame) -> pd.DataFrame: # Preprocess features for simple model, implements abstract method from BasePredictor.
        print(f"Available columns in preprocess_features: {list(data.columns)}")
        processed_data = data.copy()

        processed_data = processed_data.drop(columns=['home_team', 'away_team', 'home_team_encoded', 'away_team_encoded'], errors='ignore') # Drop columns that are not needed for the model.

        # S'assurer que toutes les colonnes sont numériques
        for col in processed_data.columns:
            if processed_data[col].dtype == 'object':
                print(f"Warning: Column {col} is not numeric, attempting conversion")
                processed_data[col] = pd.to_numeric(processed_data[col], errors='coerce')

        processed_data = processed_data.fillna(0)

        print(f"Final processed features: {list(processed_data.columns)}")
        print(f"Shape: {processed_data.shape}")

        return processed_data

    #----------------------------------------------------------------------------------------------------------------------------------------
