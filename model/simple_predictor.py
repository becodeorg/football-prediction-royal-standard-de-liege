'''
Simple predictor module : This module implements a simple predictor using a Random Forest Classifier.
'''

# Import libraries (PEP8 convention)

# -- Import standard libraries --

# -- Import third-party libraries --
import pandas as pd # Pandas is a library for data manipulation and analysis.
from sklearn.ensemble import RandomForestClassifier # RandomForestClassifier is a machine learning model for classification tasks.
from sklearn.preprocessing import LabelEncoder # LabelEncoder is used to convert categorical labels into numerical format.

# -- Import local libraries --
from .base_predictor import BasePredictor # BasePredictor is a custom class that provides a base for creating predictors.

class SimplePredictor(BasePredictor):
    '''
    Simple prediction model using Random Forest Classifier. 
    '''

    def __init__(self): # Initialize the SimplePredictor class.
        super().__init__("Random Forest Classifier") # Call the constructor of the BasePredictor class with a name for the predictor.
        self.label_encoders = {} # Initialize a dictionary to hold label encoders for categorical features.

    #----------------------------------------------------------------------------------------------------------------------------------------

    def build_model(self): # Build the Random Forest Classifier model.
        # 1. Create a Random Forest Classifier with 100 trees and a random state for reproducibility.
        self.model = RandomForestClassifier(
            n_estimators=100, # Number of trees in the forest.
            max_depth=10, # Maximum depth of the trees.
            random_state=42, # Random state for reproducibility.
            class_weight='balanced' # Use balanced class weights to handle class imbalance.
        )

        # 2. Print the model summary.
        print(f" Random Forest model built with {self.model.n_estimators} trees")

    #----------------------------------------------------------------------------------------------------------------------------------------
    
    def preprocess_features (self, data: pd.DataFrame) -> pd.DataFrame: # Preprocess features for simple model, implements abstract method from BasePredictor.
        # 1. Columnd required for the model.
        required_columns = ['home_team', 'away_team', 'home_goals', 'away_goals']

        # 2. Validation (no error if already clean)
        missing_columns = [column for column in required_columns if column not in data.columns] # Identify missing columns
        if missing_columns: # If there are missing columns, raise an error.
            raise ValueError(f"Missing columns in the data : {missing_columns}") # Raise an error if any required columns are missing
        
        # 3. Select only the required columns from the data.
        processed_data = data[required_columns].copy()

        # 4. Convert team names to numbers using LabelEncoder
        for column in ['home_team', 'away_team']:
            if column not in self.label_encoders:
                # Create a new encoder for this column during training
                self.label_encoders[column] = LabelEncoder()
                # Fit and transform the data
                processed_data[column] = self.label_encoders[column].fit_transform(processed_data[column])
            else:
                # Use existing encoder for new predictions
                # Handle unknown teams by assigning them a default value
                try:
                    processed_data[column] = self.label_encoders[column].transform(processed_data[column])
                except ValueError:
                    # If there are unknown teams, handle them gracefully
                    known_teams = set(self.label_encoders[column].classes_)
                    unknown_teams = set(processed_data[column].unique()) - known_teams
                    
                    if unknown_teams:
                        print(f"Warning: Unknown teams found in {column}: {unknown_teams}")
                        print(f"Replacing with most common team: {self.label_encoders[column].classes_[0]}")
                        # Replace unknown teams with the first known team (most frequent)
                        processed_data[column] = processed_data[column].replace(
                            list(unknown_teams), 
                            self.label_encoders[column].classes_[0]
                        )
                    
                    # Now transform with known teams only
                    processed_data[column] = self.label_encoders[column].transform(processed_data[column])

        # 5. Print the number of matches ready for prediction.
        print(f" {len(processed_data)} football match ready for SimplePredictor")

        # 6. Return the processed data with all numeric values.
        return processed_data # Return the processed data containing only numeric values.

    #----------------------------------------------------------------------------------------------------------------------------------------
    