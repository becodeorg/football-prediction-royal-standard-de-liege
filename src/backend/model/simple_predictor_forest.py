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

class SimplePredictorForest(BasePredictor):
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
            n_estimators=50, # Number of trees in the forest.
            max_depth=4, # Maximum depth of the trees.
            min_samples_leaf=5, # Minimum number of samples required to be at a leaf node.
            random_state=42, # Random state for reproducibility.
            class_weight='balanced' # Use balanced class weights to handle class imbalance.
        )

        # 2. Print the model summary.
        print(f" Random Forest model built with {self.model.n_estimators} trees")

    #----------------------------------------------------------------------------------------------------------------------------------------
    
    def preprocess_features (self, data: pd.DataFrame) -> pd.DataFrame: # Preprocess features for simple model, implements abstract method from BasePredictor.
        print(f"Available columns in preprocess_features: {list(data.columns)}")
        processed_data = data.copy()

        # Encode team names if they exist
        if 'home_team' in processed_data.columns:
            if 'home_team' not in self.label_encoders:
                self.label_encoders['home_team'] = LabelEncoder()
                processed_data['home_team_encoded'] = self.label_encoders['home_team'].fit_transform(processed_data['home_team'])
            else:
                # Handle unseen teams by assigning a default value
                try:
                    processed_data['home_team_encoded'] = self.label_encoders['home_team'].transform(processed_data['home_team'])
                except ValueError:
                    # If there are unseen teams, assign them a default value (e.g., 0)
                    known_teams = set(self.label_encoders['home_team'].classes_)
                    processed_data['home_team_encoded'] = processed_data['home_team'].apply(
                        lambda x: self.label_encoders['home_team'].transform([x])[0] if x in known_teams else 0
                    )
        
        if 'away_team' in processed_data.columns:
            if 'away_team' not in self.label_encoders:
                self.label_encoders['away_team'] = LabelEncoder()
                processed_data['away_team_encoded'] = self.label_encoders['away_team'].fit_transform(processed_data['away_team'])
            else:
                # Handle unseen teams
                try:
                    processed_data['away_team_encoded'] = self.label_encoders['away_team'].transform(processed_data['away_team'])
                except ValueError:
                    known_teams = set(self.label_encoders['away_team'].classes_)
                    processed_data['away_team_encoded'] = processed_data['away_team'].apply(
                        lambda x: self.label_encoders['away_team'].transform([x])[0] if x in known_teams else 0
                    )

        # Drop original team name columns and other non-feature columns
        processed_data = processed_data.drop(columns=['home_team', 'away_team'], errors='ignore')

        # Keep only numeric columns for the model
        numeric_columns = ['home_team_encoded', 'away_team_encoded']
        
        # Add other numeric features if they exist
        for col in processed_data.columns:
            if col not in numeric_columns and processed_data[col].dtype in ['int64', 'float64']:
                numeric_columns.append(col)
        
        # Select only the numeric columns that exist
        existing_columns = [col for col in numeric_columns if col in processed_data.columns]
        processed_data = processed_data[existing_columns]

        # Fill any remaining NaN values
        processed_data = processed_data.fillna(0)

        print(f"Final processed features: {list(processed_data.columns)}")
        print(f"Shape: {processed_data.shape}")

        return processed_data

    #----------------------------------------------------------------------------------------------------------------------------------------
    
    def train(self, data: pd.DataFrame): # Train the model with the provided DataFrame
        """
        Train the model using a DataFrame with columns: home_team, away_team, home_goals, away_goals, outcome
        """
        # Prepare data using parent class method
        X_train, X_test, y_train, y_test = self.prepare_data(data, target_column='outcome')
        
        # Call parent train method
        super().train(X_train, y_train)
        
        # Store test data for evaluation
        self.X_test = X_test
        self.y_test = y_test
        
    def predict_home_goals(self, input_df: pd.DataFrame) -> list:
        """
        Predict home team goals (simplified implementation)
        Returns a list with predicted home goals
        """
        # Simple heuristic based on outcome prediction
        outcome = self.predict(input_df)[0]
        if outcome == 'H':  # Home wins
            return [2]  # Home team likely scores 2
        elif outcome == 'A':  # Away wins  
            return [1]  # Home team likely scores 1
        else:  # Draw
            return [1]  # Home team likely scores 1
            
    def predict_away_goals(self, input_df: pd.DataFrame) -> list:
        """
        Predict away team goals (simplified implementation)
        Returns a list with predicted away goals
        """
        # Simple heuristic based on outcome prediction
        outcome = self.predict(input_df)[0]
        if outcome == 'H':  # Home wins
            return [1]  # Away team likely scores 1
        elif outcome == 'A':  # Away wins
            return [2]  # Away team likely scores 2
        else:  # Draw
            return [1]  # Away team likely scores 1
