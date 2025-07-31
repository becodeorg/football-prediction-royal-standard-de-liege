'''
Simple predictor module : This module implements a simple predictor using a Random Forest Classifier.
'''

# Import libraries (PEP8 convention)

# -- Import standard libraries --

# -- Import third-party libraries --
import pandas as pd # Pandas is a library for data manipulation and analysis.
from sklearn.tree import DecisionTreeClassifier # DecisionTreeClassifier is a class from scikit-learn for building decision tree models.
from sklearn.preprocessing import LabelEncoder # LabelEncoder is used to convert categorical labels into numerical format.
from sklearn.preprocessing import StandardScaler # StandardScaler is used to standardize features by removing the mean and scaling to unit variance.

# -- Import local libraries --
from .base_predictor import BasePredictor # BasePredictor is a custom class that provides a base for creating predictors.

class SimplePredictorTree(BasePredictor):
    '''
    Simple prediction model using Decision Tree Classifier. 
    '''

    def __init__(self): # Initialize the SimplePredictor class.
        super().__init__("Decision Tree Classifier") # Call the constructor of the BasePredictor class with the model name.
        self.label_encoders = {} # Initialize a dictionary to hold label encoders for categorical features.

    #----------------------------------------------------------------------------------------------------------------------------------------

    def build_model(self): # Build the model, implements abstract method from BasePredictor.
        # 1. Create a Decision Tree Classifier model.
        self.model = DecisionTreeClassifier(
            max_depth=3, # Limit the depth of the tree to prevent overfitting.
            random_state=42, # Random state for reproducibility.
            class_weight='balanced' # Use balanced class weights to handle class imbalance.
        )

        # 2. Print the model summary.
        print(f" Decision Tree model built with max depth {self.model.max_depth}")

    #----------------------------------------------------------------------------------------------------------------------------------------
    
    def preprocess_features (self, data: pd.DataFrame) -> pd.DataFrame: # Preprocess features for simple model, implements abstract method from BasePredictor.
        print(f"Available columns in preprocess_features: {list(data.columns)}") # Print available columns for debugging
        processed_data = data.copy() # Create a copy of the data to avoid modifying the original DataFrame

        # Drop home_team and away_team if present
        processed_data = processed_data.drop(columns=['home_team', 'away_team'], errors='ignore') # Drop team columns for simplicity


        # Check columns are categorical values
        for column in processed_data.columns: # Check if the column is categorical
            if processed_data[column].dtype == 'object': # If the column is of type object (categorical)
                print(f"Warning: Column {column} is not numeric, attempting conversion") # Attempt to convert categorical columns to numeric
                processed_data[column] = pd.to_numeric(processed_data[column], errors='coerce') # Convert to numeric, coercing errors to NaN

        processed_data = processed_data.fillna(0) # Fill NaN values with 0 to avoid issues during model training

        # Normalize numerical features
        scaler = StandardScaler() # StandardScaler is used to standardize features by removing the mean and scaling to unit variance.
        processed_data = pd.DataFrame( # Scale the features using StandardScaler
            scaler.fit_transform(processed_data), # Create a DataFrame from the scaled data
            columns=processed_data.columns, # Use the original column names
            index=processed_data.index # Preserve the original index
        )

        print(f"Final processed features: {list(processed_data.columns)}") # Print final processed features for debugging
        print(f"Shape: {processed_data.shape}") # Print the shape of the processed data for debugging

        return processed_data # Return the processed DataFrame with encoded features

    #----------------------------------------------------------------------------------------------------------------------------------------
