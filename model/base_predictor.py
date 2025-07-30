'''
Class of base for all prediction models.
'''

# Import libraries (PEP8 convention)

# -- Import standard libraries --
import os # This library provides a way of using operating system dependent functionality like reading or writing to the file system.
from abc import ABC, abstractmethod # This library allows to create abstract classes and abstract methods.

# -- Import third-party libraries --
import pandas as pd # Pandas is a library for data manipulation and analysis.
import numpy as np # NumPy is a library for numerical computations in Python.
import joblib # Joblib is a library for saving and loading Python objects, especially large numpy arrays.
from sklearn.model_selection import train_test_split # This function is used to split the dataset into training and testing sets.
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix # These functions are used to evaluate the performance of the model.

# -- Import local libraries --


class BasePredictor(ABC):
    '''
    Abstract base class for all prediction models. Defines the interface for training and predicting.
    Classe abstraite de base pour tous les modèles de prédiction. Définit l'interface pour l'entraînement et la prédiction.
    '''

    def __init__(self, model_name: str): # Initialization method for the BasePredictor class.
        self.model_name = model_name # Set the model name.
        self.model = None # Initialize the model attribute to None.
        self.is_trained = False # Initialize the is_trained attribute to False.
        self.feature_columns = None # Initialize the feature columns attribute to None.
        self.target_column = None # Initialize the target column attribute to None.

    #----------------------------------------------------------------------------------------------------------------------------------------

    @abstractmethod
    def build_model(self): # Build the prediction model, must be implemented by subclasses.
        pass

    #----------------------------------------------------------------------------------------------------------------------------------------

    @abstractmethod
    def preprocess_features(self, data: pd.DataFrame) -> pd.DataFrame: # Preprocess the features of the dataset, must be implemented by subclasses.
        pass

    #----------------------------------------------------------------------------------------------------------------------------------------

    def prepare_data(self, data: pd.DataFrame, target_column: str = 'result', test_size: float = 0.2): # Prepare the data for training by preprocessing and splitting into training and testing sets.
        # 1. Save the colomn names of the features and target.
        print(f" {len(data)} football match ready for {self.__class__.__name__}")
        self.target_column = target_column
        y = data[target_column].copy()

        # 2. Preprocess only the features (without the target column)
        features_data = data.drop(columns=[target_column])
        processed_features = self.preprocess_features(features_data)

        # 3. Now X contains the processed features
        X = processed_features
        self.feature_columns = X.columns.tolist()

        # 4. Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )

        # 5. Return the training and testing sets
        return X_train, X_test, y_train, y_test
    
    #----------------------------------------------------------------------------------------------------------------------------------------

    def train(self, X_train: pd.DataFrame, y_train: pd.Series): # Train the prediction model with the provided training data.
        # 1. Check if the model is already built, if not, build it.
        if self.model is None: # If the model is not built yet, raise an error.
            self.build_model() # Build the model using the specific model's method.
        
        # 2. If the model is already built, proceed to fit the model. Print a message indicating the start of model training.
        print(f" Model training {self.model_name}...")

        # 3. Fit the model to the training data.
        self.model.fit(X_train, y_train)
        
        # 4. Set the is_trained attribute to True to indicate that the model has been trained.
        self.is_trained = True # Set the is_trained attribute to True after training.
        print(f" Model {self.model_name} trained successfully.") # Print a success message.

    #----------------------------------------------------------------------------------------------------------------------------------------

    def predict(self, X: pd.DataFrame) -> np.ndarray: # Make predictions on new data.
        # 1. Check if the model is trained before making predictions.
        if not self.is_trained: # If the model is not trained, raise an error
            raise ValueError("Model is not trained yet. Please train the model before making predictions.") # Raise an error if the model is not trained.
        
        # 2. Preprocess the new data using the same preprocessing as training
        X_processed = self.preprocess_features(X)
        
        # 3. If the model is trained, proceed to make predictions.
        return self.model.predict(X_processed) # Return the predictions made by the model on the processed data.
    
    def predict_preprocessed(self, X: pd.DataFrame) -> np.ndarray: # Make predictions on already preprocessed data.
        # 1. Check if the model is trained before making predictions.
        if not self.is_trained: # If the model is not trained, raise an error
            raise ValueError("Model is not trained yet. Please train the model before making predictions.") # Raise an error if the model is not trained.
        
        # 2. Data is already preprocessed, use it directly
        return self.model.predict(X) # Return the predictions made by the model on the preprocessed data.
    
    #----------------------------------------------------------------------------------------------------------------------------------------

    def evaluate(self, X_test : pd.DataFrame, y_test: pd.Series) -> dict: # Evaluate the model's performance on the test set and  return a dictionary of evaluation metrics.
        # 1. Make predictions on the test set (data is already preprocessed)
        y_pred = self.predict_preprocessed(X_test)

        # 2. Calculate evaluation metrics.
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred), # Calculate the accuracy of the model.
            'classification_report': classification_report(y_test, y_pred), # Generate a classification report.
            'confusion_matrix': confusion_matrix(y_test, y_pred) # Generate a confusion matrix.
        }

        # 3. Print the evaluation metrics.
        print(f"Evaluation metrics for {self.model_name}:") # Print a message indicating the evaluation of the model.
        print(f"Accuracy: {metrics['accuracy']:.2f}") # Print the accuracy of the model.
        print(f"\nClassification report: {metrics['classification_report']}") # Print the classification report.
        
        # 4 Return the evaluation metrics.
        return metrics # Return the dictionary containing the evaluation metrics.