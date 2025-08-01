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
from sklearn.model_selection import train_test_split, cross_val_score # This function is used to split the dataset into training and testing sets.
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
        print(f" {len(data)} football match ready for {self.__class__.__name__}") # Print the number of football matches ready for processing.
        self.target_column = target_column # Set the target column name.
        y = data[target_column].copy() # Copy the target column to y for later use.

        # 2. Delete the target column from the dataset to avoid using it as a feature.
        features_to_delete = ['home_goals', 'away_goals', 'result'] # List of features to delete from the dataset.
        features_data = data.drop(columns=features_to_delete, errors='ignore') # Drop the specified features from the dataset, ignoring errors if they don't exist.

        # 3. List of valid features to keep in the dataset.
        valid_features = [
            'home_team', 'away_team', # Team names(known before the match).
            'home_recent_form', 'away_recent_form', # Recent form of the teams (known before the match).
            'h2h_home_wins', 'h2h_away_wins', # Head-to-head wins of the teams (known before the match).
            'home_league_position', 'away_league_position'] # League positions of the teams (known before the match).

        if all(column in features_data.columns for column in valid_features): # Check if all valid features are present in the dataset.
            features_data = features_data[valid_features] # Keep only the valid features in the dataset.

        # 4. Call the preprocess 
        processed_features = self.preprocess_features(features_data) # Preprocess the features using the preprocess_features method.
        
        # 5. Used processed features
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

    def evaluate_with_cross_validation(self, X:pd.DataFrame, y: pd.Series, cv_folds: int = 5) -> dict: # Evaluate the model using cross-validation and return the average accuracy.
        # Condition to check if the model is trained
        if not self.is_trained: # If the model is not trained, raise an error
            raise ValueError("Model is not trained yet. Please train the model before evaluating with cross-validation.") # Raise an error if the model is not trained.

        # 1. Cross-validation to evaluate the model's performance
        print(f"\n Cross-validation analysis ({cv_folds} folds)") # Print a message indicating the start of cross-validation analysis.

        cv_scores = cross_val_score(self.model, X, y, cv=cv_folds) # Perform cross-validation on the model with the provided data and number of folds.

        # 2. Scrores of cross-validation
        train_score = self.model.score(X, y) # Calculate the training score of the model on the provided data.
        cv_mean = cv_scores.mean() # Calculate the mean of the cross-validation scores.
        cv_std = cv_scores.std() # Calculate the standard deviation of the cross-validation scores.

        # 3. Calculate overfitting indicator
        overfitting_gap = train_score - cv_mean # Calculate the overfitting gap by subtracting the mean cross-validation score from the training score.

        # 4. Metrics
        metrics = {
            'train_accuracy': train_score, # Training accuracy of the model.
            'cv_mean_accuracy': cv_mean, # Mean accuracy from cross-validation.
            'cv_std_accuracy': cv_std, # Standard deviation of accuracy from cross-validation.
            'cv_scores': cv_scores, # Cross-validation scores.
            'overfitting_gap': overfitting_gap, # Overfitting gap between training score
            'is_overfitting': overfitting_gap > 0.1 # 10% threshold for overfitting
        }

        # 5. Print the evaluation metrics
        print(f"Training accuracy: {train_score: .2f}") # Print the training accuracy of the model.
        print(f"CV mean accuracy: {cv_mean: .2f} (+/- {cv_std: .2f})") # Print the mean accuracy from cross-validation along with its standard deviation.
        print(f"Individual CV scores : {[f'{score:.2f}' for score in cv_scores]}") # Print the individual cross-validation scores.
        print(f"Overfitting gap: {overfitting_gap:.2f}") # Print the overfitting gap between training score and cross-validation mean accuracy.
        
        if metrics['is_overfitting']: # Check if the model is overfitting based on the overfitting gap.
            print("Overfitting detected! (gap > 0.1)") # Print a message indicating that overfitting is detected.
            print(" Model memoizes training data but fails to generalize") # Print a message indicating that the model is memorizing training data but failing to generalize.

        else: # If the model is not overfitting
            print("Good generalization (no significant overfitting detected)") # Print a message indicating that the model has good generalization without significant overfitting.

        return metrics # Return the dictionary containing the evaluation metrics from cross-validation.
    
    #----------------------------------------------------------------------------------------------------------------------------------------

    def evaluate(self, X_test : pd.DataFrame, y_test: pd.Series, X_train: pd.DataFrame = None, y_train: pd.Series = None) -> dict: # Evaluate the model's performance on the test set and  return a dictionary of evaluation metrics.
        # 1. Make predictions on the test set (data is already preprocessed)
        y_pred = self.predict_preprocessed(X_test) # Make predictions using the preprocessed test data.
        test_accuracy = accuracy_score(y_test, y_pred) # Calculate the accuracy of the predictions on the test set.

        # 2. Calculate evaluation metrics.
        metrics = {
            'test_accuracy': test_accuracy, # Calculate the accuracy of the model on the test set.
            'classification_report': classification_report(y_test, y_pred), # Generate a classification report.
            'confusion_matrix': confusion_matrix(y_test, y_pred) # Generate a confusion matrix.
        }

        # 3. Print the evaluation metrics.
        if X_train is not None and y_train is not None: # If training data is provided, calculate and print additional metrics.
            train_pred = self.predict_preprocessed(X_train) # Make predictions on the training set.
            train_accuracy = accuracy_score(y_train, train_pred) # Calculate the accuracy of the predictions on the training set.
            
            metrics['train_accuracy'] = train_accuracy # Add the training accuracy to the metrics dictionary.
            metrics['overfitting_gap'] = train_accuracy - test_accuracy # Calculate the overfitting gap between training and test accuracy.
            metrics['is_overfitting'] = metrics['overfitting_gap'] > 0.1 # Determine if the model is overfitting based on the overfitting gap.
            
            print(f"\nTrain vs Test comparison") # Print a message indicating the comparison between training and test performance.
            print(f"Train accuracy: {train_accuracy:.3f}") # Print the training accuracy.
            print(f"Test accuracy: {test_accuracy:.3f}") # Print the test accuracy.
            print(f"Overfitting gap: {metrics['overfitting_gap']:.3f}") # Print the overfitting gap.
            
            if metrics['is_overfitting']: # Check if the model is overfitting based on the overfitting gap.
                print("Overfitting detected !") # Print a message indicating that overfitting is detected.
            else:
                print("Good generalization") # Print a message indicating that the model has good generalization.
        
        print(f"\nEvaluation metrics for {self.model_name}:") # Print a message indicating the evaluation metrics for the model.
        print(f"Test accuracy: {test_accuracy:.3f}") # Print the test accuracy.
        
        # 4 Return the evaluation metrics.
        return metrics # Return the dictionary containing the evaluation metrics.
