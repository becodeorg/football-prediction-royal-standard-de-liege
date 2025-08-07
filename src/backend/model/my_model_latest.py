# ----- Import libraries PEP 8 -----

# ----- Standard library -----
import pandas as pd
import logging
import sys
import os
import numpy as np

# ----- Third-party libraries -----
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, accuracy_score, brier_score_loss
from sklearn.calibration import CalibratedClassifierCV

# -----  -----
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')) # Adjust path as needed
sys.path.insert(0, project_root) # Add project root to sys.path to import local modules

# ----- Local application/library specific imports -----
from utils.data_io import load_csv
from utils.logger_config import configure_logging

# ----- Log configuration -----
configure_logging() # Global logging initialization - executed once per application startup
logger = logging.getLogger(__name__) # Module-specific logger creation using Python's __name__ magic variable

# ----- Class definition -------------------------------------------------------------------------------------------------------------------------

class ModelTrainer:
    """
    Class for building, training and tuning ML models using pipeline and GridSearchCV.
    SUPPORTED MODELS: RandomForest, XGBoost, LogisticRegression with automatic calibration.
    FEATURES: Automated preprocessing, hyperparameter optimization, probabilistic predictions.
    PURPOSE: Football match outcome prediction with Brier Score evaluation.
    """

    def __init__(self, df: pd.DataFrame, target_column: str):
        """
        Initialize the ModelTrainer with data and target specification.
        :param df: Processed DataFrame without gaps and with encoded features.
        :param target_column: Name of the target feature (what we are predicting).
        """
        self.df = df # DataFrame with preprocessed features
        self.target_column = target_column # Name of the target column
        self.model = None # Trained model or best estimator from GridSearchCV
        self.pipeline = None # Pipeline for preprocessing and model fitting
        self.X_train = self.X_test = self.y_train = self.y_test = None # Training and testing data splits

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------

    def apply_sample(self, sample_size: int, random_state: int = 42) -> None:
        """
        Apply random sampling to DataFrame for faster experimentation during development.
        PURPOSE: Reduce dataset size for quicker prototyping and hyperparameter testing.
        WARNING: This method MODIFIES self.df in place - original data is permanently replaced.

        :param sample_size: Number of rows to include in the sample
        :param random_state: Random seed for reproducible sampling (default: 42)
        :raises ValueError: If sample_size >= dataset size (invalid sampling)
        """
        # VALIDATION: Sample size must be smaller than dataset
        if sample_size >= len(self.df):
            raise ValueError(f"Sample size ({sample_size}) must be less than dataset size ({len(self.df)}).")
        
        # PERFORM: Random sampling with reproducible seed
        self.df = self.df.sample(n=sample_size, random_state=random_state) # Apply pandas sampling method

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------

    def split_data(self, test_size=0.25, random_state=42) -> None:
        """
        Split the dataset into training and testing sets using train_test_split.
        PURPOSE: Prepare data for model training and evaluation with proper stratification.
        
        :param test_size: Proportion of dataset for testing (default: 0.25 = 25%)
        :param random_state: Random seed for reproducible splits (default: 42)
        :return: None (modifies instance attributes)
        """
        # FEATURE SEPARATION: Extract features by dropping target and date columns
        X = self.df.drop(columns=[self.target_column, "Date"]) # Features matrix without target and date
        y = self.df[self.target_column] # Target variable (what we want to predict)
        
        # DATA SPLITTING: Create train/test splits with sklearn
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state # Features, target, proportion and seed
        )

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------

    def build_pipeline(self, model) -> None:
        """
        Build a sklearn pipeline with preprocessing and classification model.
        PURPOSE: Create automated data preprocessing pipeline followed by ML model.
        
        :param model: Classification model to use in the pipeline (RandomForest, XGBoost, LogisticRegression)
        :raises ValueError: If model is None
        :return: None (modifies self.pipeline)
        """
        # VALIDATION: Model must be provided
        if model is None:
            raise ValueError("You must provide a model to build the pipeline.")

        # CATEGORICAL PREPROCESSING: Define team features and their transformation
        categorical_features = ["HomeTeam", "AwayTeam"] # Team names requiring encoding
        categorical_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")), # Fill missing values with mode
            ("encoder", OneHotEncoder(handle_unknown="ignore")) # Convert team names to binary vectors
        ])

        # NUMERICAL PREPROCESSING: Define statistics features and their transformation
        numerical_features = [
            "home_form_last5", "away_form_last5", # Team form over last 5 matches
            "home_goals_scored_last5", "away_goals_scored_last5", # Goals scored in last 5 matches
            "home_goals_conceded_last5", "away_goals_conceded_last5", # Goals conceded in last 5 matches
            "home_goals_diff_last5", "away_goals_diff_last5", # Goal difference in last 5 matches
            "home_win_rate_last5", "away_win_rate_last5", # Win rate over last 5 matches
            "head2head_form_last3", "head2head_goal_diff_last3" # Head-to-head statistics
        ]

        # NUMERICAL TRANSFORMATION: Define numerical transformer for statistics features
        numerical_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="mean")), # Fill missing values with mean
            ("scaler", StandardScaler()) # Normalize features to same scale
        ])

        # PREPROCESSING COMBINATION: Merge numerical and categorical transformers
        preprocessor = ColumnTransformer(transformers=[
            ("num", numerical_transformer, numerical_features), # Apply numerical preprocessing
            ("cat", categorical_transformer, categorical_features) # Apply categorical preprocessing
        ])

        # FINAL PIPELINE: Combine preprocessing with ML model
        self.pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor), # Data transformation step
            ("model", model) # Classification model step
        ])

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------

    def find_best_hyperparameters(self, model=None, param_grid=None, cv: int = 5,
                                  scoring: str = "f1_macro") -> None:
        """
        Find the best hyperparameters using GridSearchCV with cross-validation and model calibration.
        PURPOSE: Optimize model performance through systematic hyperparameter search and probability calibration.
        
        :param model: Classification model to optimize (RandomForest, XGBoost, LogisticRegression)
        :param param_grid: Dictionary of hyperparameters to search through
        :param cv: Number of folds for cross-validation (default: 5)
        :param scoring: Optimization metric (default: "f1_macro")
        :raises ValueError: If cv < 2 or if no model provided
        :return: None (modifies self.model with best calibrated estimator)
        """
        # VALIDATION: Cross-validation must have at least 2 folds
        if cv < 2:
            raise ValueError("cv must be at least 2 or higher")

        # PIPELINE PREPARATION: Build pipeline if not already created
        if self.pipeline is None:
            if model is None:
                raise ValueError("Need to choose model")
            self.build_pipeline(model) # Create preprocessing + model pipeline

        # PARAMETER GRID: Set empty grid if none provided
        if param_grid is None:
            param_grid = {} # Empty grid means use default parameters

        # GRID SEARCH: Systematic hyperparameter optimization with cross-validation
        grid_search = GridSearchCV(
            estimator=self.pipeline, # Pipeline to optimize
            param_grid=param_grid, # Hyperparameters to test
            cv=cv, # Cross-validation folds
            scoring=scoring, # Optimization metric
            n_jobs=-1, # Parallelization (all available cores)
            verbose=3 # Detailed output during search
        )

        # TRAINING: Fit GridSearchCV to find best hyperparameters
        grid_search.fit(self.X_train, self.y_train) # Train on training data
        self.model = grid_search.best_estimator_ # Save best model from grid search

        # CALIBRATION: Improve probability predictions for Brier Score
        calibrated_model = CalibratedClassifierCV(self.model, method='isotonic', cv=3) # Isotonic calibration with 3-fold CV
        calibrated_model.fit(self.X_train, self.y_train) # Fit calibration on training data

        # FINAL MODEL: Replace with calibrated version for better probabilities
        self.model = calibrated_model # Save calibrated model as final model

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------

    def train(self) -> None:
        """
        Train the model using the training data after hyperparameter optimization.
        PURPOSE: Final training step on the calibrated model with best hyperparameters.
        NOTE: This method is typically called after find_best_hyperparameters().

        :raises ValueError: If no model has been selected via find_best_hyperparameters()
        :return: None (modifies self.model state)
        """
        # VALIDATION: Model must exist before training
        if self.model is None:
            raise ValueError("No model to train. Use find_best_hyperparameters() first.")
        
        # TRAINING: Fit the calibrated model on training data
        self.model.fit(self.X_train, self.y_train) # Train model on X_train features and y_train targets

    def predict(self):
        """
        Generate class predictions on the test set using the trained model or pipeline.
        PURPOSE: Make discrete class predictions (Défaite, Nul, Victoire) for model evaluation.
        
        :return: NumPy array of predicted class values for X_test
        :raises ValueError: If neither a trained model nor pipeline is available
        """
        # PREDICTION WITH MODEL: Use trained/calibrated model if available
        if self.model:
            return self.model.predict(self.X_test) # Predict using trained model
        
        # PREDICTION WITH PIPELINE: Fallback to pipeline if model not available
        elif self.pipeline:
            return self.pipeline.predict(self.X_test) # Predict using pipeline
        
        # ERROR HANDLING: No model or pipeline available for prediction
        logger.error("No model or pipeline found.") # Log error for debugging
        raise ValueError("Model is not trained yet") # Raise informative error

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------

    def predict_proba(self):
        """
        Generate probability predictions on the test set for Brier Score evaluation (Request from Antoine)
        
        :return: NumPy array of probability predictions for X_test.
                Each row contains probabilities for [Défaite, Nul, Victoire]
        :raises ValueError: If neither a trained model nor pipeline is available.
        """

        if self.model: # If a trained model is available, use it for probability prediction
            return self.model.predict_proba(self.X_test) # Returns probabilities for each class
        
        elif self.pipeline: # If a pipeline is available, use it for probability prediction
            return self.pipeline.predict_proba(self.X_test) # Returns probabilities for each class
        
        logger.error("No model or pipeline found.") # If neither is available, log an error
        raise ValueError("Model is not trained yet") # Raise an error if no model or pipeline is found

    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------

    def evaluate(self):
        """
        Evaluate model performance using accuracy and detailed classification report.
        PURPOSE: Comprehensive model evaluation with accuracy score and per-class metrics.
        
        :return: Accuracy score (float between 0 and 1)
        """
        # PREDICTIONS: Generate class predictions for evaluation
        y_pred = self.predict() # Get predicted classes from trained model
        
        # ACCURACY: Calculate overall accuracy score
        acc = accuracy_score(self.y_test, y_pred) # Compare true vs predicted labels
        print(f"\nACCURACY: {acc:.2f}") # Display accuracy with 2 decimal places
        
        # DETAILED REPORT: Show precision, recall, F1-score per class
        print("\n===== CLASSIFICATION REPORT =====")
        print("\n")
        print(classification_report(self.y_test, y_pred, digits=2)) # Detailed metrics for each class

        return acc # Return accuracy for further analysis

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ----- Main execution block -----

if __name__ == '__main__':
    df = load_csv(filedir="prepared", filename="B1_old.csv")
    df['FTR_xgb'] = df['FTR'].replace({-1: 0, 0: 1, 1: 2}) # Feature for XGBoost, to keep this line
    df['head2head_form_last3'] = df['head2head_form_last3'].fillna(1.0)
    df['head2head_goal_diff_last3'] = df['head2head_goal_diff_last3'].fillna(0.0)

    # ------- Model Selection ------- ATTENTION : UTILISATION D'UN SEUL MODÈLE À LA FOIS
    # -------- Random Forest Classifier --------
    
    model_trainer = ModelTrainer(df, "FTR")
    model_trainer.split_data()

    # Model pipeline and hyperparameters
    model = RandomForestClassifier(random_state=42, class_weight="balanced")
    param_grid = {
        "model__n_estimators": [150, 250, 300],
        "model__max_depth": [15, 20, 25],
        "model__min_samples_leaf": [1, 2],
        "model__max_features": ["sqrt", "log2"]
    }

    # -------- LogisticRegression --------
    #
    # model_trainer = ModelTrainer(df, "FTR")
    # model_trainer.split_data()
    
    ## Model pipeline and hyperparameters
    # model = LogisticRegression(random_state=42, max_iter=500)
    # param_grid = {
    #     "model__C": [0.1, 1, 10]
    # }

    # -------- XGBoost --------

    # model_trainer = ModelTrainer(df, "FTR_xgb")
    # model_trainer.split_data()
    
    ## Model pipeline and hyperparameters
    # model = XGBClassifier(random_state=42)
    # param_grid = {
    #     "model__n_estimators": [100, 200],
    #     "model__max_depth": [3, 6]
    # }

    model_trainer.find_best_hyperparameters(
        model=model,
        param_grid=param_grid,
        scoring="f1_macro",
        cv=5
    )
    
    model_trainer.train()
    # save_model(model_trainer.model, "RFC_Belgium_league_model.joblib")
    model_trainer.evaluate()
    
    predictions = model_trainer.predict()
    #print(predictions)

    # ------ Display results ------

    # BRIER SCORE EVALUATION: Calculate probabilistic prediction quality for model assessment
    probabilities = model_trainer.predict_proba() # Get probabilities for each class
    print("\n===== BRIER SCORE EVALUATION =====") 
    print("\nSize of the table :", probabilities.shape) # Size of the table (number of matches X number of classes)
    print("\nFirst 5 rows of probabilities:")
    print(probabilities[:5]) # Display first 5 rows of probabilities
    print("\nClass of model:", model_trainer.model.classes_) # Display classes of the model

    # DATA PREPARATION: Prepare true and predicted values for Brier Score calculation
    y_true = model_trainer.y_test # True labels from test set
    y_pred = predictions # Predicted labels from model

    # ONE-HOT ENCODING: Convert true classes to binary format for Brier Score
    classes = model_trainer.model.classes_ # Get class labels from trained model
    y_true_encoded = pd.get_dummies(y_true).reindex(columns=classes, fill_value=0) # Convert to one-hot with proper class order
    
    # BRIER SCORE CALCULATION: Compute Brier Score for each class
    brier_scores = [] # List to store individual class Brier Scores
    for i, class_label in enumerate(classes): # Iterate through each class
        brier = brier_score_loss(y_true_encoded.iloc[:, i], probabilities[:, i]) # Calculate Brier Score for this class
        brier_scores.append(brier) # Add to list
    
    # AVERAGE BRIER SCORE: Calculate mean across all classes
    avg_brier_score = np.mean(brier_scores) # Average of all class-specific Brier Scores
    
    # RESULTS DISPLAY: Show comprehensive Brier Score analysis
    print(f"\n===== BRIER SCORE ANALYSIS =====")
    print("\n")
    print(f"Brier Score : {avg_brier_score:.4f}") # Display average Brier Score with 4 decimals
    
    # DS SCORE CALCULATION: Calculate project-specific evaluation metric
    ds_score = 2 * (2 - avg_brier_score) # Project formula for evaluation score
    print(f"DS Score : {ds_score:.4f} / 4.0") # Display DS Score out of 4.0
    
    # PERFORMANCE INTERPRETATION: Provide qualitative assessment based on Brier Score
    if avg_brier_score < 0.20: # Excellent threshold
        print("EXCELLENT - Excellent probabilistic predictions !") # Excellent probabilistic predictions
    elif avg_brier_score < 0.25: # Good threshold
        print("CORRECT - Better than random, but improvable") # Better than random, but improvable
    elif avg_brier_score < 0.30: # Average threshold
        print("MOYEN - Slightly better than random") # Slightly better than random
    else: # Poor performance
        print("FAIBLE - Unreliable predictions") # Unreliable predictions

    # REFERENCE BENCHMARK: Show random baseline for comparison
    print(f"Random Brier Score reference ≈ 0.67 for 3 classes") # Random Brier Score reference for 3 classes

    # ERROR ANALYSIS: Identify and analyze model prediction errors
    erreurs = np.where(y_true != y_pred)[0] # Find indices where predictions don't match true labels

    # ERROR EXAMPLES: Display specific examples of prediction errors
    print("\n===== ERROR EXAMPLES =====")
    print("\n")
    for idx in erreurs[:5]: # Show first 5 error cases
        print(f"Index: {idx}, True: {y_true.iloc[idx]}, Predicted: {y_pred[idx]}") # Display index, true and predicted values

    # FEATURE ANALYSIS: Examine features of misclassified examples
    print("\n===== FEATURES OF MISCLASSIFIED EXAMPLES =====")
    for index in erreurs[:5]: # Analyze first 5 error cases
        print(f"\nIndex: {index}") # Display case index
        print(model_trainer.X_test.iloc[index]) # Show all features for this misclassified example