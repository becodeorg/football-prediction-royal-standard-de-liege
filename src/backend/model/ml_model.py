import pandas as pd
import logging
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, accuracy_score
from utils.logger_config import configure_logging

configure_logging()
logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    A utility class for training, tuning, and evaluating machine learning classification models using pipelines.

    This class handles:
    - Splitting data into training and testing sets
    - Building preprocessing pipelines for numerical and categorical features
    - Integrating models into sklearn pipelines
    - Performing hyperparameter tuning with GridSearchCV
    - Training and evaluating the final model

    It is model-agnostic: you can pass any compatible scikit-learn classifier (e.g., RandomForest, KNN, etc.).

    Attributes:
        df (pd.DataFrame): Input dataframe including features and target.
        target_column (str): Name of the target column.
        model: Trained model after hyperparameter tuning.
        pipeline: Full sklearn pipeline including preprocessing and model.
        X_train, X_test, y_train, y_test: Split datasets.
    """

    def __init__(self, df: pd.DataFrame, target_column: str):
        """
        Initializing a class.

        :param df: Processed DataFrame without gaps and with encoded features.
        :param target_column: Name of the target feature (what we are predicting).
        """
        self.df = df
        self.target_column = target_column
        self.model = None  # Best classifier found after GridSearchCV
        self.pipeline = None
        self.X_train = self.X_test = self.y_train = self.y_test = None

    def split_data(self, test_size=0.25, random_state=42) -> None:
        """
        Split the data into training and testing sets.

        :param test_size: Proportion of test sample (default is 0.25).
        :param random_state: Random seed for reproducibility (default is 42).
        :return: None
        """
        X = self.df.drop(columns=[self.target_column, "Date"])
        y = self.df[self.target_column]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

    def build_pipeline(self, model) -> None:
        """
        Build a sklearn pipeline with preprocessing and the provided model.

        - Numerical features are imputed and scaled.
        - Categorical features are imputed and one-hot encoded.

        :param model: A scikit-learn compatible model.
        :raises ValueError: If no model is provided.
        :return: None
        """
        if model is None:
            raise ValueError("You must provide a model to build the pipeline.")

        # Categorical features
        categorical_features = ["HomeTeam", "AwayTeam"]
        categorical_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ])

        # Numerical features (without target)
        numerical_features = [
            "home_form_last5", "away_form_last5",
            "home_goals_scored_last5", "away_goals_scored_last5",
            "home_goals_conceded_last5", "away_goals_conceded_last5",
            "home_goals_diff_last5", "away_goals_diff_last5",
            "home_win_rate_last5", "away_win_rate_last5",
            "head2head_form_last3", "head2head_goal_diff_last3"
        ]

        numerical_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler())
        ])

        # Merge in ColumnTransformer
        preprocessor = ColumnTransformer(transformers=[
            ("num", numerical_transformer, numerical_features),
            ("cat", categorical_transformer, categorical_features)
        ])

        # Final pipeline
        self.pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ])

    def find_best_hyperparameters(self, model=None, param_grid=None, cv: int = 5,
                                  scoring: str = "neg_mean_absolute_error") -> None:
        """
        Find the best hyperparameters using GridSearchCV.

        If no pipeline exists, it builds one using the provided model.

        :param model: A scikit-learn compatible model (required if pipeline is not yet built)..
        :param param_grid: Dictionary with parameters names (`str`) as keys and lists of parameter settings to try.
        :param cv: Number of cross-validation folds (default is 5). Must be >= 2.
        :param scoring: Scoring metric for evaluating models (default is 'neg_mean_absolute_error').
        :raises ValueError: If `cv` < 2 or if no model is provided when pipeline is missing.
        :return: None
        """
        if cv < 2:
            raise ValueError("cv must be at least 2 or higher")

        if self.pipeline is None:
            if model is None:
                raise ValueError("Need to choose model")
            self.build_pipeline(model)

        if param_grid is None:
            param_grid = {}

        grid_search = GridSearchCV(
            estimator=self.pipeline,
            param_grid=param_grid,
            cv=cv,  # Cross-validation
            scoring=scoring,
            n_jobs=-1,  # Parallelization (all available cores)
            verbose=3
        )

        grid_search.fit(self.X_train, self.y_train)
        self.model = grid_search.best_estimator_  # The best model is saved

    def train(self) -> None:
        """
        Train the model based on the best hyperparameters (if GridSearch was called).

        :raises ValueError: If no model has been selected via find_best_hyperparameters().
        """
        if self.model is None:
            raise ValueError("No model to train. Use find_best_hyperparameters() first.")
        self.model.fit(self.X_train, self.y_train)

    def predict(self):
        """
        Generate predictions on the test set using the trained model or pipeline.

        If a model (e.g., result of GridSearchCV) is available, it is used for prediction.
        Otherwise, the raw pipeline is used, assuming it was previously trained.

        :return: NumPy array of predicted values for X_test.
        :raises ValueError: If neither a trained model nor pipeline is available.
        """
        if self.model:
            return self.model.predict(self.X_test)
        elif self.pipeline:
            return self.pipeline.predict(self.X_test)
        logger.error("No model or pipeline found.")
        raise ValueError("Model is not trained yet")

    def evaluate(self) -> float:
        """
        Evaluate the model's performance on the test set using accuracy and classification report.

        :return: Accuracy score as a float.
        :raises ValueError: If the model or pipeline is not trained.
        """
        y_pred = self.predict()
        acc = accuracy_score(self.y_test, y_pred)
        print(f"Accuracy: {acc:.2f}")
        print("Classification Report:")
        print(classification_report(self.y_test, y_pred, digits=2))

        return acc
