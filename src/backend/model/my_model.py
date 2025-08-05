import pandas as pd
import logging
import sys
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, accuracy_score

# Add the project root to Python path     (For Local Testing only if using VS Code)
# project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
# sys.path.insert(0, project_root)

from utils.data_io import load_csv
from utils.logger_config import configure_logging

configure_logging()
logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    Class for building, training and tuning an ML model using a pipeline and GridSearchCV.
    The K-Nearest Neighbors (KNN) classifier is used.
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

    def apply_sample(self, sample_size: int, random_state: int = 42) -> None:
        """
        Apply a sample to the working DataFrame for faster experimentation.
        The original df remains untouched.

        :param sample_size: Number of rows to include in the sample.
        :param random_state: Random state for reproducibility.
        :raises ValueError: if sample_size is greater than or equal to the dataset size.
        """
        if sample_size >= len(self.df):
            raise ValueError(f"Sample size ({sample_size}) must be less than dataset size ({len(self.df)}).")

        self.df = self.df.sample(n=sample_size, random_state=random_state)

    def split_data(self, test_size=0.25, random_state=42) -> None:
        """
        Splitting data into training and testing samples.

        :param test_size: Proportion of test sample (default is 0.25).
        :param random_state: For reproducibility of the result (default is 42).
        :return: None
        """
        X = self.df.drop(columns=[self.target_column, "Date"])
        y = self.df[self.target_column]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

    def build_pipeline(self, model) -> None:
        """
        Building a sklearn pipeline with scaling and classification.

        :param model: Regression model to use in the pipeline.
        :raises ValueError: If model is None.
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
        Finding the best hyperparameters using GridSearchCV.

        :param model: Model to use in pipeline. Required if pipeline is not already built (default is None).
        :param param_grid: Dictionary of parameters to search through (default is None).
        :param cv: Number of folds for cross-validation (default is 2).
        :param scoring: Metric for optimization (default is "mae").
        :raises ValueError if cv param less than 2 and if model not.
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

    def evaluate(self):
        y_pred = self.predict()
        acc = accuracy_score(self.y_test, y_pred)
        print(f"Accuracy: {acc:.2f}")
        print("Classification Report:")
        print(classification_report(self.y_test, y_pred, digits=2))

        return acc


if __name__ == '__main__':
    df = load_csv(filedir="prepared", filename="B1_old.csv")
    # columns_list = [column for column in df.columns if column not in ("Date", "FTR")]
    # print(columns_list)
    #
    model_trainer = ModelTrainer(df, "FTR")
    model_trainer.split_data()
    
    model_trainer.find_best_hyperparameters(
        model=RandomForestClassifier(random_state=42, class_weight="balanced"),
        param_grid={
            "model__n_estimators": [100, 200],
            "model__max_depth": [10, 20],
            "model__min_samples_leaf": [1, 3]
        },
        scoring="f1_macro",
        cv=5
    )
    model_trainer.train()
    # save_model(model_trainer.model, "RFC_Belgium_league_model.joblib")
    model_trainer.evaluate()
    
    predictions = model_trainer.predict()
    print(predictions)
