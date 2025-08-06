import logging

from sklearn.ensemble import RandomForestClassifier

from src.backend.model.ml_model import ModelTrainer
from utils.data_io import load_csv, save_model
from utils.logger_config import configure_logging

configure_logging()
logger = logging.getLogger(__name__)


def run_training_pipeline(source_name: str, save: bool=False):
    """
    Run the end-to-end training pipeline for a given prepared dataset.

    This function:
    - Loads a preprocessed dataset from disk.
    - Initializes the ModelTrainer with the target column 'FTR'.
    - Splits the data into training and test sets.
    - Performs hyperparameter tuning using GridSearchCV with a RandomForestClassifier.
    - Trains the best model found.
    - Optionally saves the trained model to disk.

    :param source_name: The name of the prepared dataset (without extension).
    :param save: Whether to save the trained model to a file (default is False).
    :return: The trained model instance (e.g., a fitted RandomForestClassifier).
    :raises FileNotFoundError: If the specified dataset file does not exist.
    :raises ValueError: If training fails due to invalid parameters or missing data.

    """
    df = load_csv(filedir="prepared", filename=f"{source_name}.csv")
    logger.info(f"Training data {source_name}.csv loaded successfully.")

    trainer = ModelTrainer(df, "FTR")
    trainer.split_data()

    trainer.find_best_hyperparameters(
        model=RandomForestClassifier(random_state=42, class_weight="balanced"),
        param_grid={
            "model__n_estimators": [100, 200],
            "model__max_depth": [10, 20],
            "model__min_samples_leaf": [1, 3]
        },
        scoring="f1_macro",
        cv=5
    )
    trainer.train()
    logger.info("Model training completed.")

    if save:
        save_model(model=trainer.model, filename="trained_model.joblib")

    return trainer.model


