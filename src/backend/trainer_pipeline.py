import logging
from sklearn.ensemble import RandomForestClassifier

from src.backend.model.my_model import ModelTrainer
from config import settings
from utils.data_io import load_csv, save_model
from utils.logger_config import configure_logging
from utils.dropbox_io import upload_model_to_dropbox

configure_logging()
logger = logging.getLogger(__name__)


def run_training_pipeline(source_name: str):
    # 1. Загружаем подготовленные данные

    df = load_csv(filedir="prepared", filename=f"{source_name}.csv")
    logger.info(f"Training data {source_name}.csv loaded successfully.")

    # 2.
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
    save_model(
        model=trainer.model,
        filename="trained_model.joblib"
    )

    return trainer.model


if __name__ == '__main__':
    run_training_pipeline(source_name="B1_old")
