import logging
from pathlib import Path

import joblib
import pandas as pd

from config import settings

logger = logging.getLogger(__name__)


def load_csv(
        filedir: str,
        filename: str,
        sep: str = ",",
        encoding: str = "utf-8"
) -> pd.DataFrame:
    """
    Load a CSV file from the default directory defined in the .env file (CSV_SAVE_LOAD_PATH).

    :param filename: Name of the CSV file (not full path).
    :param sep: Column separator in the CSV file (default is ',').
    :param encoding: File encoding (default is 'utf-8').
    :return: DataFrame loaded from the CSV file.
    :raises FileNotFoundError: If the file is not found.
    :raises pd.errors.ParserError: If the file cannot be parsed as CSV.
    :raises UnicodeError: If encoding fails.
    :raises Exception: For any other unexpected error.
    """
    filepath = Path(settings.csv_save_load_path / filedir / filename)

    try:
        df = pd.read_csv(filepath, sep=sep, encoding=encoding)
        logger.info(f"CSV file loaded successfully: {filepath}")
        return df
    except FileNotFoundError:
        logger.error(f"File not found: {filepath}")
        raise
    except pd.errors.ParserError:
        logger.error(f"Parser error reading CSV: {filepath}")
        raise
    except UnicodeError:
        logger.error(f"Encoding error in file: {filepath}")
        raise
    except Exception as e:
        logger.exception(f"Unexpected error loading file '{filepath}': {e}")
        raise


def save_csv(
        df: pd.DataFrame,
        filename: str,
        mode: str = "w",
        sep: str = ",",
        encoding: str = "utf-8",
) -> None:
    """
    Saves a DataFrame to a CSV file at the path defined in .env (CSV_SAVE_LOAD_PATH).

    :param df: DataFrame to save.
    :param filename: Name of the file to be saved (e.g., 'cleaned.csv').
    :param mode: Write mode: 'w' (overwrite) or 'a' (append). Default is 'w'.
    :param sep: Delimiter used in the CSV. Default is ','.
    :param encoding: Encoding format. Default is 'utf-8'.
    """
    filedir = "prepared"
    filepath = Path(settings.csv_save_load_path / filedir / filename)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    try:
        df.to_csv(filepath, mode=mode, sep=sep, encoding=encoding, index=False)
        logger.info(f"File saved: {filepath.resolve()}")
    except FileNotFoundError:
        logger.error(f"Write path doesn't exist: {filepath}")
        raise FileNotFoundError(f"Write path doesn't exist: {filepath}")
    except PermissionError:
        logger.error(f"No permission to write file: {filepath}")
        raise PermissionError(f"No permission to write to file: {filepath}")
    except Exception as e:
        logger.exception(f"Unknown error writing file '{filepath}': {e}")
        raise Exception(f"Unknown error writing file {filepath}: {e}")


def save_model(model, filename: str) -> None:
    """
    Saves a trained model to a file in the path defined in .env (MODEL_SAVE_LOAD_PATH).

    :param model: The trained model or pipeline to save.
    :param filename: Name of the output file (default is 'pipeline.pkl').
    """
    if model is None:
        logger.error("Model is None and cannot be saved.")
        raise ValueError("Model is None and cannot be saved.")

    save_path = settings.model_save_load_path / filename
    save_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        joblib.dump(model, save_path)
        logger.info(f"Model saved to: {save_path.resolve()}")
    except PermissionError:
        logger.error(f"No permission to write model to: {save_path}")
        raise PermissionError(f"No permission to write model to: {save_path}")
    except Exception as e:
        logger.exception(f"Failed to save model to {save_path}: {e}")
        raise Exception(f"Failed to save model to {save_path}: {e}")


def load_model(filename: str):
    """
    Loads a trained model from a file in the path defined in .env (MODEL_SAVE_LOAD_PATH).

    :param filename: Name of the file to load (default is 'pipeline.pkl').
    :return: Loaded model.
    """
    load_path = settings.model_save_load_path / filename

    try:
        model = joblib.load(load_path)
        logger.info(f"Model loaded from: {load_path.resolve()}")
        return model
    except FileNotFoundError:
        logger.error(f"Model file not found: {load_path}")
        raise FileNotFoundError(f"Model file not found: {load_path}")
    except PermissionError:
        logger.error(f"No permission to read model from: {load_path}")
        raise PermissionError(f"No permission to read model from: {load_path}")
    except Exception as e:
        logger.exception(f"Failed to load model from {load_path}: {e}")
        raise Exception(f"Failed to load model from {load_path}: {e}")
