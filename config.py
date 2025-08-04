from pathlib import Path
from typing import Annotated
import logging

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from utils.logger_config import configure_logging

try:
    import streamlit as st

    IS_STREAMLIT = True
except ImportError:
    IS_STREAMLIT = False

configure_logging()
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
env_path = BASE_DIR / ".env"
load_dotenv(dotenv_path=env_path)  # load from .env файла


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables and .env file.

    Attributes:
        csv_save_load_path (Path): Path for saving/loading CSV files.
        model_save_load_path (Path): Path for saving/loading model files.
    """
    model_config = SettingsConfigDict(
        env_file=env_path,
        env_file_encoding="utf-8"
    )

    csv_save_load_path: Annotated[Path, Field(json_schema_extra={"env": "CSV_SAVE_LOAD_PATH"})]
    model_save_load_path: Annotated[Path, Field(json_schema_extra={"env": "MODEL_SAVE_LOAD_PATH"})]

    belgium_data_base_url: Annotated[str, Field(json_schema_extra={"env": "BELGIUM_DATA_BASE_URL"})]

    dropbox_access_token: Annotated[str, Field(json_schema_extra={"env": "DROPBOX_ACCESS_TOKEN"})]
    dropbox_model_path: Annotated[
        str, Field(
            default="/models/latest_model.joblib",
            json_schema_extra={"env": "DROPBOX_MODEL_PATH"}
        )
    ]

    def model_post_init(self, __context) -> None:
        """
        Post-initialization hook to resolve relative paths to absolute ones.
        """
        if not self.csv_save_load_path.is_absolute():
            self.csv_save_load_path = (BASE_DIR / self.csv_save_load_path).resolve()
            logger.debug("Resolved csv_save_load_path: %s", self.csv_save_load_path)

        if not self.model_save_load_path.is_absolute():
            self.model_save_load_path = (BASE_DIR / self.model_save_load_path).resolve()
            logger.debug("Resolved model_save_load_path: %s", self.model_save_load_path)


def get_settings() -> Settings:
    """
    Automatically select config source:
    - Streamlit Cloud (via st.secrets)
    - Else fallback to .env or system env
    :return: Settings
    """
    if IS_STREAMLIT:
        try:
            # Only attempt to access secrets if they exist
            if "CSV_SAVE_LOAD_PATH" in st.secrets:
                logger.info("Using Streamlit secrets for settings.")
                return Settings(
                    csv_save_load_path=Path(st.secrets["CSV_SAVE_LOAD_PATH"]),
                    model_save_load_path=Path(st.secrets["MODEL_SAVE_LOAD_PATH"]),
                    belgium_data_base_url=st.secrets["BELGIUM_DATA_BASE_URL"]
                )
        except Exception as e:
            logger.warning(f"Streamlit secrets not found or inaccessible: {e}")

    logger.info("Using .env or system environment for settings.")
    return Settings()


settings = get_settings()
