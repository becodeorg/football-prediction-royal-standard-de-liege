from pathlib import Path
from typing import Annotated

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings

env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)  # load from .env файла


class Settings(BaseSettings):
    csv_save_load_path: Annotated[Path, Field(env="CSV_SAVE_LOAD_PATH")]
    model_save_load_path: Annotated[Path, Field(env="MODEL_SAVE_LOAD_PATH")]

    sportmonks_api_key: Annotated[str, Field(env="SPORTMONKS_API_KEY")]

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
