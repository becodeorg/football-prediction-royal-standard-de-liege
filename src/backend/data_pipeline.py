import pandas as pd
import logging
import sys
import os

# Add the project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.backend.registry.downloader_registry import DOWNLOADER_REGISTRY
from src.backend.registry.transformer_registry import TRANSFORMER_REGISTRY
from src.backend.cleaner.cleaner import DataCleaner
from utils.logger_config import configure_logging
from utils.data_io import save_csv

configure_logging()
logger = logging.getLogger(__name__)


def run_pipeline(
        source_name: str,
        save: bool = False
) -> pd.DataFrame:
    if source_name not in DOWNLOADER_REGISTRY:
        raise ValueError(f"No downloader found for source: {source_name}")
    if source_name not in TRANSFORMER_REGISTRY:
        raise ValueError(f"No transformer found for source: {source_name}")

    logger.info(f"[PIPELINE] Running for source='{source_name}'")

    downloader = DOWNLOADER_REGISTRY[source_name]
    transformer = TRANSFORMER_REGISTRY[source_name]

    downloader = downloader()
    transformer = transformer()

    df_raw = downloader.fetch()
    df_transform = transformer.transform(df_raw)
    cleaner = DataCleaner(df=df_transform)
    df_ready = cleaner.clean_all()

    if save:
        save_csv(df=df_ready, filename=f"{source_name}.csv")

    return df_ready


if __name__ == '__main__':
    run_pipeline(source_name="Belgium_league_2526")
