import pandas as pd
import logging

from src.backend.backend_utils.combine_dfs import combine
from src.backend.registry.downloader_registry import DOWNLOADER_REGISTRY
from src.backend.registry.transformer_registry import TRANSFORMER_REGISTRY
from src.backend.cleaner.cleaner import DataCleaner
from utils.logger_config import configure_logging
from utils.data_io import save_csv, load_csv

configure_logging()
logger = logging.getLogger(__name__)


def run_pipeline(
        source_name: str,
        save: bool = False
) -> pd.DataFrame:
    """
    Run the full data preparation pipeline for a given data source.

    This function:
    - Validates that the specified source has both a downloader and transformer registered.
    - Downloads fresh raw data using the appropriate downloader.
    - Loads historical data from CSV and combines it with actual data if necessary.
    - Applies transformation and cleaning to produce a ready-to-use dataset.
    - Optionally saves the final cleaned dataset to a CSV file.

    :param source_name: Identifier for the data source (must be registered in both registries).
    :param save: Whether to save the cleaned DataFrame to a file (default is False).
    :return: A cleaned and transformed DataFrame ready for model training or analysis.
    :raises ValueError: If the specified source is not found in the downloader or transformer registries.
    """
    if source_name not in DOWNLOADER_REGISTRY:
        raise ValueError(f"No downloader found for source: {source_name}")
    if source_name not in TRANSFORMER_REGISTRY:
        raise ValueError(f"No transformer found for source: {source_name}")

    logger.info(f"[PIPELINE] Running for source='{source_name}'")

    downloader = DOWNLOADER_REGISTRY[source_name]
    transformer = TRANSFORMER_REGISTRY[source_name]

    downloader = downloader()
    transformer = transformer()
    historacal_df = load_csv(
        filedir="raw",
        filename="historical_B1_data.csv"
    )

    raw_df = downloader.fetch()
    actual_df = combine(
        actual_df=raw_df,
        historical_df=historacal_df
    )
    df_transform = transformer.transform(raw_df=actual_df)
    cleaner = DataCleaner(df=df_transform)
    df_ready = cleaner.clean_all()

    if save:
        save_csv(df=df_ready, filename=f"{source_name}.csv")

    return df_ready



