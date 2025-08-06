import logging

import pandas as pd

from src.backend.downloaders.base_downloader import BaseDownloader
from utils.data_io import load_csv

logger = logging.getLogger(__name__)


class LocalDownloader(BaseDownloader):
    """
    Downloader that loads data from a local CSV file.
    """

    def fetch(self) -> pd.DataFrame:
        """
        Load and return the dataset from a local CSV file.

        :return: Raw dataset loaded from CSV (pd.DataFrame).
        """
        df_raw = load_csv(filedir="raw", filename="historical_B1_data.csv")

        return df_raw
