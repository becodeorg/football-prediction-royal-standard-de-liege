import logging
from urllib.parse import urljoin
import requests
from datetime import datetime as dt
import time

from bs4 import BeautifulSoup
import pandas as pd

from src.backend.downloaders.base_downloader import BaseDownloader
from config import settings

logger = logging.getLogger(__name__)


class BelgiumLeagueDownloader(BaseDownloader):
    """
    Downloads the current season's Belgian Pro League data (B1.csv)
    from football-data.co.uk by parsing the site and determining the correct URL.
    """
    _base_url = settings.belgium_data_base_url
    _page_path = "belgiumm.php"
    _league_code = "B1.csv"
    _data_folder_prefix = "mmz4281/"

    def __init__(self):
        self._session = requests.Session()

    @property
    def page_url(self):
        return urljoin(self._base_url, self._page_path)

    def fetch(self) -> pd.DataFrame:
        """
        Download the CSV data and return it as a DataFrame.

        :return:Raw dataset loaded from football.co.uk (pd.DataFrame)
        """
        csv_url = self._find_csv_url()

        logger.info(f"Downloading csv from: {csv_url}")
        df = pd.read_csv(csv_url)
        logger.info(f"DataFrame loaded: {df.shape[0]} rows × {df.shape[1]} columns.")

        return df

    @staticmethod
    def _get_current_season_code() -> str:
        """
        Return current season code in format YYZZ, e.g. '2425' for 2024-2025.

        :return: season code (str)
        """
        now = dt.now()
        start_year = now.year if now.month >= 7 else now.year - 1
        end_year = start_year + 1

        season_code = f"{str(start_year)[-2:]}{str(end_year)[-2:]}"

        return season_code

    def _find_csv_url(self) -> str:
        """
        Download the HTML index page and extract the CSV URL
        for the current season and league.

        :return: full URL string to the CSV file
        :raises FileNotFoundError: If the CSV link is not found in the page content.
        :raises requests.RequestException: If the index page cannot be downloaded.
        """
        html = self._download_index_page()
        full_url = self._extract_csv_url_from_html(html)

        return full_url

    def _download_index_page(self, max_retries: int = 3, delay: float = 1.0) -> str:
        """

        :param max_retries: Number of retry attempts.
        :param delay: Initial delay between retries in seconds.
        :return: HTML content as a string.
        :raises requests.RequestException: If all attempts fail.
        """
        logger.info(f"Requesting index page: {self.page_url}")

        for attempt in range(1, max_retries + 1):
            try:
                response = self._session.get(self.page_url, timeout=1)
                response.raise_for_status()
                return response.text
            except requests.RequestException as e:
                logger.warning(f"Attempt {attempt} failed: {e}")
                if attempt == max_retries:
                    logger.error("All retry attempts failed.")
                    raise
                time.sleep(delay)

    def _extract_csv_url_from_html(self, html: str) -> str:
        """
        Parse the provided HTML content and extract the full CSV URL
        for the current season and league.

        :param html: Raw HTML content of the index page.
        :return: Full URL to the CSV file.
        :raises FileNotFoundError: If the target CSV link is not found in the HTML.
        """
        soup = BeautifulSoup(html, "html.parser")
        season_code = self._get_current_season_code()
        target_path = f"{self._data_folder_prefix}{season_code}/{self._league_code}"

        for link in soup.find_all("a", href=True):
            href = link.get("href", "")

            if target_path in href:
                full_url = urljoin(self._base_url, href)
                logger.info(f"Found scv for league: {self._league_code}, season: {season_code}")

                return full_url

        logger.error(f"CSV for season {season_code} and league {self._league_code} not found.")
        raise FileNotFoundError(f"CSV for season {season_code} and league {self._league_code} not found.")


