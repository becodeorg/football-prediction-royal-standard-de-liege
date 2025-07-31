from abc import ABC, abstractmethod
import logging
import pandas as pd
from utils.logger_config import configure_logging

configure_logging()
logger = logging.getLogger(__name__)


class BaseDownloader(ABC):
    """
    Abstract base class for all data downloaders.

    Subclasses must implement the `fetch` method to return data
    as a pandas DataFrame.
    """
    @abstractmethod
    def fetch(self) -> pd.DataFrame:
        """
        Fetch data and return it as a pandas DataFrame.

        :return: The downloaded dataset (pd.DataFrame).
        :raise: NotImplementedError: If the method is not implemented in a subclass.
        """
        raise NotImplementedError("Subclasses must implement the fetch method.")
