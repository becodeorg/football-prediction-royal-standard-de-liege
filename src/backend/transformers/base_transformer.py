from abc import ABC, abstractmethod
import logging
import pandas as pd
from utils.logger_config import configure_logging

configure_logging()
logger = logging.getLogger(__name__)




class BaseTransformer(ABC):
    """
    Abstract base class for data transformers.

    Any subclass must implement the `transform` method which takes a raw
    pandas DataFrame and returns a transformed DataFrame.
    """
    @abstractmethod
    def transform(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the input raw DataFrame into a processed format.

        :param raw_df: Raw input data (pd.DataFrame).
        :return: pd.DataFrame: Transformed data.
        :raises: NotImplementedError: If the method is not implemented in a subclass.
        """
        raise NotImplementedError("Subclasses must implement the transform method.")
