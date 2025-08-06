import logging
import pandas as pd

from utils.logger_config import configure_logging

configure_logging()
logger = logging.getLogger(__name__)


def combine(
        actual_df: pd.DataFrame,
        historical_df: pd.DataFrame,
        treshhold: int = 100
) -> pd.DataFrame:
    """
    Combine actual and historical data if actual data is below a certain threshold.

    This function checks the number of rows in `actual_df`. If it's less than or equal to
    the threshold, it concatenates `actual_df` with `historical_df`. Otherwise, only
    `actual_df` is returned.

    :param actual_df: The most recent data to use (e.g. fresh matchday data).
    :param historical_df: Previously stored data to fall back on if `actual_df` is too small.
    :param treshhold: Minimum number of rows required in `actual_df` to use it as-is (default is 100).
    :return: A combined or original DataFrame depending on the threshold condition.
    """
    if len(actual_df) <= treshhold:
        new_df = pd.concat([actual_df, historical_df], ignore_index=True)

        logger.info(f"Create a new data frame using historical data")
        return new_df

    logger.info(f"Using actual data frame")
    return actual_df
