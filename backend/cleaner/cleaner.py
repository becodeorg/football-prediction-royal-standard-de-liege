import logging
import unicodedata
from typing import Optional
import numpy as np
import pandas as pd
from backend.cleaner.cleaning_config import CleaningConfig
from backend.cleaner.cleaning_config import my_config

logger = logging.getLogger(__name__)


class DataCleaner:
    """
    A class for loading, cleaning, and saving tabular data from CSV files.
    """

    def __init__(self, df: pd.DataFrame):
        """
        Initializes the object with a DataFrame.

        :param df: The DataFrame to be stored in the object.
        """
        self.df = df

    def remove_duplicates(
            self,
            subset: Optional[str] = None,
            keep: str = "first"
    ) -> None:
        """
        Removes duplicate rows from the DataFrame.

        :param subset: Name of columns to check for duplicates.
        If None, checks all columns (default is None).
        :param keep: Which duplicate to keep: "first", "last", or False:
        removes all duplicates (default is "first").
        :return: None
        """
        self.df = self.df.drop_duplicates(subset=subset, keep=keep)

    def remove_columns_by_missing_percentage(
            self,
            percent: int = 60,
            exceptions: Optional[list[str]] = None,
    ) -> None:
        """
        Removes columns with missing values exceeding the specified percentage.

        :param percent: The percentage threshold of missing values. Columns with missing values greater than or
        equal to this percentage will be dropped (default is 60%).
        :param exceptions: A list of column names that should not be dropped, even if they have
        a high percentage of missing values.
        :return: None
        """
        if exceptions is None:
            exceptions = []

        missing_percentage = self.get_missing_percentage()

        columns_to_drop = [
            col
            for col, perc in missing_percentage.items()
            if perc >= percent and col not in exceptions
        ]

        self.df = self.df.drop(columns=columns_to_drop)

        logger.info(
            f"Dropped columns due to missing percentage >= {percent}%: {columns_to_drop}, except: {exceptions}"
        )

    def drop_columns(
            self,
            columns: Optional[list[str]] = None
    ) -> None:
        """
        Drop specified columns from the DataFrame.

        :param columns: A list of columns to drop (default is None).
        :return: None
        """
        if columns:
            self.df = self.df.drop(columns=columns)

    def drop_rows_with_missing_value(
            self,
            required_columns: Optional[list[str]] = None
    ) -> None:
        """
        Drop rows where any of the required columns have missing values.

        :param required_columns: A list of columns where to check missing values.
        :return:
        """
        if required_columns:
            self.df = self.df.dropna(subset=required_columns)

    def get_missing_percentage(self) -> pd.Series:
        """
        Return the percentage of missing values per column.
        :return: pd.Index: The percentage of missing values for each column.
        """
        missing_percentage = (self.df.isnull().sum() / len(self.df)) * 100

        return missing_percentage

    def normalize_text_columns(
            self,
            title_case_columns: Optional[list[str]] = None
    ) -> None:
        """
        Cleans string columns by:
        - Removing leading/trailing spaces
        - Converting to lowercase
        - Replacing underscores with spaces
        - Removing accents (e.g., 'Liège' → 'Liege')
        - Applying .capitalize() to all
        - Applying .title() to selected columns

        :param title_case_columns: list of column names to apply .title() formatting
        """
        title_case_columns = title_case_columns or []

        for col in self.df.columns:
            if self.df[col].dtype == "object":
                self.df[col] = self.df[col].apply(
                    lambda x: (
                        self._clean_string(
                            x, use_title_case=(col in title_case_columns)
                        )
                        if isinstance(x, str) and not x.startswith("http")
                        else x
                    )
                )

    @staticmethod
    def _clean_string(
            text: str,
            use_title_case: bool = False
    ) -> str:
        """
        Cleans and normalizes a string by applying the following transformations:
        - Strips leading and trailing whitespace
        - Converts to lowercase
        - Replaces underscores with spaces
        - Removes accented characters (e.g., 'Liège' → 'Liege')
        - Capitalizes the string or applies title case if specified

        :param text: The input string to clean.
        :param use_title_case: If True, capitalizes the first letter of each word (title case).
                           If False, only the first letter of the string is capitalized.
        :return: The cleaned and normalized string.
        """
        text = text.strip().lower().replace("_", " ")
        # removing accents, e.g. 'Liège' → 'Liege'
        text = (
            unicodedata.normalize("NFKD", text)
            .encode("ASCII", "ignore")
            .decode("utf-8")
        )

        return text.title() if use_title_case else text.capitalize()

    def remove_by_column_values(
            self, column: Optional[str] = None,
            value: Optional[list[str]] = None
    ) -> None:
        """
        Removes rows where the specified column has values in the given list.

        :param column: Column to filter.
        :param value: List of values to remove.
        :return: None
        """
        if column is None or value is None:
            return

        value_str = ", ".join(f"'{elem}'" for elem in value)
        self.df = self.df.query(f"{column} not in [{value_str}]")

    def replace_rare_values(
            self,
            columns: Optional[list[str]] = None,
            min_amount: int = 20,
            strategy: str = "drop"
    ) -> None:
        """
        Replaces rare categories in the specified column with replacement parameter.

        :param columns: Columns to process.
        :param min_amount: Minimum count to keep category (default is 20).
        :param strategy: "replace" to replace with NA, "drop" to drop rows.
        :return: None
        """
        if columns is None:
            return
        for column in columns:
            value_counts = self.df[column].value_counts()
            rare_values = value_counts[value_counts < min_amount].index

            if strategy == "replace":
                if pd.api.types.is_numeric_dtype(self.df[column]):
                    # Use np.nan for numeric columns
                    self.df[column] = self.df[column].apply(
                        lambda x: np.nan if x in rare_values else x
                    )
                elif pd.api.types.is_object_dtype(
                        self.df[column]
                ) or pd.api.types.is_categorical_dtype(self.df[column]):
                    # Use pd.NA for categorical/text columns
                    self.df[column] = self.df[column].apply(
                        lambda x: pd.NA if x in rare_values else x
                    )
            elif strategy == "drop":
                initial_shape = self.df.shape
                self.df = self.df[~self.df[column].isin(rare_values)]

                logger.info(f"Dropped {initial_shape[0] - self.df.shape[0]} rows due to rare values in '{column}'")
            else:
                logger.error(f"Invalid strategy passed to replace_rare_values: '{strategy}'")
                raise ValueError("Strategy must be 'replace' or 'drop'")

    def handle_errors(self) -> None:
        """
        Handles erroneous values in numeric columns, replacing them with NaN

        :return: None
        """
        numeric_cols = self.df.select_dtypes(include=["number"]).columns

        for col in numeric_cols:
            self.df[col] = pd.to_numeric(self.df[col], errors="coerce")

    def clean_all(self, config: CleaningConfig = my_config) -> pd.DataFrame:
        """
        Perform all cleaning steps on the dataframe using a configuration object.

        :return: Cleaned pandas DataFrame.
        """

        logger.info("Starting full data cleaning process")

        self.remove_duplicates(
            subset=config.drop_duplicates_subset, keep=config.drop_duplicates_strategy
        )
        self.remove_columns_by_missing_percentage(
            percent=config.missing_percent, exceptions=config.exceptions
        )
        self.drop_columns(columns=config.to_drop)
        self.drop_rows_with_missing_value(required_columns=config.required_columns)
        self.normalize_text_columns(title_case_columns=config.title_case_columns)
        self.remove_by_column_values(
            column=config.column_for_special_remove,
            value=config.values_for_special_remove,
        )
        self.replace_rare_values(
            columns=config.rare_values_columns,
            min_amount=config.rare_values_min_amount,
            strategy=config.rare_values_strategy,
        )
        self.handle_errors()

        logger.info("Finished data cleaning")

        return self.df
