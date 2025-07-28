from dataclasses import dataclass
from typing import Optional


@dataclass
class CleaningConfig:
    """
    Configuration for all cleaning steps to be applied to a DataFrame.

    Attributes:
        drop_duplicates_subset (str | None): Column name to check for duplicates.
        drop_duplicates_strategy (str): Strategy to keep duplicates ("first", "last").
        missing_percent (int): Threshold for dropping columns based on missing percentage.
        exceptions (list[str] | None): Columns to exclude from missing percentage drop.
        to_drop (list[str] | None): Columns to explicitly drop.
        required_columns (list[str] | None): Columns that must not have missing values.
        title_case_columns (list[str] | None): Columns to apply `.title()` formatting.
        column_for_special_remove (str | None): Column name for filtering specific values.
        values_for_special_remove (list[str] | None): Values to remove from the specified column.
        rare_values_columns (list[str] | None): Columns to check for rare values.
        rare_values_min_amount (int): Minimum frequency threshold for values to be considered rare.
        rare_values_strategy (str): Strategy to handle rare values ("drop", "replace").
    """

    drop_duplicates_subset: Optional[str] = None
    drop_duplicates_strategy: str = "first"
    missing_percent: int = 60
    exceptions: Optional[list[str]] = None
    to_drop: Optional[list[str]] = None
    required_columns: Optional[list[str]] = None
    title_case_columns: Optional[list[str]] = None
    column_for_special_remove: Optional[str] = None
    values_for_special_remove: Optional[list[str]] = None
    rare_values_columns: Optional[list[str]] = None
    rare_values_min_amount: int = (20,)
    rare_values_strategy: str = "drop"
