import logging

from utils.data_io import load_csv

logger = logging.getLogger(__name__)


class FeaturePrepare:
    """
    Prepares features for the football match prediction model.

    Loads the dataset with match data and provides
    a method to extract features for a specific match given
    the home and away teams.
    """
    def __init__(self):
        self.df = load_csv(filedir="prepared", filename="Belgium_league_2526.csv")
        self.feature_cols = self._get_feature_cols()

    def _get_feature_cols(self) -> list[str]:
        """
        Identifies and returns the list of feature column names
        to be used for prediction, excluding non-feature columns.

        :return: List of feature column names.
        """
        feature_cols = [
            column for column in self.df.columns
            if column not in ("Date", "FTR")
        ]

        return feature_cols

    def prepare_features(self, home_team: str, away_team: str) -> dict[str, float]:
        """
        Extracts the features for the given home and away teams from
        the dataset to use as input for the prediction model.

        :param home_team: Name of the home team
        :param away_team: Name of the away team.
        :return: A dictionary mapping feature names to their values.
        :raises: ValueError: If no matching row is found for the given teams.
        """
        match_row = self.df[
            (self.df["HomeTeam"] == home_team) & (self.df["AwayTeam"] == away_team)
            ]
        if match_row.empty:
            logger.error(f"No data found for match {home_team} vs {away_team}")
            raise ValueError(f"No data found for match {home_team} vs {away_team}")

        features = match_row.iloc[0][self.feature_cols].to_dict()

        return features