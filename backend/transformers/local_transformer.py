import logging
from typing import Optional
from collections import defaultdict
from dataclasses import dataclass, field
import pandas as pd
from .base_transformer import BaseTransformer

logger = logging.getLogger(__name__)

# Internal constant mapping result + home/away to points
RESULT_POINTS = {
    ("H", True): 3,
    ("H", False): 0,
    ("A", True): 0,
    ("A", False): 3,
    ("D", True): 1,
    ("D", False): 1,
}


@dataclass
class MatchStat:
    """
    Represents the statistics of a single match for a team.

    Attributes:
        points: Points earned in the match (e.g., 3 for win, 1 for draw, 0 for loss).
        scored: Number of goals scored by the team.
        conceded: Number of goals conceded by the team.
    """
    points: int
    scored: int
    conceded: int


@dataclass
class StatsSummary:
    """
    Summary of average statistics calculated over a set of matches.

    Attributes:
        avg_points: Average points earned across matches; None if no data.
        avg_scored: Average goals scored per match; None if no data.
        avg_conceded: Average goals conceded per match; None if no data.
    """
    avg_points: Optional[float]
    avg_scored: Optional[float]
    avg_conceded: Optional[float]


@dataclass
class FormFeatures:
    """
    Aggregated lists of form-related features over multiple matches for a team.

    Attributes:
        form_last_5: List of average points from last 5 matches per game.
        goals_avg: List of average goals scored from last 5 matches per game.
        conceded_avg: List of average goals conceded from last 5 matches per game.
    """
    form_last_5: list[Optional[float]] = field(default_factory=list)
    goals_avg: list[Optional[float]] = field(default_factory=list)
    conceded_avg: list[Optional[float]] = field(default_factory=list)


class LocalTransformer(BaseTransformer):
    """
    Transformer that generates match-level features from raw football match data.
    Features include recent team form, scoring/conceding averages, and target outcome.

    Expects input DataFrame to contain the following columns:
        - Date
        - HomeTeam
        - AwayTeam
        - FTHG (full-time home goals)
        - FTAG (full-time away goals)
        - FTR (full-time result: 'H', 'D', or 'A')
    """

    REQUIRED_COLUMNS = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR", "Div"]

    def transform(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the full transformation pipeline on the raw match data.

        :param raw_df: Raw match data.
        :return: Transformed dataset with added features.
        """
        if raw_df.empty:
            logger.error("Input DataFrame is empty")
            raise ValueError("Input DataFrame cannot be empty")

        logger.info("Starting LocalTransformer.transform")

        df = self._prepare_dataframe(raw_df=raw_df)
        df_transformed = self._build_transform_df(df=df)

        return df_transformed

    @staticmethod
    def _check_required_columns(df: pd.DataFrame) -> None:
        """
        Check if the DataFrame contains all required columns.

        :param df: DataFrame to check.
        :raises KeyError: If any required column is missing.
        """

        required_cols = LocalTransformer.REQUIRED_COLUMNS

        for col in required_cols:
            if col not in df.columns:
                logger.error(f"Missing required column '{col}' in data frame")
                raise KeyError(f"Column '{col}' is required for transformation")

    def _prepare_dataframe(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert date strings to datetime and sort by date.

        :param raw_df: Raw input data.
        :return: Sorted and date-formatted DataFrame.
        """

        self._check_required_columns(raw_df)

        df = raw_df.copy()
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date").reset_index(drop=True)

        return df

    def _build_transform_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Build a transformed DataFrame with engineered features.

        :param df: Preprocessed match data.
        :return: Feature-rich transformed data.
        """
        self._check_required_columns(df)

        team_history: defaultdict = defaultdict(list)
        home_features, away_features = self._calculate_form_features(
            df=df,
            team_history=team_history
        )

        df_transformed = pd.DataFrame({
            "date": df["Date"],
            "home_team": df["HomeTeam"],
            "away_team": df["AwayTeam"],
            "home_goals": df["FTHG"],
            "away_goals": df["FTAG"],
            "home_form_last_5": home_features.form_last_5,
            "away_form_last_5": away_features.form_last_5,
            "home_goals_avg": home_features.goals_avg,
            "away_goals_avg": away_features.goals_avg,
            "home_conceded_avg": home_features.conceded_avg,
            "away_conceded_avg": away_features.conceded_avg,
            "competition": df["Div"],
            "season": df["Date"].apply(self._get_season),
            "target": df["FTR"].map({"H": 1, "D": 0, "A": -1}),
        })

        logger.info("Transformed DataFrame successfully built.")

        return df_transformed

    @staticmethod
    def _result_to_points(result: str, is_home: bool) -> int:
        """
        Convert match result into points for home/away team.

        :param result: Match result, one of {"H", "D", "A"}.
        :param is_home: Whether the team is home.
        :return: Points earned.
        :raises ValueError: If result is not one of the expected values.
        """
        if result not in {"H", "D", "A"}:
            logger.error(f"Invalid match result encountered: {result}")
            raise ValueError(f"Invalid match result: {result}. Expected one of 'H', 'D', 'A'.")

        return RESULT_POINTS.get((result, is_home), 0)

    @staticmethod
    def _compute_last_5_games_stats(
            history: list[MatchStat]
    ) -> StatsSummary:
        """
        Compute average points, goals scored, and goals conceded over last 5 games.

        :param history: Team's match history as (points, scored, conceded).
        :return: Averages of last 5 games.
        """
        last_5 = history[-5:]

        if not last_5:
            return StatsSummary(None, None, None)

        points = [x.points for x in last_5]
        scored = [x.scored for x in last_5]
        conceded = [x.conceded for x in last_5]

        return StatsSummary(
            avg_points=sum(points) / len(points),
            avg_scored=sum(scored) / len(scored),
            avg_conceded=sum(conceded) / len(conceded),
        )

    def _calculate_form_features(
            self,
            df: pd.DataFrame,
            team_history: defaultdict[pd.Series, list[MatchStat]]
    ) -> tuple[FormFeatures, FormFeatures]:
        """
        Calculate historical performance features for each match.

        :param df: Match data sorted by date.
        :param team_history: Keeps track of each team's recent results.

        :return: Tuple of lists containing features for all matches.
        """

        home_features = FormFeatures()
        away_features = FormFeatures()

        for _, row in df.iterrows():
            home = row["HomeTeam"]
            away = row["AwayTeam"]
            match_result = row["FTR"]
            home_goals = row["FTHG"]
            away_goals = row["FTAG"]

            home_stats = self._compute_last_5_games_stats(team_history[home])
            away_stats = self._compute_last_5_games_stats(team_history[away])

            home_features.form_last_5.append(home_stats.avg_points)
            away_features.form_last_5.append(away_stats.avg_points)

            home_features.goals_avg.append(home_stats.avg_scored)
            away_features.goals_avg.append(away_stats.avg_scored)

            home_features.conceded_avg.append(home_stats.avg_conceded)
            away_features.conceded_avg.append(away_stats.avg_conceded)

            self._update_team_history(
                team_history=team_history,
                team=home,
                points=self._result_to_points(match_result, True),
                scored=home_goals,
                conceded=away_goals
            )
            self._update_team_history(
                team_history=team_history,
                team=away,
                points=self._result_to_points(match_result, False),
                scored=away_goals,
                conceded=home_goals
            )

        logger.info("Form features calculated for all matches.")

        return home_features, away_features

    @staticmethod
    def _update_team_history(team_history, team, points, scored, conceded):
        team_history[team].append(MatchStat(
            points=points,
            scored=scored,
            conceded=conceded
        ))

    @staticmethod
    def _get_season(date: pd.Timestamp) -> str:
        """
        Derive the season string (e.g. '2023-2024') from a given date.

        :param date: Match date.
        :return: Season identifier.
        """
        year = date.year
        return f"{year}-{year + 1}" if date.month >= 7 else f"{year - 1}-{year}"
