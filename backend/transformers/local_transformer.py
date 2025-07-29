import logging
from collections import defaultdict
import pandas as pd
from .base_transformer import BaseTransformer

logger = logging.getLogger(__name__)


class LocalTransformer(BaseTransformer):
    """
    Transformer that generates match-level features from raw football match data.
    Features include recent team form, scoring/conceding averages, and target outcome.
    """
    def transform(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the full transformation pipeline on the raw match data.

        :param raw_df: Raw match data.
        :return: Transformed dataset with added features.
        """
        logger.info("Starting LocalTransformer.transform")
        df = self._prepare_dataframe(raw_df=raw_df)
        return self._build_transform_df(df=df)

    @staticmethod
    def _prepare_dataframe(raw_df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert date strings to datetime and sort by date.

        :param raw_df: Raw input data.
        :return: Sorted and date-formatted DataFrame.
        """
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
        team_history: defaultdict = defaultdict(list)
        (
            home_form_last_5,
            away_form_last_5,
            home_goals_avg,
            away_goals_avg,
            home_conceded_avg,
            away_conceded_avg,
        ) = self._calculate_form_features(df=df, team_history=team_history)

        df_transformed = pd.DataFrame({
            "date": df["Date"],
            "home_team": df["HomeTeam"],
            "away_team": df["AwayTeam"],
            "home_goals": df["FTHG"],
            "away_goals": df["FTAG"],
            "home_form_last_5": home_form_last_5,
            "away_form_last_5": away_form_last_5,
            "home_goals_avg": home_goals_avg,
            "away_goals_avg": away_goals_avg,
            "home_conceded_avg": home_conceded_avg,
            "away_conceded_avg": away_conceded_avg,
            "competition": df["Div"] if "Div" in df.columns else None,
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
        """
        if result == "H":
            return 3 if is_home else 0
        elif result == "A":
            return 0 if is_home else 3
        elif result == "D":
            return 1
        return 0

    @staticmethod
    def _compute_last_5_stats(
            history: list[tuple[int, int, int]]
    ) -> tuple[float, float, float]:
        """
        Compute average points, goals scored, and goals conceded over last 5 games.

        :param history: Team's match history as (points, scored, conceded).
        :return: Averages of last 5 games.
        """
        last_5 = history[-5:]
        if not last_5:
            return (None, None, None)
        points = [x[0] for x in last_5]
        scored = [x[1] for x in last_5]
        conceded = [x[2] for x in last_5]
        return (
            sum(points) / len(points),
            sum(scored) / len(scored),
            sum(conceded) / len(conceded),
        )

    def _calculate_form_features(
            self,
            df: pd.DataFrame,
            team_history: defaultdict
    ) -> tuple[list, list, list, list, list, list]:
        """
        Calculate historical performance features for each match.

        :param df: Match data sorted by date.
        :param team_history: Keeps track of each team's recent results.

        :return: Tuple of lists containing features for all matches.
        """

        home_form, away_form = [], []
        home_goals_avg, away_goals_avg = [], []
        home_conceded_avg, away_conceded_avg = [], []

        for _, row in df.iterrows():
            home, away = row["HomeTeam"], row["AwayTeam"]
            ftr, hg, ag = row["FTR"], row["FTHG"], row["FTAG"]

            home_stats = self._compute_last_5_stats(team_history[home])
            away_stats = self._compute_last_5_stats(team_history[away])

            home_form.append(home_stats[0])
            away_form.append(away_stats[0])
            home_goals_avg.append(home_stats[1])
            away_goals_avg.append(away_stats[1])
            home_conceded_avg.append(home_stats[2])
            away_conceded_avg.append(away_stats[2])

            team_history[home].append((self._result_to_points(ftr, True), hg, ag))
            team_history[away].append((self._result_to_points(ftr, False), ag, hg))

        logger.info("Form features calculated for all matches.")

        return (
            home_form,
            away_form,
            home_goals_avg,
            away_goals_avg,
            home_conceded_avg,
            away_conceded_avg,
        )

    @staticmethod
    def _get_season(date: pd.Timestamp) -> str:
        """
        Derive the season string (e.g. '2023-2024') from a given date.

        :param date: Match date.
        :return: Season identifier.
        """
        year = date.year
        return f"{year}-{year + 1}" if date.month >= 7 else f"{year - 1}-{year}"
