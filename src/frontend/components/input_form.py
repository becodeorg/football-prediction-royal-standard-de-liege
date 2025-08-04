from typing import Optional

import streamlit as st
import logging

from utils.data_io import load_csv

logger = logging.getLogger(__name__)


class MatchesData:
    """
    A class to handle the user input form for selecting football teams
    in a Streamlit app.

    This class loads a list of teams from a prepared CSV file, renders
    two selection boxes (home team and away team) in the UI, and provides
    methods to retrieve the selected teams.
    """
    def __init__(self):
        self.input_data = {}
        self.teams = self._get_tems_list()

    @staticmethod
    def _get_tems_list() -> list[str]:
        """
        Loads the dataset and extracts a sorted list of team names that appear
        both as home and away teams.

        :return: Sorted list of unique team names.
        """
        df = load_csv(filedir="prepared", filename="B1_old.csv")
        teams = sorted(set(df["HomeTeam"].unique()) & set(df["AwayTeam"].unique()))

        return teams

    def render(self) -> None:
        """
        Renders two Streamlit select boxes side-by-side for selecting
        the home team and away team.

        The away team list excludes the currently selected home team
        to prevent selecting the same team twice.

        :return: None
        """
        col1, col2 = st.columns(2)
        with col1:
            self.input_data['HomeTeam'] = st.selectbox(label='HomeTeam:',
                                                       options=self.teams,
                                                       index=None,
                                                       placeholder='--Select--')

        with col2:
            away_team = [team for team in self.teams if team != self.input_data["HomeTeam"]]
            self.input_data['AwayTeam'] = st.selectbox(label='AwayTeam:',
                                                       options=away_team,
                                                       index=None,
                                                       placeholder='--Select--')

    def get_selected_teams(self) -> tuple[Optional[str], Optional[str]]:
        """
        Retrieves the currently selected home and away teams.

        :return: A tuple containing the selected home team and away team names.
                 Returns (None, None) if either team has not been selected.
        """
        home = self.input_data.get('HomeTeam', "")
        away = self.input_data.get('AwayTeam', "")
        if home == "" or away == "":
            return None, None
        return home, away



