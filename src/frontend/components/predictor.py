from typing import Optional
import streamlit as st
import pandas as pd
import logging

from src.frontend.components.input_form import MatchesData
from src.frontend.components.feature_prepare import FeaturePrepare

logger = logging.getLogger(__name__)


class MatchPrediction:
    """
    Handles the logic for making a football match prediction.

    This includes:
    - Collecting user input (selected teams),
    - Preparing the features required by the prediction model,
    - Making a prediction,
    - Interpreting the result as a readable string.
    """
    def __init__(self, model):
        self.model = model
        self.input_form = MatchesData()
        self.feature_preparer = FeaturePrepare()

    def handle_prediction(self) -> Optional[str]:
        """
        Executes the full prediction pipeline:

        - Retrieves selected home and away teams from the input form.
        - Validates that both teams are selected.
        - Prepares model input features.
        - Performs prediction using the trained model.
        - Converts the numerical result into a human-readable match outcome.

        :return: A string representing the result:
                 - "HomeTeam wins"
                 - "AwayTeam wins"
                 - "Draw"
                 None: If teams are not selected or feature preparation fails.
        """
        home_team, away_team = self.input_form.get_selected_teams()

        if not home_team or not away_team:
            st.warning("Please select both teams")
            return

        try:
            features = self.feature_preparer.prepare_features(
                home_team=home_team,
                away_team=away_team
            )
        except ValueError as e:
            logger.warning(f"Feature preparation failed: {e}")
            st.error("Data for one or both teams not found in dataset.")
            return

        # Create a single-row DataFrame for prediction
        input_df = pd.DataFrame([features])
        # Predict the match result
        prediction = self.model.predict(input_df)[0]

        result = self._interpret_result(prediction, home_team, away_team)

        return result

    @staticmethod
    def _interpret_result(prediction: int, home_team: str, away_team: str) -> str:
        """
        Interprets the numeric prediction output.

        :param prediction: The model's output (1, -1, or 0).
        :param home_team: Name of the home team.
        :param away_team: Name of the away team.
        :return: A human-readable match outcome.
        """
        if prediction == 1:
            return f"{home_team} wins"
        elif prediction == -1:
            return f"{away_team} wins"
        else:
            return "Draw"
