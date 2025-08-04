<<<<<<< Updated upstream
"""
Football Prediction App - Clean Architecture Version
A streamlit application for predicting Belgian Pro League matches.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from app_styles import AppStyle

# -------------------- Configuration --------------------
src_path = os.path.abspath(os.path.dirname(__file__) + "/..")
sys.path.insert(0, src_path)

# Data configuration
DATA_PATH = r"C:\\Users\\Administrateur\\Documents\\PythonBecode\\Jup En l'Air\\JPL Structured\\football-prediction-royal-standard-de-liege\\data\\raw\\dataset_old_2.csv"

# Team configuration
JUPILER_TEAMS = [
    "Anderlecht", "Club Brugge", "Standard", "Genk",
    "Gent", "Charleroi", "Cercle Brugge", "Antwerp", 
    "Westerlo", "OH Leuven", "Union Saint-Gilloise", "Mechelen",
    "Eupen", "Kortrijk", "RWD Molenbeek", "St Truiden"
]

# -------------------- Clean Data Loading --------------------
@st.cache_data
def load_data():
    """Load and preprocess the football dataset"""
    try:
        df = pd.read_csv(DATA_PATH)
        df['Year'] = pd.to_datetime(df['Date'], errors='coerce').dt.year
        raw_years = sorted(df['Year'].dropna().unique(), reverse=True)
        season_labels = [f"Season {int(y)}-{int(y)+1}" for y in raw_years]
        season_labels.insert(0, "All seasons")
        return df, raw_years, season_labels
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame(), [], ["All seasons"]

# -------------------- Clean Model Loading --------------------
@st.cache_resource
def load_predictor():
    """Load and initialize the prediction model"""
    from backend.model.simple_predictor_forest import SimplePredictorForest
    
    df = pd.read_csv(DATA_PATH)
    predictor = SimplePredictorForest()
    predictor.train(df)
    
    return predictor

# -------------------- UI Components --------------------
def create_team_selector():
    """Create the team selection interface"""
    st.markdown("Select two teams to get a match prediction.")
    
    col_team1, col_vs, col_team2 = st.columns([4, 1, 4])
    
    with col_team1:
        home_team = st.selectbox("Home Team", JUPILER_TEAMS, key="home")
    
    with col_vs:
        st.markdown("<h3 style='text-align: center;'>VS</h3>", unsafe_allow_html=True)
    
    with col_team2:
        away_options = [team for team in JUPILER_TEAMS if team != home_team]
        away_team = st.selectbox("Away Team", away_options, key="away")
    
    return home_team, away_team

def predict_match(predictor, home_team, away_team):
    """Simple prediction function"""
    input_data = pd.DataFrame({
        'HomeTeam': [home_team],
        'AwayTeam': [away_team]
    })
    
    result = predictor.predict(input_data)[0]
    
    if result == 'H':
        return f"{home_team} wins"
    elif result == 'A':
        return f"{away_team} wins"
    else:
        return "Draw"

def get_team_stats(team, df):
    """Extract basic team statistics"""
    try:
        # Filter matches where team played at home or away
        home_matches = df[df['HomeTeam'] == team] if 'HomeTeam' in df.columns else pd.DataFrame()
        away_matches = df[df['AwayTeam'] == team] if 'AwayTeam' in df.columns else pd.DataFrame()
        
        total_matches = len(home_matches) + len(away_matches)
        
        # Calculate basic stats
        if 'FTHG' in df.columns and 'FTAG' in df.columns:
            goals_scored = home_matches['FTHG'].sum() + away_matches['FTAG'].sum()
            goals_conceded = home_matches['FTAG'].sum() + away_matches['FTHG'].sum()
        else:
            goals_scored = goals_conceded = 0
            
        if 'FTR' in df.columns:
            wins = (home_matches['FTR'] == 'H').sum() + (away_matches['FTR'] == 'A').sum()
            draws = (home_matches['FTR'] == 'D').sum() + (away_matches['FTR'] == 'D').sum()
        else:
            wins = draws = 0
            
        losses = total_matches - wins - draws
        
        return {
            "Matches Played": total_matches,
            "Wins": wins,
            "Draws": draws,
            "Losses": losses,
            "Goals Scored": goals_scored,
            "Goals Conceded": goals_conceded
        }
        
    except Exception as e:
        st.error(f"Error calculating stats for {team}: {e}")
        return {
            "Matches Played": 0,
            "Wins": 0,
            "Draws": 0,
            "Losses": 0,
            "Goals Scored": 0,
            "Goals Conceded": 0
        }

def display_team_comparison(home_team, away_team, df):
    """Display team statistics comparison"""
    st.subheader("Team Statistics Comparison")
    
    # Get stats for both teams
    home_stats = get_team_stats(home_team, df)
    away_stats = get_team_stats(away_team, df)
    
    # Display stats side by side
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"### 🏠 {home_team}")
        for key, value in home_stats.items():
            st.write(f"**{key}:** {value}")
    
    with col2:
        st.markdown(f"### ✈️ {away_team}")
        for key, value in away_stats.items():
            st.write(f"**{key}:** {value}")
    
    # Simple comparison chart
    if any(home_stats.values()) or any(away_stats.values()):
        fig, ax = plt.subplots(figsize=(10, 6))
        
        stats_names = list(home_stats.keys())
        home_values = list(home_stats.values())
        away_values = list(away_stats.values())
        
        x = np.arange(len(stats_names))
        width = 0.35
        
        ax.bar(x - width/2, home_values, width, label=home_team, alpha=0.8)
        ax.bar(x + width/2, away_values, width, label=away_team, alpha=0.8)
        
        ax.set_xlabel('Statistics')
        ax.set_ylabel('Values')
        ax.set_title('Team Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(stats_names, rotation=45, ha='right')
        ax.legend()
        
        plt.tight_layout()
        st.pyplot(fig)

# -------------------- Main Application --------------------
def main():
    """Main application function"""
    # Load data and model
    df_full, raw_years, season_labels = load_data()
    predictor = load_predictor()
    
    # Apply styling
    AppStyle.apply_background_color("#1a0202ff")
    AppStyle.center_title("Belgian Pro League Prediction")
    
    # Team selection
    home_team, away_team = create_team_selector()
    
    # Prediction section
    if st.button("Predict Match", type="primary"):
        with st.spinner("Making prediction..."):
            prediction = predict_match(predictor, home_team, away_team)
            
            st.success("Prediction completed!")
            st.subheader("Prediction Result")
            st.markdown(f"**Predicted outcome:** {prediction}")
            
            # Store teams for stats display
            st.session_state["show_stats"] = True
            st.session_state["last_home"] = home_team
            st.session_state["last_away"] = away_team
    
    # Display team comparison if prediction was made
    if st.session_state.get("show_stats", False):
        last_home = st.session_state.get("last_home")
        last_away = st.session_state.get("last_away")
        
        if last_home and last_away:
            display_team_comparison(last_home, last_away, df_full)
    
    # Footer
    AppStyle.add_footer(author="Clean Architecture Team", year="2025")

# -------------------- Run Application --------------------
if __name__ == "__main__":
    main()
=======
import streamlit as st
import logging

from src.frontend.styles.app_styles import AppStyle
from src.frontend.components.predictor import MatchPrediction
from config import settings
from utils.logger_config import configure_logging
from utils.data_io import load_model

configure_logging()
logger = logging.getLogger(__name__)


class FootballPredictorApp:
    """
    Main class to launch the Streamlit-based football match predictor app.
    This class is responsible for loading the model, rendering the UI,
    and handling user interaction.
    """

    def __init__(self):
        self.model = load_model(filename="trained_model.joblib")
        self.predictor = MatchPrediction(self.model)

    def run(self) -> None:
        """
        Run the Streamlit application.
        Renders UI components, captures user input, and displays prediction.
        :return: None
        """
        AppStyle.apply_background_color('#87CEEB')
        AppStyle.center_title('Football Belgium League Predictor')

        # Render team selection form
        self.predictor.input_form.render()

        # Center the "Predict" button using columns
        col1, col2, col3 = st.columns([2, 1, 2])
        with col2:
            if st.button("Predict"):
                prediction = self.predictor.handle_prediction()
                AppStyle.show_prediction_block(prediction)

        AppStyle.add_footer("Красавчик")


if __name__ == "__main__":
    app = FootballPredictorApp()
    app.run()
>>>>>>> Stashed changes
