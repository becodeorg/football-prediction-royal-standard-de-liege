<<<<<<< HEAD
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import pandas as pd
from app_styles import AppStyle

# -------------------- Load Data & Seasons --------------------
DATA_PATH = os.path.abspath(os.path.dirname(__file__) + '/../../data/raw/dataset_old_2.csv')
df_full = pd.read_csv(DATA_PATH)
# Extract available seasons (years from Date column)
df_full['Year'] = pd.to_datetime(df_full['Date'], errors='coerce').dt.year
# Format seasons as 'Saison YYYY-YYYY'
raw_years = sorted(df_full['Year'].dropna().unique(), reverse=True)
season_labels = [f"Saison {int(y)}-{int(y)+1}" for y in raw_years]
season_labels.insert(0, "Toutes saisons")

# Import du modèle
sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/.."))
from backend.model.simple_predictor_forest import SimplePredictorForest

# -------------------- Style Section --------------------
AppStyle.apply_background_color("#1a0202ff")
AppStyle.center_title("Jupiler Pro League Prediction")

# -------------------- Data Section --------------------
jupiler_teams = [
    "Anderlecht", "Club Brugge", "Standard", "Genk",
    "Gent", "Charleroi", "Cercle Brugge", "Antwerp",
    "Westerlo", "OH Leuven", "Union Saint-Gilloise", "Mechelen",
    "Eupen", "Kortrijk", "RWD Molenbeek", "St Truiden"
]


# -------------------- UI Section --------------------
st.markdown("Select two teams to get a match prediction.")
col_team1, col_vs, col_team2 = st.columns([4, 1, 4])
with col_team1:
    home_team = st.selectbox("Home Team", jupiler_teams, key="home")
with col_vs:
    st.markdown("<h3 style='text-align: center;'>VS</h3>", unsafe_allow_html=True)
with col_team2:
    away_options = [team for team in jupiler_teams if team != home_team]
    away_team = st.selectbox("Away Team", away_options, key="away")


# --- Season selection for stats ---

# -------------------- Model Training Section --------------------
@st.cache_data(show_spinner=False)
def load_and_train_model():
    df = pd.read_csv(os.path.abspath(os.path.dirname(__file__) + '/../../data/raw/dataset_old_2.csv'))
    # Rename columns to match expected format for SimplePredictorForest
    train_df = df.rename(columns={
        'HomeTeam': 'home_team',
        'AwayTeam': 'away_team', 
        'FTHG': 'home_goals',
        'FTAG': 'away_goals',
        'FTR': 'outcome'  # SimplePredictorForest expects 'outcome'
    })
    # Keep only necessary columns
    train_df = train_df[['home_team', 'away_team', 'home_goals', 'away_goals', 'outcome']].copy()
    
    # Initialize and train the predictor
    predictor = SimplePredictorForest()
    predictor.build_model()
    
    # Use try-catch to handle different training signatures
    try:
        # SimplePredictorForest expects a DataFrame
        predictor.train(train_df)
    except Exception as e:
        st.error(f"Error training model: {e}")
        # Fallback: create a simple mock predictor
        predictor = create_mock_predictor()
    
    return predictor

def create_mock_predictor():
    """Create a mock predictor for fallback if training fails"""
    class MockPredictor:
        def predict(self, input_df):
            return ['H']  # Always predict home win
        def predict_home_goals(self, input_df):
            return [2]  # Always predict 2 goals
        def predict_away_goals(self, input_df):
            return [1]  # Always predict 1 goal
    return MockPredictor()

predictor = load_and_train_model()


def predict_match_outcome_with_score(home, away):
    """Predict match outcome and score with robust error handling"""
    input_df = pd.DataFrame([{
        'home_team': home,
        'away_team': away,
        'home_goals': 0,
        'away_goals': 0
    }])
    
    try:
        # Try to get outcome prediction
        outcome_model = predictor.predict(input_df)[0]
    except Exception as e:
        print(f"Error in outcome prediction: {e}")
        outcome_model = 'H'  # Default to home win
    
    try:
        # Try to get score predictions
        home_score = int(round(predictor.predict_home_goals(input_df)[0]))
        away_score = int(round(predictor.predict_away_goals(input_df)[0]))
    except Exception as e:
        print(f"Error in score prediction: {e}")
        # Use heuristic based on outcome
        if outcome_model == 'H':
            home_score, away_score = 2, 1
        elif outcome_model == 'A':
            home_score, away_score = 1, 2
        else:  # Draw
            home_score, away_score = 1, 1
    
    # Calculate outcome from score
    if home_score > away_score:
        outcome_score = 'H'
    elif home_score < away_score:
        outcome_score = 'A'
    else:
        outcome_score = 'D'
    
    return outcome_score, home_score, away_score, outcome_model

# --- Real stats extraction from dataset ---

def get_real_stats(team, df):
    """Extract real stats for a team with robust error handling"""
    try:
        # Filtrage par équipe
        home = df[df['HomeTeam'] == team] if 'HomeTeam' in df.columns else pd.DataFrame()
        away = df[df['AwayTeam'] == team] if 'AwayTeam' in df.columns else pd.DataFrame()
        
        matches_played = len(home) + len(away)
        
        # Calculs sécurisés avec vérification des colonnes
        if 'FTHG' in df.columns and 'FTAG' in df.columns:
            goals_scored = home['FTHG'].sum() + away['FTAG'].sum()
            goals_conceded = home['FTAG'].sum() + away['FTHG'].sum()
        else:
            goals_scored = goals_conceded = 0
            
        if 'FTR' in df.columns:
            wins = (home['FTR'] == 'H').sum() + (away['FTR'] == 'A').sum()
            draws = (home['FTR'] == 'D').sum() + (away['FTR'] == 'D').sum()
        else:
            wins = draws = 0
            
        losses = matches_played - wins - draws
        
        # Calcul du classement simplifié
        if 'HomeTeam' in df.columns and 'AwayTeam' in df.columns and 'FTR' in df.columns:
            teams = list(set(df['HomeTeam'].unique().tolist() + df['AwayTeam'].unique().tolist()))
            team_wins = {t: (df[df['HomeTeam'] == t]['FTR'] == 'H').sum() + (df[df['AwayTeam'] == t]['FTR'] == 'A').sum() for t in teams}
            sorted_teams = sorted(team_wins.items(), key=lambda x: x[1], reverse=True)
            position = next((i+1 for i, (t, _) in enumerate(sorted_teams) if t == team), None)
        else:
            position = None
            
        # Données enrichies - vérifier si les colonnes existent
        team_row = pd.concat([home, away]).iloc[0] if (len(home) > 0 or len(away) > 0) else None
        
        # Fonction pour extraire une valeur de façon sécurisée
        def safe_get(row, column, default=None):
            if row is not None and column in row:
                val = row[column]
                return val if val is not None and str(val).lower() != 'nan' else default
            return default
        
        return {
            "Position": position if position else "N/A",
            "Goals Scored": int(goals_scored),
            "Goals Conceded": int(goals_conceded),
            "Wins": int(wins),
            "Draws": int(draws),
            "Losses": int(losses),
            "Matches Played": int(matches_played),
            "Logo": safe_get(team_row, 'home_logoURL'),
            "Color": safe_get(team_row, 'home_color'),
            "Alt Color": safe_get(team_row, 'home_altColor'),
            "Venue": safe_get(team_row, 'homeVenue_fullName', 'Unknown Stadium'),
            "Venue City": safe_get(team_row, 'homeVenue_city', 'Unknown City'),
            "Venue Capacity": safe_get(team_row, 'homeVenue_capacity', 'Unknown'),
            "Club Goals 24/25": safe_get(team_row, 'club_goals_24_25'),
            "Club Assists 24/25": safe_get(team_row, 'club_assists_24_25'),
            "Club Games 24/25": safe_get(team_row, 'club_games_24_25'),
            "Club Minutes 24/25": safe_get(team_row, 'club_minutes_24_25')
        }
    except Exception as e:
        print(f"Error in get_real_stats for {team}: {e}")
        # Retourner des stats par défaut en cas d'erreur
        return {
            "Position": "N/A",
            "Goals Scored": 0,
            "Goals Conceded": 0,
            "Wins": 0,
            "Draws": 0,
            "Losses": 0,
            "Matches Played": 0,
            "Logo": None,
            "Color": None,
            "Alt Color": None,
            "Venue": "Unknown Stadium",
            "Venue City": "Unknown City",
            "Venue Capacity": "Unknown",
            "Club Goals 24/25": None,
            "Club Assists 24/25": None,
            "Club Games 24/25": None,
            "Club Minutes 24/25": None
        }




def show_stats_and_heatmap(home_team, away_team, season_year):
    st.subheader(f"Team Statistics (Saison {int(season_year)}-{int(season_year)+1})")
    st.markdown("<span style='color:white'><b>Note :</b> La saison sélectionnée n'impacte que les statistiques affichées ci-dessous, pas la prédiction du match.</span>", unsafe_allow_html=True)
    season_idx = st.selectbox("Sélectionnez la saison pour les statistiques d'équipe", list(range(len(season_labels))), format_func=lambda i: season_labels[i], key="season")
    if season_idx == 0:
        # Toutes saisons
        df_season = df_full.copy()
    else:
        season_year = raw_years[season_idx - 1]
        df_season = df_full[df_full['Year'] == season_year]
    stats_home = get_real_stats(home_team, df_season)
    stats_away = get_real_stats(away_team, df_season)
    col1, col2 = st.columns(2)
    # --- Affichage Home ---
    with col1:
        st.markdown(f"### 🏠 {home_team}")
        logo_home = stats_home["Logo"]
        if logo_home and isinstance(logo_home, str) and logo_home.strip() and str(logo_home).lower() != 'nan':
            st.image(logo_home, width=80)
        # Affichage propre du stade
        def format_stadium(venue, city, capacity):
            def clean(val):
                return "-" if val is None or (isinstance(val, float) and np.isnan(val)) or str(val).lower() == "nan" else str(val)
            return f"{clean(venue)} ({clean(city)}, {clean(capacity)})"
        st.write(f"**Stadium:** {format_stadium(stats_home['Venue'], stats_home['Venue City'], stats_home['Venue Capacity'])}")
        # Couleurs supprimées
        # Affiche les stats club seulement si elles sont non NaN
        for stat_key, stat_label in [
            ('Club Goals 24/25', 'Club Goals 24/25'),
            ('Club Assists 24/25', 'Club Assists 24/25'),
            ('Club Games 24/25', 'Club Games 24/25'),
            ('Club Minutes 24/25', 'Club Minutes 24/25')
        ]:
            val = stats_home.get(stat_key, None)
            if val is not None and str(val).lower() != 'nan':
                st.write(f"**{stat_label}:** {val}")
        for key in ["Position", "Goals Scored", "Goals Conceded", "Wins", "Draws", "Losses", "Matches Played"]:
            st.write(f"**{key}** : {stats_home[key]}")
    # --- Affichage Away ---
    with col2:
        st.markdown(f"### 🛫 {away_team}")
        logo_away = stats_away["Logo"]
        if logo_away and isinstance(logo_away, str) and logo_away.strip() and str(logo_away).lower() != 'nan':
            st.image(logo_away, width=80)
        st.write(f"**Stadium:** {format_stadium(stats_away['Venue'], stats_away['Venue City'], stats_away['Venue Capacity'])}")
        # Couleurs supprimées
        for stat_key, stat_label in [
            ('Club Goals 24/25', 'Club Goals 24/25'),
            ('Club Assists 24/25', 'Club Assists 24/25'),
            ('Club Games 24/25', 'Club Games 24/25'),
            ('Club Minutes 24/25', 'Club Minutes 24/25')
        ]:
            val = stats_away.get(stat_key, None)
            if val is not None and str(val).lower() != 'nan':
                st.write(f"**{stat_label}:** {val}")
        for key in ["Position", "Goals Scored", "Goals Conceded", "Wins", "Draws", "Losses", "Matches Played"]:
            st.write(f"**{key}** : {stats_away[key]}")
    st.subheader("Visual Comparison")
    stats_labels = ["Position", "Goals Scored", "Goals Conceded", "Wins", "Draws", "Losses", "Matches Played"]
    def safe_int(val):
        try:
            if val is None or (isinstance(val, float) and np.isnan(val)):
                return 0
            return int(val)
        except Exception:
            return 0
    home_stats = [safe_int(stats_home[label]) for label in stats_labels]
    away_stats = [safe_int(stats_away[label]) for label in stats_labels]

    # Bar chart comparatif
    fig_bar, ax_bar = plt.subplots(figsize=(10, 4))
    x = np.arange(len(stats_labels))
    ax_bar.bar(x - 0.2, home_stats, width=0.4, label=home_team, color='#1f77b4')
    ax_bar.bar(x + 0.2, away_stats, width=0.4, label=away_team, color='#ff7f0e')
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(stats_labels, rotation=30, ha='right')
    ax_bar.set_ylabel('Value')
    ax_bar.set_title('Comparaison des statistiques principales')
    ax_bar.legend()
    st.pyplot(fig_bar)

    # Radar chart
    from math import pi
    categories = stats_labels
    N = len(categories)
    values_home = home_stats + [home_stats[0]]
    values_away = away_stats + [away_stats[0]]
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]
    fig_radar = plt.figure(figsize=(6, 6))
    ax_radar = plt.subplot(111, polar=True)
    ax_radar.plot(angles, values_home, linewidth=2, linestyle='solid', label=home_team, color='#1f77b4')
    ax_radar.fill(angles, values_home, alpha=0.25, color='#1f77b4')
    ax_radar.plot(angles, values_away, linewidth=2, linestyle='solid', label=away_team, color='#ff7f0e')
    ax_radar.fill(angles, values_away, alpha=0.25, color='#ff7f0e')
    ax_radar.set_xticks(angles[:-1])
    ax_radar.set_xticklabels(categories)
    ax_radar.set_title('Profil global des équipes')
    ax_radar.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1))
    st.pyplot(fig_radar)

# -------------------- Main App Logic --------------------





# --- Gestion de l'état pour afficher les stats ---
if "last_pred_home" not in st.session_state:
    st.session_state["last_pred_home"] = None
if "last_pred_away" not in st.session_state:
    st.session_state["last_pred_away"] = None
if "show_stats" not in st.session_state:
    st.session_state["show_stats"] = False

if st.button("Predict Match"):
    try:
        outcome, home_score, away_score, outcome_model = predict_match_outcome_with_score(home_team, away_team)
        st.subheader("Prediction Result")
        # Affichage cohérent avec le score
        if outcome == 'H':
            st.markdown(f"**Predicted outcome (from score):** {home_team} wins")
        elif outcome == 'A':
            st.markdown(f"**Predicted outcome (from score):** {away_team} wins")
        elif outcome == 'D':
            st.markdown("**Predicted outcome (from score):** Draw")
        else:
            st.markdown(f"**Predicted outcome (from score):** {outcome}")
        # Afficher le score
        if home_score is not None and away_score is not None:
            st.markdown(f"**Predicted score:** {home_team} {home_score} - {away_team} {away_score}")
        # Afficher l'issue du modèle SEULEMENT si le score n'est pas disponible
        elif outcome != outcome_model:
            if outcome_model == 'H':
                st.markdown(f"<span style='color:orange'>Issue du modèle : {home_team} wins</span>", unsafe_allow_html=True)
            elif outcome_model == 'A':
                st.markdown(f"<span style='color:orange'>Issue du modèle : {away_team} wins</span>", unsafe_allow_html=True)
            elif outcome_model == 'D':
                st.markdown("<span style='color:orange'>Issue du modèle : Draw</span>", unsafe_allow_html=True)
            else:
                st.markdown(f"<span style='color:orange'>Issue du modèle : {outcome_model}</span>", unsafe_allow_html=True)
        # Mémoriser les équipes pour les stats
        st.session_state["last_pred_home"] = home_team
        st.session_state["last_pred_away"] = away_team
        st.session_state["show_stats"] = True
    except Exception as e:
        st.error(f"Erreur lors de la prédiction : {e}")
        import traceback
        st.text(traceback.format_exc())

# Afficher les stats et heatmap si une prédiction a été faite
if st.session_state["show_stats"] and st.session_state["last_pred_home"] and st.session_state["last_pred_away"]:
    show_stats_and_heatmap(st.session_state["last_pred_home"], st.session_state["last_pred_away"], raw_years[0])

# -------------------- Footer --------------------
AppStyle.add_footer(author="Hervé, Konstantin et Santo a.k.a The Dream Team", year="2025")
=======
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import pandas as pd
from app_styles import AppStyle

# -------------------- Load Data & Seasons --------------------
DATA_PATH = os.path.abspath(os.path.dirname(__file__) + '/../../data/raw/dataset_old_2.csv')
df_full = pd.read_csv(DATA_PATH)
# Extract available seasons (years from Date column)
df_full['Year'] = pd.to_datetime(df_full['Date'], errors='coerce').dt.year
# Format seasons as 'Saison YYYY-YYYY'
raw_years = sorted(df_full['Year'].dropna().unique(), reverse=True)
season_labels = [f"Saison {int(y)}-{int(y)+1}" for y in raw_years]
season_labels.insert(0, "Toutes saisons")

# Import du modèle
sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/.."))
from backend.model.simple_predictor_forest import SimplePredictorForest

# -------------------- Style Section --------------------
AppStyle.apply_background_color("#1a0202ff")
AppStyle.center_title("Jupiler Pro League Prediction")

# -------------------- Data Section --------------------
jupiler_teams = [
    "Anderlecht", "Club Brugge", "Standard", "Genk",
    "Gent", "Charleroi", "Cercle Brugge", "Antwerp",
    "Westerlo", "OH Leuven", "Union Saint-Gilloise", "Mechelen",
    "Eupen", "Kortrijk", "RWD Molenbeek", "St Truiden"
]


# -------------------- UI Section --------------------
st.markdown("Select two teams to get a match prediction.")
col_team1, col_vs, col_team2 = st.columns([4, 1, 4])
with col_team1:
    home_team = st.selectbox("Home Team", jupiler_teams, key="home")
with col_vs:
    st.markdown("<h3 style='text-align: center;'>VS</h3>", unsafe_allow_html=True)
with col_team2:
    away_options = [team for team in jupiler_teams if team != home_team]
    away_team = st.selectbox("Away Team", away_options, key="away")


# --- Season selection for stats ---

# -------------------- Model Training Section --------------------
@st.cache_data(show_spinner=False)
def load_and_train_model():
    df = pd.read_csv(os.path.abspath(os.path.dirname(__file__) + '/../../data/raw/dataset_old_2.csv'))
    # Rename columns to match expected format for SimplePredictorForest
    train_df = df.rename(columns={
        'HomeTeam': 'home_team',
        'AwayTeam': 'away_team', 
        'FTHG': 'home_goals',
        'FTAG': 'away_goals',
        'FTR': 'outcome'  # SimplePredictorForest expects 'outcome'
    })
    # Keep only necessary columns
    train_df = train_df[['home_team', 'away_team', 'home_goals', 'away_goals', 'outcome']].copy()
    
    # Initialize and train the predictor
    predictor = SimplePredictorForest()
    predictor.build_model()
    
    # Use try-catch to handle different training signatures
    try:
        # SimplePredictorForest expects a DataFrame
        predictor.train(train_df)
    except Exception as e:
        st.error(f"Error training model: {e}")
        # Fallback: create a simple mock predictor
        predictor = create_mock_predictor()
    
    return predictor

def create_mock_predictor():
    """Create a mock predictor for fallback if training fails"""
    class MockPredictor:
        def predict(self, input_df):
            return ['H']  # Always predict home win
        def predict_home_goals(self, input_df):
            return [2]  # Always predict 2 goals
        def predict_away_goals(self, input_df):
            return [1]  # Always predict 1 goal
    return MockPredictor()

predictor = load_and_train_model()


def predict_match_outcome_with_score(home, away):
    """Predict match outcome and score with robust error handling"""
    input_df = pd.DataFrame([{
        'home_team': home,
        'away_team': away,
        'home_goals': 0,
        'away_goals': 0
    }])
    
    try:
        # Try to get outcome prediction
        outcome_model = predictor.predict(input_df)[0]
    except Exception as e:
        print(f"Error in outcome prediction: {e}")
        outcome_model = 'H'  # Default to home win
    
    try:
        # Try to get score predictions
        home_score = int(round(predictor.predict_home_goals(input_df)[0]))
        away_score = int(round(predictor.predict_away_goals(input_df)[0]))
    except Exception as e:
        print(f"Error in score prediction: {e}")
        # Use heuristic based on outcome
        if outcome_model == 'H':
            home_score, away_score = 2, 1
        elif outcome_model == 'A':
            home_score, away_score = 1, 2
        else:  # Draw
            home_score, away_score = 1, 1
    
    # Calculate outcome from score
    if home_score > away_score:
        outcome_score = 'H'
    elif home_score < away_score:
        outcome_score = 'A'
    else:
        outcome_score = 'D'
    
    return outcome_score, home_score, away_score, outcome_model

# --- Real stats extraction from dataset ---

def get_real_stats(team, df):
    """Extract real stats for a team with robust error handling"""
    try:
        # Filtrage par équipe
        home = df[df['HomeTeam'] == team] if 'HomeTeam' in df.columns else pd.DataFrame()
        away = df[df['AwayTeam'] == team] if 'AwayTeam' in df.columns else pd.DataFrame()
        
        matches_played = len(home) + len(away)
        
        # Calculs sécurisés avec vérification des colonnes
        if 'FTHG' in df.columns and 'FTAG' in df.columns:
            goals_scored = home['FTHG'].sum() + away['FTAG'].sum()
            goals_conceded = home['FTAG'].sum() + away['FTHG'].sum()
        else:
            goals_scored = goals_conceded = 0
            
        if 'FTR' in df.columns:
            wins = (home['FTR'] == 'H').sum() + (away['FTR'] == 'A').sum()
            draws = (home['FTR'] == 'D').sum() + (away['FTR'] == 'D').sum()
        else:
            wins = draws = 0
            
        losses = matches_played - wins - draws
        
        # Calcul du classement simplifié
        if 'HomeTeam' in df.columns and 'AwayTeam' in df.columns and 'FTR' in df.columns:
            teams = list(set(df['HomeTeam'].unique().tolist() + df['AwayTeam'].unique().tolist()))
            team_wins = {t: (df[df['HomeTeam'] == t]['FTR'] == 'H').sum() + (df[df['AwayTeam'] == t]['FTR'] == 'A').sum() for t in teams}
            sorted_teams = sorted(team_wins.items(), key=lambda x: x[1], reverse=True)
            position = next((i+1 for i, (t, _) in enumerate(sorted_teams) if t == team), None)
        else:
            position = None
            
        # Données enrichies - vérifier si les colonnes existent
        team_row = pd.concat([home, away]).iloc[0] if (len(home) > 0 or len(away) > 0) else None
        
        # Fonction pour extraire une valeur de façon sécurisée
        def safe_get(row, column, default=None):
            if row is not None and column in row:
                val = row[column]
                return val if val is not None and str(val).lower() != 'nan' else default
            return default
        
        return {
            "Position": position if position else "N/A",
            "Goals Scored": int(goals_scored),
            "Goals Conceded": int(goals_conceded),
            "Wins": int(wins),
            "Draws": int(draws),
            "Losses": int(losses),
            "Matches Played": int(matches_played),
            "Logo": safe_get(team_row, 'home_logoURL'),
            "Color": safe_get(team_row, 'home_color'),
            "Alt Color": safe_get(team_row, 'home_altColor'),
            "Venue": safe_get(team_row, 'homeVenue_fullName', 'Unknown Stadium'),
            "Venue City": safe_get(team_row, 'homeVenue_city', 'Unknown City'),
            "Venue Capacity": safe_get(team_row, 'homeVenue_capacity', 'Unknown'),
            "Club Goals 24/25": safe_get(team_row, 'club_goals_24_25'),
            "Club Assists 24/25": safe_get(team_row, 'club_assists_24_25'),
            "Club Games 24/25": safe_get(team_row, 'club_games_24_25'),
            "Club Minutes 24/25": safe_get(team_row, 'club_minutes_24_25')
        }
    except Exception as e:
        print(f"Error in get_real_stats for {team}: {e}")
        # Retourner des stats par défaut en cas d'erreur
        return {
            "Position": "N/A",
            "Goals Scored": 0,
            "Goals Conceded": 0,
            "Wins": 0,
            "Draws": 0,
            "Losses": 0,
            "Matches Played": 0,
            "Logo": None,
            "Color": None,
            "Alt Color": None,
            "Venue": "Unknown Stadium",
            "Venue City": "Unknown City",
            "Venue Capacity": "Unknown",
            "Club Goals 24/25": None,
            "Club Assists 24/25": None,
            "Club Games 24/25": None,
            "Club Minutes 24/25": None
        }




def show_stats_and_heatmap(home_team, away_team, season_year):
    st.subheader(f"Team Statistics (Saison {int(season_year)}-{int(season_year)+1})")
    st.markdown("<span style='color:white'><b>Note :</b> La saison sélectionnée n'impacte que les statistiques affichées ci-dessous, pas la prédiction du match.</span>", unsafe_allow_html=True)
    season_idx = st.selectbox("Sélectionnez la saison pour les statistiques d'équipe", list(range(len(season_labels))), format_func=lambda i: season_labels[i], key="season")
    if season_idx == 0:
        # Toutes saisons
        df_season = df_full.copy()
    else:
        season_year = raw_years[season_idx - 1]
        df_season = df_full[df_full['Year'] == season_year]
    stats_home = get_real_stats(home_team, df_season)
    stats_away = get_real_stats(away_team, df_season)
    col1, col2 = st.columns(2)
    # --- Affichage Home ---
    with col1:
        st.markdown(f"### 🏠 {home_team}")
        logo_home = stats_home["Logo"]
        if logo_home and isinstance(logo_home, str) and logo_home.strip() and str(logo_home).lower() != 'nan':
            st.image(logo_home, width=80)
        # Affichage propre du stade
        def format_stadium(venue, city, capacity):
            def clean(val):
                return "-" if val is None or (isinstance(val, float) and np.isnan(val)) or str(val).lower() == "nan" else str(val)
            return f"{clean(venue)} ({clean(city)}, {clean(capacity)})"
        st.write(f"**Stadium:** {format_stadium(stats_home['Venue'], stats_home['Venue City'], stats_home['Venue Capacity'])}")
        # Couleurs supprimées
        # Affiche les stats club seulement si elles sont non NaN
        for stat_key, stat_label in [
            ('Club Goals 24/25', 'Club Goals 24/25'),
            ('Club Assists 24/25', 'Club Assists 24/25'),
            ('Club Games 24/25', 'Club Games 24/25'),
            ('Club Minutes 24/25', 'Club Minutes 24/25')
        ]:
            val = stats_home.get(stat_key, None)
            if val is not None and str(val).lower() != 'nan':
                st.write(f"**{stat_label}:** {val}")
        for key in ["Position", "Goals Scored", "Goals Conceded", "Wins", "Draws", "Losses", "Matches Played"]:
            st.write(f"**{key}** : {stats_home[key]}")
    # --- Affichage Away ---
    with col2:
        st.markdown(f"### 🛫 {away_team}")
        logo_away = stats_away["Logo"]
        if logo_away and isinstance(logo_away, str) and logo_away.strip() and str(logo_away).lower() != 'nan':
            st.image(logo_away, width=80)
        st.write(f"**Stadium:** {format_stadium(stats_away['Venue'], stats_away['Venue City'], stats_away['Venue Capacity'])}")
        # Couleurs supprimées
        for stat_key, stat_label in [
            ('Club Goals 24/25', 'Club Goals 24/25'),
            ('Club Assists 24/25', 'Club Assists 24/25'),
            ('Club Games 24/25', 'Club Games 24/25'),
            ('Club Minutes 24/25', 'Club Minutes 24/25')
        ]:
            val = stats_away.get(stat_key, None)
            if val is not None and str(val).lower() != 'nan':
                st.write(f"**{stat_label}:** {val}")
        for key in ["Position", "Goals Scored", "Goals Conceded", "Wins", "Draws", "Losses", "Matches Played"]:
            st.write(f"**{key}** : {stats_away[key]}")
    st.subheader("Visual Comparison")
    stats_labels = ["Position", "Goals Scored", "Goals Conceded", "Wins", "Draws", "Losses", "Matches Played"]
    def safe_int(val):
        try:
            if val is None or (isinstance(val, float) and np.isnan(val)):
                return 0
            return int(val)
        except Exception:
            return 0
    home_stats = [safe_int(stats_home[label]) for label in stats_labels]
    away_stats = [safe_int(stats_away[label]) for label in stats_labels]

    # Bar chart comparatif
    fig_bar, ax_bar = plt.subplots(figsize=(10, 4))
    x = np.arange(len(stats_labels))
    ax_bar.bar(x - 0.2, home_stats, width=0.4, label=home_team, color='#1f77b4')
    ax_bar.bar(x + 0.2, away_stats, width=0.4, label=away_team, color='#ff7f0e')
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(stats_labels, rotation=30, ha='right')
    ax_bar.set_ylabel('Value')
    ax_bar.set_title('Comparaison des statistiques principales')
    ax_bar.legend()
    st.pyplot(fig_bar)

    # Radar chart
    from math import pi
    categories = stats_labels
    N = len(categories)
    values_home = home_stats + [home_stats[0]]
    values_away = away_stats + [away_stats[0]]
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]
    fig_radar = plt.figure(figsize=(6, 6))
    ax_radar = plt.subplot(111, polar=True)
    ax_radar.plot(angles, values_home, linewidth=2, linestyle='solid', label=home_team, color='#1f77b4')
    ax_radar.fill(angles, values_home, alpha=0.25, color='#1f77b4')
    ax_radar.plot(angles, values_away, linewidth=2, linestyle='solid', label=away_team, color='#ff7f0e')
    ax_radar.fill(angles, values_away, alpha=0.25, color='#ff7f0e')
    ax_radar.set_xticks(angles[:-1])
    ax_radar.set_xticklabels(categories)
    ax_radar.set_title('Profil global des équipes')
    ax_radar.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1))
    st.pyplot(fig_radar)

# -------------------- Main App Logic --------------------





# --- Gestion de l'état pour afficher les stats ---
if "last_pred_home" not in st.session_state:
    st.session_state["last_pred_home"] = None
if "last_pred_away" not in st.session_state:
    st.session_state["last_pred_away"] = None
if "show_stats" not in st.session_state:
    st.session_state["show_stats"] = False

if st.button("Predict Match"):
    try:
        outcome, home_score, away_score, outcome_model = predict_match_outcome_with_score(home_team, away_team)
        st.subheader("Prediction Result")
        # Affichage cohérent avec le score
        if outcome == 'H':
            st.markdown(f"**Predicted outcome (from score):** {home_team} wins")
        elif outcome == 'A':
            st.markdown(f"**Predicted outcome (from score):** {away_team} wins")
        elif outcome == 'D':
            st.markdown("**Predicted outcome (from score):** Draw")
        else:
            st.markdown(f"**Predicted outcome (from score):** {outcome}")
        # Afficher le score
        if home_score is not None and away_score is not None:
            st.markdown(f"**Predicted score:** {home_team} {home_score} - {away_team} {away_score}")
        # Afficher l'issue du modèle SEULEMENT si le score n'est pas disponible
        elif outcome != outcome_model:
            if outcome_model == 'H':
                st.markdown(f"<span style='color:orange'>Issue du modèle : {home_team} wins</span>", unsafe_allow_html=True)
            elif outcome_model == 'A':
                st.markdown(f"<span style='color:orange'>Issue du modèle : {away_team} wins</span>", unsafe_allow_html=True)
            elif outcome_model == 'D':
                st.markdown("<span style='color:orange'>Issue du modèle : Draw</span>", unsafe_allow_html=True)
            else:
                st.markdown(f"<span style='color:orange'>Issue du modèle : {outcome_model}</span>", unsafe_allow_html=True)
        # Mémoriser les équipes pour les stats
        st.session_state["last_pred_home"] = home_team
        st.session_state["last_pred_away"] = away_team
        st.session_state["show_stats"] = True
    except Exception as e:
        st.error(f"Erreur lors de la prédiction : {e}")
        import traceback
        st.text(traceback.format_exc())

# Afficher les stats et heatmap si une prédiction a été faite
if st.session_state["show_stats"] and st.session_state["last_pred_home"] and st.session_state["last_pred_away"]:
    show_stats_and_heatmap(st.session_state["last_pred_home"], st.session_state["last_pred_away"], raw_years[0])

# -------------------- Footer --------------------
AppStyle.add_footer(author="Hervé, Konstantin et Santo a.k.a The Dream Team", year="2025")
>>>>>>> 9096206ad8408b41db8a3ff3e33a206f03b915cc
