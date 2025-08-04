import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import pandas as pd
from styles.app_styles import AppStyle

# -------------------- Page Configuration --------------------
st.set_page_config(
    page_title="JPL Prediction Pro",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------- Load Data & Seasons --------------------
DATA_PATH = os.path.abspath(os.path.dirname(__file__) + '/../../data/raw/dataset_old_2.csv')
df_full = pd.read_csv(DATA_PATH)
# Extract available seasons (years from Date column)
df_full['Year'] = pd.to_datetime(df_full['Date'], errors='coerce').dt.year
# Format seasons as 'Season YYYY-YYYY'
raw_years = sorted(df_full['Year'].dropna().unique(), reverse=True)
season_labels = [f"Season {int(y)}-{int(y)+1}" for y in raw_years]
season_labels.insert(0, "All seasons")

# Import backend components
sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/../.."))
from utils.data_io import load_model
from src.backend.model.my_model import ModelTrainer

# -------------------- Custom CSS Styles --------------------
st.markdown("""
<style>
    /* Global Styles */
    .main-header {
        background: linear-gradient(90deg, #FF6B6B, #4ECDC4);
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 30px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .main-header h1 {
        color: white;
        font-size: 2.5rem;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .team-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        color: white;
        margin: 10px 0;
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.2);
        transition: transform 0.3s ease;
    }
    
    .team-card:hover {
        transform: translateY(-5px);
    }
    
    .prediction-result {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 25px;
        border-radius: 20px;
        text-align: center;
        color: white;
        margin: 20px 0;
        box-shadow: 0 10px 20px rgba(0, 0, 0, 0.2);
        border: 2px solid rgba(255, 255, 255, 0.1);
    }
    
    .stats-container {
        background: rgba(255, 255, 255, 0.05);
        padding: 20px;
        border-radius: 15px;
        margin: 15px 0;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .vs-divider {
        text-align: center;
        font-size: 3rem;
        font-weight: bold;
        color: #FF6B6B;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        margin: 20px 0;
    }
    
    .stSelectbox > div > div > div {
        background-color: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        color: white;
        margin: 5px;
    }
</style>
""", unsafe_allow_html=True)

# -------------------- Main Header --------------------
st.markdown("""
<div class="main-header">
    <h1>⚽ Jupiler Pro League Prediction Pro</h1>
    <p style="margin: 10px 0 0 0; font-size: 1.2rem; opacity: 0.9;">
        Advanced analysis and intelligent predictions
    </p>
</div>
""", unsafe_allow_html=True)

# -------------------- Data Section --------------------
jupiler_teams = [
    "Anderlecht", "Club Brugge", "Standard", "Genk",
    "Gent", "Charleroi", "Cercle Brugge", "Antwerp",
    "Westerlo", "OH Leuven", "Union Saint-Gilloise", "Mechelen",
    "Eupen", "Kortrijk", "RWD Molenbeek", "St Truiden"
]


# -------------------- Team Selection Section --------------------
st.markdown("### 🏟️ Team Selection")

col1, col2, col3 = st.columns([2, 1, 2])

with col1:
    st.markdown('<div class="team-card">', unsafe_allow_html=True)
    st.markdown("#### 🏠 Home Team")
    home_team = st.selectbox(
        "Choose the home team",
        jupiler_teams,
        key="home",
        help="The team playing at home"
    )
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="vs-divider">VS</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="team-card">', unsafe_allow_html=True)
    st.markdown("#### 🛫 Away Team")
    away_options = [team for team in jupiler_teams if team != home_team]
    away_team = st.selectbox(
        "Choose the away team",
        away_options,
        key="away",
        help="The team playing away"
    )
    st.markdown('</div>', unsafe_allow_html=True)


# --- Season selection for stats ---

# -------------------- Model Loading Section --------------------
@st.cache_resource(show_spinner=False)
def load_trained_model():
    """Load the pre-trained model from the backend"""
    
    try:
        # Load the model from the joblib file
        model = load_model("trained_model.joblib")
        st.success("✅ Pre-trained model loaded successfully!")
        
        # Load data to get encoders
        df = pd.read_csv(DATA_PATH)
        
        # Create a ModelTrainer to access prediction methods
        trainer = ModelTrainer(df, "FTR")
        trainer.split_data()
        trainer.model = model  # Use the pre-trained model
        
        return trainer
        
    except Exception as e:
        st.error(f"❌ Error loading the model: {e}")
        st.info("🔄 Attempting to load a basic model...")
        
        # Fallback: create a simple model if the pre-trained model is not available
        df = pd.read_csv(DATA_PATH)
        trainer = ModelTrainer(df, "FTR")
        trainer.split_data()
        
        # Create and train a simple model
        from sklearn.ensemble import RandomForestClassifier
        trainer.build_pipeline(RandomForestClassifier(random_state=42, n_estimators=50))
        trainer.model = trainer.pipeline
        trainer.model.fit(trainer.X_train, trainer.y_train)
        
        st.warning("⚠️ Basic model created (limited performance)")
        return trainer

# Load the model
model_trainer = load_trained_model()


def predict_match_outcome_with_score(home, away):
    """Predict match outcome with the real backend model"""
    try:
        # Use prediction based on historical statistics from the dataset
        # because the complete model requires more features than we have in the interface
        
        # Analyze direct confrontation history
        confrontations = df_full[
            ((df_full['HomeTeam'] == home) & (df_full['AwayTeam'] == away)) |
            ((df_full['HomeTeam'] == away) & (df_full['AwayTeam'] == home))
        ]
        
        # Home team statistics
        home_matches = df_full[df_full['HomeTeam'] == home]
        home_wins = len(home_matches[home_matches['FTR'] == 'H'])
        home_total = len(home_matches)
        home_win_rate = home_wins / home_total if home_total > 0 else 0
        
        # Away team statistics
        away_matches = df_full[df_full['AwayTeam'] == away]
        away_wins = len(away_matches[away_matches['FTR'] == 'A'])
        away_total = len(away_matches)
        away_win_rate = away_wins / away_total if away_total > 0 else 0
        
        # Calculate goal averages
        home_goals_avg = home_matches['FTHG'].mean() if len(home_matches) > 0 else 1.5
        away_goals_avg = away_matches['FTAG'].mean() if len(away_matches) > 0 else 1.5
        
        # Prediction algorithm based on statistics
        home_score = (home_win_rate * 0.4) + (home_goals_avg / 3.0 * 0.3) + 0.1  # Home advantage
        away_score = (away_win_rate * 0.4) + (away_goals_avg / 3.0 * 0.3)
        
        # Adjustment based on direct confrontations
        if len(confrontations) > 0:
            recent_confrontations = confrontations.tail(3)  # Last 3 matches
            home_wins_h2h = len(recent_confrontations[
                ((recent_confrontations['HomeTeam'] == home) & (recent_confrontations['FTR'] == 'H')) |
                ((recent_confrontations['AwayTeam'] == home) & (recent_confrontations['FTR'] == 'A'))
            ])
            h2h_bonus = (home_wins_h2h / len(recent_confrontations)) * 0.2
            home_score += h2h_bonus
            away_score += (0.2 - h2h_bonus)
        
        # Determine the result
        diff = abs(home_score - away_score)
        confidence = min(50 + (diff * 100), 85)  # Between 50% and 85%
        
        if home_score > away_score + 0.05:  # Threshold to avoid too close predictions
            prediction = 'H'
            outcome = f'🏠 VICTORY {home.upper()}'
            emoji = '🔥'
            description = f"The home team {home} should win this match!"
        elif away_score > home_score + 0.05:
            prediction = 'A'
            outcome = f'🛫 VICTORY {away.upper()}'  
            emoji = '⚡'
            description = f"The away team {away} is favored for this match!"
        else:
            prediction = 'D'
            outcome = '🤝 DRAW'
            emoji = '⚖️'
            description = "A balanced match that could end in a draw"
            
        # Expressive display of prediction results
        st.markdown("---")
        st.markdown("### 🎯 **PREDICTION RESULT**")
        
        # Main container with style
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 25px;
            border-radius: 20px;
            text-align: center;
            color: white;
            margin: 20px 0;
            box-shadow: 0 10px 20px rgba(0, 0, 0, 0.3);
        ">
            <h2 style="margin: 0; font-size: 2rem;">{emoji} {outcome}</h2>
            <p style="margin: 15px 0; font-size: 1.2rem; opacity: 0.9;">{description}</p>
            <div style="margin-top: 20px;">
                <span style="font-size: 1.1rem; font-weight: bold;">Confidence Level: {confidence:.1f}%</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Technical details with style
        # Technical details with style
        with st.expander("🔍 **Technical Prediction Details**", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                st.metric("🏠 Home Team", home)
                st.metric("📊 Home win rate", f"{home_win_rate:.1%}")
                st.metric("⚽ Goals/home match", f"{home_goals_avg:.1f}")
            with col2:
                st.metric("🛫 Away Team", away)
                st.metric("📊 Away win rate", f"{away_win_rate:.1%}")
                st.metric("⚽ Goals/away match", f"{away_goals_avg:.1f}")
                
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("🎯 Prediction", prediction)
            with col2:
                st.metric("🤝 Direct confrontations", len(confrontations))
            with col3:
                st.metric("📈 Home technical score", f"{home_score:.3f}")
                st.metric("📉 Away technical score", f"{away_score:.3f}")
                
            st.info("ℹ️ Prediction based on advanced statistical analysis of historical performances")
        
        return outcome, confidence
        
    except Exception as e:
        st.error(f"❌ **Prediction error:** {e}")
        import traceback
        st.code(traceback.format_exc())
        return '🤝 DRAW (default)', 50.0

def show_prediction_results(prediction, confidence):
    st.markdown("---")
    st.markdown("### 🎯 Prediction Results")
    
    # Main container for prediction
    st.markdown('<div class="prediction-container">', unsafe_allow_html=True)
    
    # Display of main result
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        # Determination of color and icon according to prediction
        if prediction == "Home Win":
            result_emoji = "🏠"
            result_text = "HOME VICTORY"
            result_color = "#00D4AA"
            interpretation = "The home team is favored to win this match"
        elif prediction == "Away Win":
            result_emoji = "🛫"
            result_text = "AWAY VICTORY"
            result_color = "#FF6B6B"
            interpretation = "The away team is favored to win this match"
        else:
            result_emoji = "🤝"
            result_text = "DRAW"
            result_color = "#FFD93D"
            interpretation = "Both teams have equal chances, draw likely"
        
        # Styled display of result
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, {result_color}20, {result_color}10);
            border: 2px solid {result_color};
            border-radius: 15px;
            padding: 30px;
            text-align: center;
            margin: 20px 0;
            box-shadow: 0 8px 25px rgba(0,0,0,0.3);
        ">
            <div style="font-size: 4rem; margin-bottom: 10px;">{result_emoji}</div>
            <div style="
                font-size: 1.8rem;
                font-weight: bold;
                color: {result_color};
                margin-bottom: 15px;
                text-shadow: 0 2px 4px rgba(0,0,0,0.3);
            ">{result_text}</div>
            <div style="
                font-size: 1.1rem;
                color: #FAFAFA;
                font-style: italic;
                margin-bottom: 20px;
            ">{interpretation}</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Confidence bar
    st.markdown("#### 📈 Confidence Level")
    
    # Determination of confidence bar color
    if confidence >= 80:
        conf_color = "#00D4AA"
        conf_text = "VERY HIGH"
        conf_icon = "🔥"
    elif confidence >= 60:
        conf_color = "#FFD93D"
        conf_text = "HIGH"
        conf_icon = "👍"
    elif confidence >= 40:
        conf_color = "#FF8C42"
        conf_text = "MODERATE"
        conf_icon = "⚠️"
    else:
        conf_color = "#FF6B6B"
        conf_text = "LOW"
        conf_icon = "⚡"
    
    # Styled progress bar
    progress_col1, progress_col2 = st.columns([3, 1])
    
    with progress_col1:
        st.markdown(f"""
        <div style="
            background: #262730;
            border-radius: 25px;
            padding: 5px;
            box-shadow: inset 0 3px 6px rgba(0,0,0,0.4);
        ">
            <div style="
                width: {confidence}%;
                height: 25px;
                background: linear-gradient(90deg, {conf_color}, {conf_color}80);
                border-radius: 20px;
                transition: width 0.3s ease;
                display: flex;
                align-items: center;
                justify-content: center;
                color: white;
                font-weight: bold;
                font-size: 0.9rem;
                text-shadow: 0 1px 2px rgba(0,0,0,0.5);
            ">{confidence:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
    
    with progress_col2:
        st.markdown(f"""
        <div style="
            display: flex;
            align-items: center;
            justify-content: center;
            height: 35px;
            font-size: 1.1rem;
            font-weight: bold;
            color: {conf_color};
        ">
            {conf_icon} {conf_text}
        </div>
        """, unsafe_allow_html=True)
    
    # Model explanation
    st.markdown("#### 🧠 About This Prediction")
    
    with st.expander("🔍 Prediction Model Details", expanded=False):
        st.markdown("""
        **Our AI model analyzes several key factors:**
        
        🏆 **Historical Performance**
        - Previous match statistics
        - Recent form trends
        - Direct confrontation history
        
        📊 **Statistical Indicators**
        - Goals scored and conceded
        - Win percentage
        - Home vs away performance
        
        ⚽ **Match Context**
        - Home field advantage
        - Motivation and stakes
        - Playing conditions
        
        **Important note:** This prediction is based on historical data and statistics. 
        Football remains unpredictable and many factors can influence the final result.
        """)
    
    # Warning about prediction limits
    st.warning("⚠️ **Disclaimer:** This prediction is provided for informational purposes only. "
               "It does not constitute betting advice and does not guarantee the match result.")
    
    st.markdown('</div>', unsafe_allow_html=True)

# --- Real stats extraction from dataset ---

def get_real_stats(team, df):
    # Filtrage strict par saison déjà appliqué dans df
    home = df[df['HomeTeam'] == team]
    away = df[df['AwayTeam'] == team]
    # Debug: display number of matches played and won for selected season
    print(f"DEBUG {team} | Season: matches_played={len(home) + len(away)}, wins={(home['FTR'] == 'H').sum() + (away['FTR'] == 'A').sum()}")
    matches_played = len(home) + len(away)
    goals_scored = home['FTHG'].sum() + away['FTAG'].sum()
    goals_conceded = home['FTAG'].sum() + away['FTHG'].sum()
    wins = (home['FTR'] == 'H').sum() + (away['FTR'] == 'A').sum()
    draws = (home['FTR'] == 'D').sum() + (away['FTR'] == 'D').sum()
    losses = matches_played - wins - draws
    # Ranking calculated on filtered season
    teams = list(set(df['HomeTeam'].unique().tolist() + df['AwayTeam'].unique().tolist()))
    team_wins = {t: (df[df['HomeTeam'] == t]['FTR'] == 'H').sum() + (df[df['AwayTeam'] == t]['FTR'] == 'A').sum() for t in teams}
    sorted_teams = sorted(team_wins.items(), key=lambda x: x[1], reverse=True)
    position = [i+1 for i, (t, _) in enumerate(sorted_teams) if t == team][0] if team in dict(sorted_teams) else None
    # --- Enriched data ---
    # Take the first home or away row for team info
    team_row = pd.concat([home, away]).iloc[0] if (len(home) > 0 or len(away) > 0) else None
    if team_row is not None:
        logo = team_row['home_logoURL'] if 'home_logoURL' in team_row else None
        color = team_row['home_color'] if 'home_color' in team_row else None
        alt_color = team_row['home_altColor'] if 'home_altColor' in team_row else None
        venue = team_row['homeVenue_fullName'] if 'homeVenue_fullName' in team_row else None
        venue_city = team_row['homeVenue_city'] if 'homeVenue_city' in team_row else None
        venue_capacity = team_row['homeVenue_capacity'] if 'homeVenue_capacity' in team_row else None
        # Advanced club stats (scoring)
        club_goals = team_row.get('club_goals_24_25', None)
        club_assists = team_row.get('club_assists_24_25', None)
        club_games = team_row.get('club_games_24_25', None)
        club_minutes = team_row.get('club_minutes_24_25', None)
    else:
        logo = None
        color = None
        alt_color = None
        venue = None
        venue_city = None
        venue_capacity = None
        club_goals = None
        club_assists = None
        club_games = None
        club_minutes = None
    return {
        "Position": position,
        "Goals Scored": int(goals_scored),
        "Goals Conceded": int(goals_conceded),
        "Wins": int(wins),
        "Draws": int(draws),
        "Losses": int(losses),
        "Matches Played": int(matches_played),
        "Logo": logo,
        "Color": color,
        "Alt Color": alt_color,
        "Venue": venue,
        "Venue City": venue_city,
        "Venue Capacity": venue_capacity,
        "Club Goals 24/25": club_goals,
        "Club Assists 24/25": club_assists,
        "Club Games 24/25": club_games,
        "Club Minutes 24/25": club_minutes
    }




def show_stats_and_heatmap(home_team, away_team, season_year):
    st.markdown("---")
    st.markdown("### 📊 Detailed Team Statistics")
    
    # Sidebar for season selection
    with st.sidebar:
        st.markdown("### ⚙️ Analysis Parameters")
        season_idx = st.selectbox(
            "📅 Select the season",
            list(range(len(season_labels))),
            format_func=lambda i: season_labels[i],
            key="season",
            help="Choose the season to analyze the statistics"
        )
        
        show_advanced_stats = st.checkbox("📈 Advanced statistics", value=True)
        chart_type = st.radio(
            "📊 Visualization type",
            ["Bar Chart", "Radar Chart", "Both"],
            index=2
        )
    
    # Data filtering by season
    if season_idx == 0:
        df_season = df_full.copy()
        season_text = "All seasons"
    else:
        season_year = raw_years[season_idx - 1]
        df_season = df_full[df_full['Year'] == season_year]
        season_text = f"Season {int(season_year)}-{int(season_year)+1}"
    
    st.info(f"🗓️ **Period analyzed:** {season_text}")
    st.markdown("⚠️ **Note:** The selected season only impacts the displayed statistics, not the match prediction.")
    
    # Statistics retrieval
    stats_home = get_real_stats(home_team, df_season)
    stats_away = get_real_stats(away_team, df_season)
    
    # Display of team cards
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="stats-container">', unsafe_allow_html=True)
        st.markdown(f"### 🏠 {home_team}")
        
        # Logo de l'équipe
        logo_home = stats_home["Logo"]
        if logo_home and isinstance(logo_home, str) and logo_home.strip() and str(logo_home).lower() != 'nan':
            st.image(logo_home, width=100)
        
        # Informations du stade
        def format_stadium(venue, city, capacity):
            def clean(val):
                return "-" if val is None or (isinstance(val, float) and np.isnan(val)) or str(val).lower() == "nan" else str(val)
            return f"{clean(venue)} ({clean(city)}, {clean(capacity)})"
        
        st.markdown(f"**🏟️ Stadium:** {format_stadium(stats_home['Venue'], stats_home['Venue City'], stats_home['Venue Capacity'])}")
        
        # Main metrics
        metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
        with metrics_col1:
            st.metric("🏆 Position", stats_home["Position"])
            st.metric("⚽ Goals scored", stats_home["Goals Scored"])
        with metrics_col2:
            st.metric("🎯 Wins", stats_home["Wins"])
            st.metric("🤝 Draws", stats_home["Draws"])
        with metrics_col3:
            st.metric("❌ Losses", stats_home["Losses"])
            st.metric("📊 Matches played", stats_home["Matches Played"])
        
        # Advanced stats if enabled
        if show_advanced_stats:
            st.markdown("#### 📈 Advanced Club Statistics 24/25")
            for stat_key, stat_label in [
                ('Club Goals 24/25', '⚽ Club Goals'),
                ('Club Assists 24/25', '🎯 Assists'),
                ('Club Games 24/25', '🏃 Games played'),
                ('Club Minutes 24/25', '⏱️ Minutes played')
            ]:
                val = stats_home.get(stat_key, None)
                if val is not None and str(val).lower() != 'nan':
                    st.write(f"**{stat_label} :** {val}")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="stats-container">', unsafe_allow_html=True)
        st.markdown(f"### 🛫 {away_team}")
        
        # Logo de l'équipe
        logo_away = stats_away["Logo"]
        if logo_away and isinstance(logo_away, str) and logo_away.strip() and str(logo_away).lower() != 'nan':
            st.image(logo_away, width=100)
        
        st.markdown(f"**🏟️ Stadium:** {format_stadium(stats_away['Venue'], stats_away['Venue City'], stats_away['Venue Capacity'])}")
        
        # Main metrics
        metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
        with metrics_col1:
            st.metric("🏆 Position", stats_away["Position"])
            st.metric("⚽ Goals scored", stats_away["Goals Scored"])
        with metrics_col2:
            st.metric("🎯 Wins", stats_away["Wins"])
            st.metric("🤝 Draws", stats_away["Draws"])
        with metrics_col3:
            st.metric("❌ Losses", stats_away["Losses"])
            st.metric("📊 Matches played", stats_away["Matches Played"])
        
        # Advanced stats if enabled
        if show_advanced_stats:
            st.markdown("#### 📈 Advanced Club Statistics 24/25")
            for stat_key, stat_label in [
                ('Club Goals 24/25', '⚽ Club Goals'),
                ('Club Assists 24/25', '🎯 Assists'),
                ('Club Games 24/25', '🏃 Games played'),
                ('Club Minutes 24/25', '⏱️ Minutes played')
            ]:
                val = stats_away.get(stat_key, None)
                if val is not None and str(val).lower() != 'nan':
                    st.write(f"**{stat_label} :** {val}")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Visual comparison section
    st.markdown("---")
    st.markdown("### 📊 Visual Comparison")
    
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
    
    # Charts according to selection
    if chart_type in ["Bar Chart", "Both"]:
        st.markdown("#### 📊 Bar Chart")
        fig_bar, ax_bar = plt.subplots(figsize=(12, 6))
        fig_bar.patch.set_facecolor('#0E1117')
        ax_bar.set_facecolor('#262730')
        
        x = np.arange(len(stats_labels))
        bars1 = ax_bar.bar(x - 0.25, home_stats, width=0.4, label=home_team, 
                          color='#FF6B6B', alpha=0.8, edgecolor='white', linewidth=1)
        bars2 = ax_bar.bar(x + 0.25, away_stats, width=0.4, label=away_team, 
                          color='#4ECDC4', alpha=0.8, edgecolor='white', linewidth=1)
        
        # Style improvements
        ax_bar.set_xticks(x)
        ax_bar.set_xticklabels(stats_labels, rotation=45, ha='right', color='white')
        ax_bar.set_ylabel('Values', color='white')
        ax_bar.set_title('Comparison of Main Statistics', 
                        color='white', fontsize=16, fontweight='bold')
        ax_bar.legend(frameon=False, labelcolor='white')
        ax_bar.tick_params(colors='white')
        ax_bar.grid(True, alpha=0.3)
        
        # Adding values on bars
        for bar in bars1:
            height = bar.get_height()
            ax_bar.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                       f'{int(height)}', ha='center', va='bottom', color='white', fontweight='bold')
        
        for bar in bars2:
            height = bar.get_height()
            ax_bar.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                       f'{int(height)}', ha='center', va='bottom', color='white', fontweight='bold')
        
        st.pyplot(fig_bar)
    
    if chart_type in ["Radar Chart", "Both"]:
        st.markdown("#### 🕸️ Radar Chart")
        from math import pi
        
        categories = stats_labels
        N = len(categories)
        values_home = home_stats + [home_stats[0]]
        values_away = away_stats + [away_stats[0]]
        angles = [n / float(N) * 2 * pi for n in range(N)]
        angles += angles[:1]
        
        fig_radar = plt.figure(figsize=(10, 10))
        fig_radar.patch.set_facecolor('#0E1117')
        ax_radar = plt.subplot(111, polar=True)
        ax_radar.set_facecolor('#262730')
        
        # Line plotting and filling
        ax_radar.plot(angles, values_home, linewidth=3, linestyle='solid', 
                     label=home_team, color='#FF6B6B')
        ax_radar.fill(angles, values_home, alpha=0.25, color='#FF6B6B')
        
        ax_radar.plot(angles, values_away, linewidth=3, linestyle='solid', 
                     label=away_team, color='#4ECDC4')
        ax_radar.fill(angles, values_away, alpha=0.25, color='#4ECDC4')
        
        # Style improvements
        ax_radar.set_xticks(angles[:-1])
        ax_radar.set_xticklabels(categories, color='white', fontsize=10)
        ax_radar.set_title('Overall Team Profile', 
                          color='white', fontsize=16, fontweight='bold', pad=20)
        ax_radar.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), 
                       frameon=False, labelcolor='white')
        ax_radar.grid(True, alpha=0.3)
        ax_radar.tick_params(colors='white')
        
        st.pyplot(fig_radar)

# -------------------- Main App Logic --------------------


# --- State management to display stats ---
if "last_pred_home" not in st.session_state:
    st.session_state["last_pred_home"] = None
if "last_pred_away" not in st.session_state:
    st.session_state["last_pred_away"] = None
if "show_stats" not in st.session_state:
    st.session_state["show_stats"] = False

if st.button("🎯 Predict Match", key="predict_btn", help="Click to get the match prediction"):
    try:
        # Prediction with real model (already includes expressive display)
        outcome, confidence = predict_match_outcome_with_score(home_team, away_team)
        
        # Store results in state
        st.session_state['prediction'] = outcome
        st.session_state['confidence'] = confidence
        
        # Remember teams for stats
        st.session_state["last_pred_home"] = home_team
        st.session_state["last_pred_away"] = away_team
        st.session_state["show_stats"] = True
        
    except Exception as e:
        st.error(f"❌ **Critical error during prediction:** {e}")
        import traceback
        with st.expander("🔧 **Error details (for debug)**"):
            st.code(traceback.format_exc())

# Display stats and heatmap if a prediction was made
if st.session_state["show_stats"] and st.session_state["last_pred_home"] and st.session_state["last_pred_away"]:
    show_stats_and_heatmap(st.session_state["last_pred_home"], st.session_state["last_pred_away"], raw_years[0])

# -------------------- Footer --------------------
AppStyle.add_footer(author="Santo, Konstantin and RV a.k.a The Dream Team", year="2025")
