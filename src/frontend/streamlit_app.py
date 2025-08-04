import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import pandas as pd
from styles.app_styles import AppStyle
from components.team_components import get_team_logo, get_team_stadium

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
# Specify date format to avoid parsing warning (format: DD/MM/YYYY)
df_full['Year'] = pd.to_datetime(df_full['Date'], format='%d/%m/%Y', errors='coerce').dt.year
# Format seasons as 'Season YYYY-YYYY'
raw_years = sorted(df_full['Year'].dropna().unique(), reverse=True)
season_labels = [f"Season {int(y)}-{int(y)+1}" for y in raw_years]
season_labels.insert(0, "All seasons")

# Import backend components
sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/../.."))
from utils.data_io import load_model
from src.backend.model.my_model import ModelTrainer
from src.frontend.components.feature_prepare import FeaturePrepare

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
    
    .team-logo {
        width: 80px !important;
        height: 80px !important;
        object-fit: contain;
        border-radius: 10px;
        background: rgba(255, 255, 255, 0.1);
        padding: 5px;
        margin-bottom: 15px;
        display: block;
        margin-left: auto;
        margin-right: auto;
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
    "Anderlecht", "Club Brugge", "Standard", "Genk", "Oostende",
    "Gent", "Charleroi", "Cercle Brugge", "Antwerp", "Mouscron",
    "Westerlo", "Union Saint-Gilloise", "Mechelen",
    "Eupen", "Kortrijk", "RWD Molenbeek", "St. Truiden", "Seraing", "Waregem",
    "Waasland-Beveren", "Beerschot VA", "Dender", "Oud-Heverlee Leuven"
]


# -------------------- Team Selection Section --------------------
st.markdown("### 🏟️ Team Selection")

col1, col2, col3 = st.columns([2, 1, 2])

with col1:
    st.markdown("#### 🏠 Home Team")
    home_team = st.selectbox(
        "Choose the home team",
        jupiler_teams,
        key="home",
        help="The team playing at home"
    )

with col2:
    st.markdown('<div class="vs-divider">VS</div>', unsafe_allow_html=True)

with col3:
    st.markdown("#### 🛫 Away Team")
    away_options = [team for team in jupiler_teams if team != home_team]
    away_team = st.selectbox(
        "Choose the away team",
        away_options,
        key="away",
        help="The team playing away"
    )

# -------------------- Model Loading Section --------------------
@st.cache_resource(show_spinner=False)
def load_trained_model():
    """Load the pre-trained model from the backend"""
    
    try:
        # Load the model from the joblib file
        model = load_model("trained_model.joblib")
        
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

def generate_realistic_score(home_team, away_team, prediction, features):
    """Generate realistic score based on ML prediction and team features from your architecture"""
    # Extract goal-related features from your FeaturePrepare system
    home_goals_last5 = features.get('home_goals_scored_last5', 1.5)
    away_goals_last5 = features.get('away_goals_scored_last5', 1.5)
    
    # Extract additional features from your feature engineering
    home_form = features.get('home_form_last5', 0.5)
    away_form = features.get('away_form_last5', 0.5)
    home_win_rate = features.get('home_win_rate_last10', 0.5)
    away_win_rate = features.get('away_win_rate_last10', 0.5)
    
    # Base goals from recent form (your ML features)
    base_home = max(0.5, home_goals_last5 / 5)  # Average per match
    base_away = max(0.5, away_goals_last5 / 5)  # Average per match
    
    # Apply form and win rate adjustments from your features
    form_adjustment_home = (home_form - 0.5) * 0.5  # Form impact
    form_adjustment_away = (away_form - 0.5) * 0.5
    
    win_rate_adjustment_home = (home_win_rate - 0.5) * 0.3  # Win rate impact
    win_rate_adjustment_away = (away_win_rate - 0.5) * 0.3
    
    # Adjust base goals with your ML features
    adjusted_home = base_home + form_adjustment_home + win_rate_adjustment_home
    adjusted_away = base_away + form_adjustment_away + win_rate_adjustment_away
    
    # Apply ML prediction logic
    if prediction == 'H':  # Home wins
        home_goals = max(1, int(round(adjusted_home + 0.7)))  # Boost home team
        away_goals = max(0, int(round(adjusted_away - 0.2)))  # Slight reduction away
    elif prediction == 'A':  # Away wins  
        home_goals = max(0, int(round(adjusted_home - 0.2)))  # Slight reduction home
        away_goals = max(1, int(round(adjusted_away + 0.5)))  # Boost away team
    else:  # Draw
        # For draws, balance the scores based on your ML features
        avg_goals = (adjusted_home + adjusted_away) / 2
        home_goals = max(0, int(round(avg_goals)))
        away_goals = home_goals  # Same score for draw
    
    # Ensure realistic scores (0-5 range typically)
    home_goals = min(home_goals, 5)
    away_goals = min(away_goals, 5)
    
    return home_goals, away_goals

def predict_fallback_with_stats(home, away):
    """Fallback statistical prediction when ML model can't be used"""
    # Simple statistical approach using dataset
    home_matches = df_full[df_full['HomeTeam'] == home]
    away_matches = df_full[df_full['AwayTeam'] == away]
    
    home_goals_avg = home_matches['FTHG'].mean() if len(home_matches) > 0 else 1.5
    away_goals_avg = away_matches['FTAG'].mean() if len(away_matches) > 0 else 1.5
    
    # Simple prediction
    home_goals = max(0, int(round(home_goals_avg + 0.3)))  # Home advantage
    away_goals = max(0, int(round(away_goals_avg)))
    
    if home_goals > away_goals:
        outcome = f'🏠 VICTORY {home.upper()}'
    elif away_goals > home_goals:
        outcome = f'🛫 VICTORY {away.upper()}'
    else:
        outcome = '🤝 DRAW'
    
    return outcome, 60.0  # Lower confidence for fallback


def predict_match_outcome_with_score(home, away):
    """Predict match outcome using the real trained model and feature pipeline"""
    try:
        # Initialize feature preparer
        feature_preparer = FeaturePrepare()
        
        # Prepare features using your existing system
        try:
            features = feature_preparer.prepare_features(home_team=home, away_team=away)
        except ValueError as e:
            return predict_fallback_with_stats(home, away)
        
        # Create DataFrame for prediction (required by your model)
        input_df = pd.DataFrame([features])
        
        # Make prediction using the real trained model
        prediction = model_trainer.model.predict(input_df)[0]
        
        # Get prediction probabilities for confidence
        try:
            probabilities = model_trainer.model.predict_proba(input_df)[0]
            confidence = max(probabilities) * 100  # Convert to percentage
        except:
            confidence = 75.0  # Fallback confidence
        
        # Interpret result (your model returns H/A/D)
        if prediction == 'H':
            outcome = f'🏠 VICTORY {home.upper()}'
            emoji = '🔥'
            description = f"The home team {home} should win this match!"
        elif prediction == 'A':
            outcome = f'🛫 VICTORY {away.upper()}'  
            emoji = '⚡'
            description = f"The away team {away} is favored for this match!"
        else:  # 'D'
            outcome = '🤝 DRAW'
            emoji = '⚖️'
            description = "A balanced match that should end in a draw"
        
        # Generate realistic score based on prediction and team stats
        predicted_home_goals, predicted_away_goals = generate_realistic_score(
            home, away, prediction, features
        )
            
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
            <h3 style="margin: 15px 0; font-size: 1.8rem; color: #FFD700;">
                🥅 PREDICTED SCORE: {predicted_home_goals} - {predicted_away_goals} 🥅
            </h3>
            <p style="margin: 15px 0; font-size: 1.2rem; opacity: 0.9;">{description}</p>
            <div style="margin-top: 20px;">
                <span style="font-size: 1.1rem; font-weight: bold;">AI Confidence: {confidence:.1f}%</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Technical details with style
        with st.expander("🔍 **Technical Prediction Details**", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                st.metric("🏠 Home Team", home)
                st.metric("🎯 ML Prediction", prediction)
                st.metric("🥅 Predicted Home Goals", predicted_home_goals)
            with col2:
                st.metric("🛫 Away Team", away)
                st.metric("🤖 AI Confidence", f"{confidence:.1f}%")
                st.metric("🥅 Predicted Away Goals", predicted_away_goals)
                
            # Show key features used by the model
            st.markdown("#### 🧠 **Key Features Used by AI Model**")
            
            # Display most important features from your FeaturePrepare system
            important_features = {}
            feature_categories = {
                'Form & Performance': ['form', 'win_rate', 'performance'],
                'Goals & Scoring': ['goals', 'scored', 'conceded'],
                'Head-to-Head': ['head2head', 'h2h', 'vs'],
                'Recent Matches': ['last5', 'last10', 'recent'],
                'Home/Away Stats': ['home_', 'away_']
            }
            
            for category, keywords in feature_categories.items():
                category_features = {}
                for key, value in features.items():
                    if any(keyword in key.lower() for keyword in keywords):
                        category_features[key] = value
                
                if category_features:
                    st.markdown(f"**{category}:**")
                    feature_df = pd.DataFrame(list(category_features.items()), 
                                            columns=['Feature', 'Value'])
                    st.dataframe(feature_df, use_container_width=True)
                    st.markdown("---")
            
            st.info("ℹ️ Prediction made using your complete ML architecture: FeaturePrepare → ModelTrainer → RandomForest with GridSearchCV optimization")
        
        return outcome, confidence
        
    except Exception as e:
        st.error(f"❌ **ML Prediction error:** {e}")
        st.info("🔄 Falling back to statistical prediction...")
        return predict_fallback_with_stats(home, away)

# --- Real stats extraction from dataset ---

def get_real_stats(team, df):
    # Strict filtering by season already applied in df
    home = df[df['HomeTeam'] == team]
    away = df[df['AwayTeam'] == team]
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
    
    # Get stadium info from our static dictionary
    stadium_info = get_team_stadium(team)
    venue = stadium_info["name"]
    venue_city = stadium_info["city"] 
    venue_capacity = stadium_info["capacity"]
    
    # --- Enriched data from dataset (if available) ---
    # Take the first home or away row for team info
    team_row = pd.concat([home, away]).iloc[0] if (len(home) > 0 or len(away) > 0) else None
    if team_row is not None:
        # Advanced club stats (scoring) - only if columns exist
        club_goals = team_row.get('club_goals_24_25', None)
        club_assists = team_row.get('club_assists_24_25', None)
        club_games = team_row.get('club_games_24_25', None)
        club_minutes = team_row.get('club_minutes_24_25', None)
    else:
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
        
        show_advanced_stats = st.checkbox("📈 Advanced statistics", value=False)
        chart_type = st.radio(
            "📊 Visualization type",
            ["Bar Chart", "Radar Chart", "Performance Comparison", "All Charts"],
            index=0
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
        st.markdown(f"### 🏠 {home_team}")
        
        # Logo de l'équipe depuis notre dictionnaire
        logo_home = get_team_logo(home_team)
        if logo_home:
            st.markdown(f'<img src="{logo_home}" class="team-logo" alt="{home_team} logo">', unsafe_allow_html=True)
        else:
            st.markdown('<div style="text-align: center; font-size: 4rem; margin: 15px 0;">🏠</div>', unsafe_allow_html=True)
        
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
    
    with col2:
        st.markdown(f"### 🛫 {away_team}")
        
        # Logo de l'équipe depuis notre dictionnaire
        logo_away = get_team_logo(away_team)
        if logo_away:
            st.markdown(f'<img src="{logo_away}" class="team-logo" alt="{away_team} logo">', unsafe_allow_html=True)
        else:
            st.markdown('<div style="text-align: center; font-size: 4rem; margin: 15px 0;">🛫</div>', unsafe_allow_html=True)
        
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
    if chart_type == "Bar Chart" or chart_type == "All Charts":
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
    
    if chart_type == "Performance Comparison" or chart_type == "All Charts":
        st.markdown("#### 🏆 Performance Comparison")
        
        # Create two pie charts side by side for win/draw/loss comparison
        fig_perf, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        fig_perf.patch.set_facecolor('#0E1117')
        
        # Home team pie chart
        home_results = [stats_home["Wins"], stats_home["Draws"], stats_home["Losses"]]
        home_labels = ['Wins', 'Draws', 'Losses']
        home_colors = ['#2ECC71', '#F39C12', '#E74C3C']
        
        # Only show pie chart if there are matches played
        if sum(home_results) > 0:
            wedges1, texts1, autotexts1 = ax1.pie(home_results, labels=home_labels, colors=home_colors,
                                                  autopct='%1.1f%%', startangle=90, textprops={'color': 'white'})
            ax1.set_title(f'{home_team}\nTotal Matches: {stats_home["Matches Played"]}', 
                         color='white', fontsize=14, fontweight='bold')
        else:
            ax1.text(0.5, 0.5, 'No Data\nAvailable', transform=ax1.transAxes, 
                    ha='center', va='center', color='white', fontsize=16)
            ax1.set_title(f'{home_team}\nNo Matches', color='white', fontsize=14, fontweight='bold')
        
        # Away team pie chart
        away_results = [stats_away["Wins"], stats_away["Draws"], stats_away["Losses"]]
        away_labels = ['Wins', 'Draws', 'Losses']
        
        # Only show pie chart if there are matches played
        if sum(away_results) > 0:
            wedges2, texts2, autotexts2 = ax2.pie(away_results, labels=away_labels, colors=home_colors,
                                                  autopct='%1.1f%%', startangle=90, textprops={'color': 'white'})
            ax2.set_title(f'{away_team}\nTotal Matches: {stats_away["Matches Played"]}', 
                         color='white', fontsize=14, fontweight='bold')
        else:
            ax2.text(0.5, 0.5, 'No Data\nAvailable', transform=ax2.transAxes, 
                    ha='center', va='center', color='white', fontsize=16)
            ax2.set_title(f'{away_team}\nNo Matches', color='white', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        st.pyplot(fig_perf)
        
        # Additional comparison metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### ⚽ Attack vs Defense")
            home_attack_def = [stats_home["Goals Scored"], stats_home["Goals Conceded"]]
            away_attack_def = [stats_away["Goals Scored"], stats_away["Goals Conceded"]]
            
            fig_att_def, ax_att_def = plt.subplots(figsize=(8, 6))
            fig_att_def.patch.set_facecolor('#0E1117')
            ax_att_def.set_facecolor('#262730')
            
            x_teams = [home_team, away_team]
            goals_scored = [stats_home["Goals Scored"], stats_away["Goals Scored"]]
            goals_conceded = [stats_home["Goals Conceded"], stats_away["Goals Conceded"]]
            
            x_pos = np.arange(len(x_teams))
            bars1 = ax_att_def.bar(x_pos - 0.2, goals_scored, width=0.4, label='Goals Scored', 
                                  color='#2ECC71', alpha=0.8)
            bars2 = ax_att_def.bar(x_pos + 0.2, goals_conceded, width=0.4, label='Goals Conceded', 
                                  color='#E74C3C', alpha=0.8)
            
            ax_att_def.set_xticks(x_pos)
            ax_att_def.set_xticklabels(x_teams, color='white')
            ax_att_def.set_ylabel('Goals', color='white')
            ax_att_def.set_title('Attack vs Defense', color='white', fontweight='bold')
            ax_att_def.legend(frameon=False, labelcolor='white')
            ax_att_def.tick_params(colors='white')
            ax_att_def.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar in bars1:
                height = bar.get_height()
                ax_att_def.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                               f'{int(height)}', ha='center', va='bottom', color='white', fontweight='bold')
            
            for bar in bars2:
                height = bar.get_height()
                ax_att_def.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                               f'{int(height)}', ha='center', va='bottom', color='white', fontweight='bold')
            
            st.pyplot(fig_att_def)
        
        with col2:
            st.markdown("#### 📊 Win Rate Comparison")
            
            # Calculate win rates
            home_win_rate = (stats_home["Wins"] / max(stats_home["Matches Played"], 1)) * 100
            away_win_rate = (stats_away["Wins"] / max(stats_away["Matches Played"], 1)) * 100
            
            fig_win_rate, ax_win_rate = plt.subplots(figsize=(8, 6))
            fig_win_rate.patch.set_facecolor('#0E1117')
            ax_win_rate.set_facecolor('#262730')
            
            teams = [home_team, away_team]
            win_rates = [home_win_rate, away_win_rate]
            colors = ['#FF6B6B', '#4ECDC4']
            
            bars = ax_win_rate.bar(teams, win_rates, color=colors, alpha=0.8)
            ax_win_rate.set_ylabel('Win Rate (%)', color='white')
            ax_win_rate.set_title('Win Rate Comparison', color='white', fontweight='bold')
            ax_win_rate.tick_params(colors='white')
            ax_win_rate.grid(True, alpha=0.3, axis='y')
            
            # Add percentage labels
            for bar, rate in zip(bars, win_rates):
                ax_win_rate.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                               f'{rate:.1f}%', ha='center', va='bottom', color='white', fontweight='bold')
            
            st.pyplot(fig_win_rate)
        
        with col3:
            st.markdown("#### 🎯 Goals per Match")
            
            home_goals_per_match = stats_home["Goals Scored"] / max(stats_home["Matches Played"], 1)
            away_goals_per_match = stats_away["Goals Scored"] / max(stats_away["Matches Played"], 1)
            
            fig_gpm, ax_gpm = plt.subplots(figsize=(8, 6))
            fig_gpm.patch.set_facecolor('#0E1117')
            ax_gpm.set_facecolor('#262730')
            
            teams = [home_team, away_team]
            goals_per_match = [home_goals_per_match, away_goals_per_match]
            
            bars = ax_gpm.bar(teams, goals_per_match, color=['#FF6B6B', '#4ECDC4'], alpha=0.8)
            ax_gpm.set_ylabel('Goals per Match', color='white')
            ax_gpm.set_title('Scoring Efficiency', color='white', fontweight='bold')
            ax_gpm.tick_params(colors='white')
            ax_gpm.grid(True, alpha=0.3, axis='y')
            
            # Add value labels
            for bar, gpm in zip(bars, goals_per_match):
                ax_gpm.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                           f'{gpm:.2f}', ha='center', va='bottom', color='white', fontweight='bold')
            
            st.pyplot(fig_gpm)
    
    if chart_type == "Radar Chart" or chart_type == "All Charts":
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
