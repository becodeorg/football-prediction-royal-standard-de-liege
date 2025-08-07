"""
⚽ Football Prediction App - Compact Version
Professional Streamlit app for Jupiler Pro League predictions
"""

import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os
from typing import Tuple, Optional
import plotly.graph_objects as go
import plotly.express as px

# ========== CONFIGURATION ==========
current_dir = os.path.dirname(__file__)

APP_CONFIG = {
    "page_title": "JPL Prediction Pro",
    "page_icon": "⚽",
    "layout": "wide"
}

# Import styles
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/../.."))
from styles.app_styles import AppStyle
from utils.data_io import load_csv

# ========== CONSTANTS ==========
class Constants:
    """Application constants derived from data analysis."""
    DEFAULT_FALLBACK_CONFIDENCE = 50.0  # Only for statistical fallback
    AVG_GOALS_PER_TEAM = 1.5  # League average when no data
    MIN_MATCHES_FOR_RELIABLE_DATA = 5  # Minimum matches for quality analysis
    HOME_ADVANTAGE_THRESHOLD = 0.1  # Minimum home advantage to consider

# ========== DATA MANAGEMENT ==========
class DataManager:
    """Unified data management for teams and predictions."""
    
    def __init__(self, selected_season="All Seasons"):
        """
        Initialize DataManager with season-specific data loading.
        
        Args:
            selected_season (str): Season to filter data by. Default "All Seasons"
                for no filtering. Format: "YYYY-YYYY" (e.g., "2019-2020")
                
        Attributes:
            selected_season (str): Currently selected season
            team_stats (pd.DataFrame): Team statistics data
            teams (list): Sorted list of available team names
            match_data (pd.DataFrame): Historical match data
            model: Trained ML model for predictions
        """
        self.selected_season = selected_season
        self.team_stats = self._load_team_stats()
        self.teams = sorted(self.team_stats['Team'].tolist()) if self.team_stats is not None else []
        self.match_data = self._load_match_data()
        self.model = self._load_model()
        
    def get_available_seasons(self):
        """Get list of available seasons from the data."""
        if self.match_data.empty:
            return ["All Seasons"]
        
        # Calculate seasons from dates
        self.match_data['Date'] = pd.to_datetime(self.match_data['Date'], format='%Y-%m-%d')
        
        def get_season(date):
            if date.month >= 7:  # Juillet à décembre
                return f'{date.year}-{date.year+1}'
            else:  # Janvier à juin
                return f'{date.year-1}-{date.year}'
        
        self.match_data['Season'] = self.match_data['Date'].apply(get_season)
        seasons = sorted(self.match_data['Season'].unique())
        return ["All Seasons"] + seasons
    
    def filter_data_by_season(self, df):
        """Filter dataframe by selected season."""
        if self.selected_season == "All Seasons" or df.empty:
            return df
        
        # Ensure Season column exists
        if 'Season' not in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')
            def get_season(date):
                if date.month >= 7:
                    return f'{date.year}-{date.year+1}'
                else:
                    return f'{date.year-1}-{date.year}'
            df['Season'] = df['Date'].apply(get_season)
        
        return df[df['Season'] == self.selected_season]
    
    def _load_team_stats(self):
        """
        Load team statistics from CSV file.
        
        Returns:
            pd.DataFrame or None: DataFrame containing team statistics with columns:
                - Team: Team name
                - Attack: Attack rating (0-100)
                - Midfield: Midfield rating (0-100) 
                - Defense: Defense rating (0-100)
                - Logo_URL: URL to team logo image
                Returns None if file not found or loading fails.
                
        Raises:
            FileNotFoundError: When jupiler_teams_data.csv is not found
            pd.errors.EmptyDataError: When CSV file is empty
            Exception: For other data loading errors
        """
        try:
            path = os.path.join(current_dir, 'components', 'jupiler_teams_data.csv')
            return pd.read_csv(path)
        except FileNotFoundError:
            st.error("❌ Team statistics file not found")
        except pd.errors.EmptyDataError:
            st.error("❌ Team statistics file is empty")
        except Exception as e:
            st.error(f"❌ Error loading team stats: {e}")
        return None
    
    def _load_match_data(self):
        """
        Load historical match data from B1_old.csv file.
        
        Returns:
            pd.DataFrame: DataFrame containing match data with columns:
                - Date: Match date (YYYY-MM-DD format)
                - HomeTeam: Home team name
                - AwayTeam: Away team name  
                - FTHG: Full Time Home Goals
                - FTAG: Full Time Away Goals
                - FTR: Full Time Result (1=Home win, 0=Draw, -1=Away win)
                - Year: Extracted year from date
                - Plus additional feature columns for ML model
                Returns empty DataFrame if loading fails.
                
        Raises:
            FileNotFoundError: When B1_old.csv is not found
            pd.errors.EmptyDataError: When CSV file is empty
            Exception: For other data loading errors
        """
        try:
            path = os.path.abspath(os.path.join(current_dir, '../../data/prepared/B1_old.csv'))
            df = pd.read_csv(path)
            df['Year'] = pd.to_datetime(df['Date'], format='%Y-%m-%d', errors='coerce').dt.year
            return df
        except FileNotFoundError:
            st.error("❌ Match data file not found")
        except pd.errors.EmptyDataError:
            st.error("❌ Match data file is empty")
        except Exception as e:
            st.error(f"❌ Error loading match data: {e}")
        return pd.DataFrame()

    def _load_model(self):
        """
        Load trained ML model from joblib file.
        
        Returns:
            sklearn model or None: Trained model object with predict() and 
                potentially predict_proba() methods. Returns None if model
                not found or loading fails.
                
        Raises:
            FileNotFoundError: When trained_model.joblib is not found
            Exception: For other model loading errors
        """
        try:
            path = os.path.join(current_dir, '..', 'backend', 'model', 'trained_model', 'trained_model.joblib')
            return joblib.load(path)
        except FileNotFoundError:
            st.warning("⚠️ ML model not found, using statistical fallback")
        except Exception as e:
            st.warning(f"⚠️ Error loading ML model: {e}")
        return None
    
    def get_team_data(self, team_name: str) -> Tuple[int, int, int, Optional[str]]:
        """
        Retrieve comprehensive team statistics and logo URL.
        
        Args:
            team_name (str): Name of the team to get data for
            
        Returns:
            Tuple[int, int, int, Optional[str]]: Four-element tuple containing:
                - Attack rating (0-100): Team's offensive capability
                - Midfield rating (0-100): Team's midfield strength  
                - Defense rating (0-100): Team's defensive capability
                - Logo URL (str or None): URL to team logo image, None if not available
                
        Note:
            If team not found or no stats available, returns league averages.
            League averages are calculated from all teams in the dataset.
        """
        if self.team_stats is None:
            # Use league averages if no stats available
            league_avg = self._calculate_league_averages()
            return league_avg, league_avg, league_avg, None
        
        team_row = self.team_stats[self.team_stats['Team'] == team_name]
        if not team_row.empty:
            row = team_row.iloc[0]
            return int(row['Attack']), int(row['Midfield']), int(row['Defense']), row['Logo_URL']
        
        # Use league averages as fallback
        league_avg = self._calculate_league_averages()
        return league_avg, league_avg, league_avg, None
    
    def _calculate_league_averages(self) -> int:
        """
        Calculate average team statistics across the entire league.
        
        Returns:
            int: Average rating across all teams and all categories (Attack, Midfield, Defense).
                Returns 75 as fallback if no team statistics are available.
                
        Note:
            Used as fallback when individual team stats are unavailable.
            Averages are computed from Attack, Midfield, and Defense ratings.
        """
        if self.team_stats is not None and not self.team_stats.empty:
            avg_attack = int(self.team_stats['Attack'].mean())
            avg_midfield = int(self.team_stats['Midfield'].mean()) 
            avg_defense = int(self.team_stats['Defense'].mean())
            return int((avg_attack + avg_midfield + avg_defense) / 3)
        return 75  # Only if absolutely no data
    
    def predict_match(self, home_team: str, away_team: str) -> Tuple[str, float]:
        """
    Predict match outcome using ML model or statistical fallback.
    
    Args:
        home_team: Name of the home team
        away_team: Name of the away team
        
    Returns:
        Tuple of (outcome_description, confidence_percentage)
        
    Raises:
        ValueError: If teams are invalid
    """
        try:
            # Try ML prediction
            if self.model is not None:
                features = self._prepare_features(home_team, away_team)
                if features:
                    input_df = pd.DataFrame([features])
                    prediction = self.model.predict(input_df)[0]
                    
                    # Get confidence directly from model probabilities
                    try:
                        probabilities = self.model.predict_proba(input_df)[0]
                        confidence = max(probabilities) * 100
                    except:
                        # If model doesn't support predict_proba, estimate from prediction certainty
                        confidence = self._estimate_confidence_from_features(features)
                    
                    outcome = self._interpret_prediction(prediction, home_team, away_team)
                    self._generate_score(home_team, away_team, prediction)
                    return outcome, confidence

            # Fallback prediction   Because we don't have enough data matches for some teams
            return self._fallback_prediction(home_team, away_team)
            
        except:
            return self._fallback_prediction(home_team, away_team)
    
    def _prepare_features(self, home_team: str, away_team: str):
        """Prepare features for ML model."""
        try:
            df = load_csv(filedir="prepared", filename="B1_old.csv")
            feature_cols = [col for col in df.columns if col not in ("Date", "FTR")]
            match_row = df[(df["HomeTeam"] == home_team) & (df["AwayTeam"] == away_team)]
            if not match_row.empty:
                return match_row.iloc[0][feature_cols].to_dict()
        except:
            pass
        return None
    
    def _interpret_prediction(self, prediction, home_team: str, away_team: str) -> str:
        """Convert prediction to readable outcome."""
        if prediction == 1 or prediction == 'H':
            return f' {home_team.upper()} Wins'
        elif prediction == -1 or prediction == 'A':
            return f' {away_team.upper()} Wins'
        else:
            return ' DRAW'
    
    def _generate_score(self, home_team: str, away_team: str, prediction):
        """Generate realistic score based on team historical performance."""
        # Get actual historical goal averages for these teams
        home_avg = self._get_team_goal_average(home_team, home=True)
        away_avg = self._get_team_goal_average(away_team, home=False)
        
        # Use team names as seed for consistency
        seed_value = hash(f"{home_team}{away_team}") % 1000000
        np.random.seed(seed_value)
        
        # Generate goals based on team averages (more realistic than random 0-4)
        home_goals = max(0, int(np.random.poisson(home_avg)))
        away_goals = max(0, int(np.random.poisson(away_avg)))
        
        # Adjust based on model prediction
        if prediction == 1 or prediction == 'H':  # Home win
            if home_goals <= away_goals:
                home_goals = away_goals + 1
        elif prediction == -1 or prediction == 'A':  # Away win
            if away_goals <= home_goals:
                away_goals = home_goals + 1
        else:  # Draw prediction
            # For draws, make scores closer
            avg_goals = (home_goals + away_goals) // 2
            home_goals = away_goals = avg_goals
        
        self._set_match_score(home_goals, away_goals)
    
    def _fallback_prediction(self, home_team: str, away_team: str) -> Tuple[str, float]:
        """
        Advanced statistical fallback prediction using multiple data layers.
        
        This method is used when the ML model is unavailable or fails to predict.
        It employs a sophisticated multi-layer approach combining team performance,
        FIFA-style ratings, head-to-head history, and form analysis.
        
        Args:
            home_team (str): Name of the home team
            away_team (str): Name of the away team
            
        Returns:
            Tuple[str, float]: Two-element tuple containing:
                - Outcome description (str): Human-readable prediction like "VICTORY TEAM_NAME" or "DRAW"
                - Confidence percentage (float): Prediction confidence (50.0-95.0%)
                
        Algorithm Layers:
            1. Team Performance Analysis: Historical goal scoring and conceding patterns
            2. FIFA-style Stats: Attack, Midfield, Defense ratings comparison
            3. Head-to-Head Factor: Historical matchup results between these teams
            4. Strength Calculation: Weighted combination based on data quality
            5. Home Advantage: Statistical advantage for playing at home
            6. Score Generation: Realistic score using Poisson distribution
            7. Confidence Estimation: Multi-factor confidence calculation
            
        Note:
            Confidence is calculated based on team strength difference, data quality,
            and team consistency. Higher confidence indicates more reliable prediction.
        """
        
        # Layer 1: Direct team performance analysis
        home_team_performance = self._analyze_team_performance(home_team, home=True)
        away_team_performance = self._analyze_team_performance(away_team, home=False)
        
        # Layer 2: Team stats comparison (FIFA-style prediction)
        home_stats = self.get_team_data(home_team)
        away_stats = self.get_team_data(away_team)
        
        # Layer 3: Historical head-to-head if available
        h2h_factor = self._get_head_to_head_factor(home_team, away_team)
        
        # Layer 4: League position and form analysis
        home_strength = self._calculate_team_strength(home_team, home_stats, home_team_performance)
        away_strength = self._calculate_team_strength(away_team, away_stats, away_team_performance)
        
        # Apply home advantage
        home_advantage = self._calculate_home_advantage()
        home_strength += home_advantage
        
        # Generate realistic score
        home_goals, away_goals = self._generate_logical_score(
            home_strength, away_strength, home_team_performance, away_team_performance
        )
        
        # Determine outcome with confidence based on strength difference
        strength_diff = abs(home_strength - away_strength)
        confidence = self._calculate_smart_confidence(strength_diff, home_team_performance, away_team_performance)
        
        if home_goals > away_goals:
            outcome = f' VICTORY {home_team.upper()}'
        elif away_goals > home_goals:
            outcome = f' VICTORY {away_team.upper()}'
        else:
            outcome = ' DRAW'
        
        self._set_match_score(home_goals, away_goals)
        
        return outcome, confidence
    
    def _set_match_score(self, home_goals: int, away_goals: int):
        """Helper to set match score in session state."""
        st.session_state["home_goals"] = home_goals
        st.session_state["away_goals"] = away_goals
    
    def _get_team_goal_average(self, team_name: str, home: bool) -> float:
        """Get actual goal average for a team from historical data."""
        if home:
            team_matches = self.match_data[self.match_data['HomeTeam'] == team_name]
            goals = team_matches['FTHG'].mean() if len(team_matches) > 0 else Constants.AVG_GOALS_PER_TEAM
        else:
            team_matches = self.match_data[self.match_data['AwayTeam'] == team_name]
            goals = team_matches['FTAG'].mean() if len(team_matches) > 0 else Constants.AVG_GOALS_PER_TEAM
        
        return goals
    
    def _calculate_home_advantage(self) -> float:
        """Calculate home advantage from actual match data."""
        if self.match_data.empty:
            return 0.0
        
        home_avg = self.match_data['FTHG'].mean()
        away_avg = self.match_data['FTAG'].mean()
        return home_avg - away_avg
    
    def _estimate_confidence_from_features(self, features: dict) -> float:
        """Estimate confidence based on feature strength when model doesn't support predict_proba."""
        # Use actual feature values to estimate confidence
        feature_values = list(features.values())
        if not feature_values:
            return Constants.DEFAULT_FALLBACK_CONFIDENCE
        
        # Normalize features and calculate confidence
        # Higher feature variance = lower confidence
        mean_val = np.mean(feature_values)
        std_val = np.std(feature_values)
        
        # Convert to confidence percentage (60-95% range based on data consistency)
        confidence = max(60, min(95, 100 - (std_val * 10)))
        return confidence
    
    def _analyze_team_performance(self, team_name: str, home: bool) -> dict:
        """
        Comprehensive team performance analysis with multiple metrics.
        
        Analyzes team performance considering goals scored/conceded, recent form,
        consistency, and data quality. All analysis is filtered by the selected season.
        
        Args:
            team_name (str): Name of the team to analyze
            home (bool): If True, analyze home performance; if False, analyze away performance
            
        Returns:
            dict: Performance metrics dictionary with the following keys:
                - avg_goals_scored (float): Average goals scored per match
                - avg_goals_conceded (float): Average goals conceded per match  
                - form (float): Recent form factor (0.0-1.0, where 1.0 is excellent)
                - consistency (float): Team consistency (0.0-1.0, where 1.0 is very consistent)
                - data_quality (float): Quality of available data (0.0-1.0, where 1.0 is excellent)
                
        Algorithm:
            1. Filter matches by season and home/away status
            2. Calculate goal scoring and conceding averages
            3. Analyze recent form (last 5 matches)
            4. Calculate consistency based on goal variance
            5. Assess data quality based on number of matches and recency
            
        Note:
            Returns default neutral values if no match data is available.
            Data quality affects how much this analysis is trusted in predictions.
        """
        # Filter data by season first
        filtered_data = self.filter_data_by_season(self.match_data)
        
        if home:
            team_matches = filtered_data[filtered_data['HomeTeam'] == team_name]
            goals_for = team_matches['FTHG'].tolist() if len(team_matches) > 0 else []
            goals_against = team_matches['FTAG'].tolist() if len(team_matches) > 0 else []
        else:
            team_matches = filtered_data[filtered_data['AwayTeam'] == team_name]
            goals_for = team_matches['FTAG'].tolist() if len(team_matches) > 0 else []
            goals_against = team_matches['FTHG'].tolist() if len(team_matches) > 0 else []
        
        if not goals_for:
            return {
                'avg_goals_scored': Constants.AVG_GOALS_PER_TEAM,
                'avg_goals_conceded': Constants.AVG_GOALS_PER_TEAM,
                'form': 0.5,  # Neutral form
                'consistency': 0.5,
                'data_quality': 0.0  # No data
            }
        
        avg_scored = np.mean(goals_for)
        avg_conceded = np.mean(goals_against)
        
        # Recent form (last 5 matches if available)
        recent_matches = min(Constants.MIN_MATCHES_FOR_RELIABLE_DATA, len(goals_for))
        recent_scored = np.mean(goals_for[-recent_matches:]) if recent_matches > 0 else avg_scored
        recent_form = min(1.0, recent_scored / max(0.1, avg_scored))  # Form factor 0-1
        
        # Consistency (lower std = more consistent)
        consistency = 1.0 / (1.0 + np.std(goals_for)) if len(goals_for) > 1 else 0.5
        
        # Enhanced data quality calculation
        quality_factor = min(1.0, len(goals_for) / 15.0)  # 15+ matches = excellent data
        recency_bonus = 0.1 if len(goals_for) >= Constants.MIN_MATCHES_FOR_RELIABLE_DATA else 0.0
        
        return {
            'avg_goals_scored': avg_scored,
            'avg_goals_conceded': avg_conceded,
            'form': recent_form,
            'consistency': consistency,
            'data_quality': min(1.0, quality_factor + recency_bonus)
        }
    
    def _get_head_to_head_factor(self, home_team: str, away_team: str) -> float:
        """Calculate head-to-head historical factor, filtered by season."""
        # Filter data by season first
        filtered_data = self.filter_data_by_season(self.match_data)
        
        h2h_matches = filtered_data[
            ((filtered_data['HomeTeam'] == home_team) & (filtered_data['AwayTeam'] == away_team)) |
            ((filtered_data['HomeTeam'] == away_team) & (filtered_data['AwayTeam'] == home_team))
        ]
        
        if len(h2h_matches) == 0:
            return 0.0  # No historical data
        
        # Count wins for each team
        home_wins = len(h2h_matches[
            ((h2h_matches['HomeTeam'] == home_team) & (h2h_matches['FTHG'] > h2h_matches['FTAG'])) |
            ((h2h_matches['AwayTeam'] == home_team) & (h2h_matches['FTAG'] > h2h_matches['FTHG']))
        ])
        
        total_matches = len(h2h_matches)
        home_win_rate = home_wins / total_matches
        
        # Return advantage factor (-0.5 to +0.5)
        return (home_win_rate - 0.5)
    
    def _calculate_team_strength(self, team_name: str, team_stats: tuple, performance: dict) -> float:
        """Calculate comprehensive team strength."""
        att, mid, def_, _ = team_stats
        
        # FIFA stats component (0-100 scale, normalize to 0-1)
        fifa_strength = (att + mid + def_) / 300.0
        
        # Performance component
        goal_ratio = performance['avg_goals_scored'] / max(0.1, performance['avg_goals_conceded'])
        perf_strength = min(1.0, goal_ratio / 3.0)  # Normalize excellent teams
        
        # Weighted combination based on data quality
        data_quality = performance['data_quality']
        
        if data_quality > 0.7:  # Good data, trust performance more
            strength = 0.3 * fifa_strength + 0.7 * perf_strength
        elif data_quality > 0.3:  # Some data, balanced approach
            strength = 0.5 * fifa_strength + 0.5 * perf_strength
        else:  # Poor data, rely on FIFA stats
            strength = 0.8 * fifa_strength + 0.2 * perf_strength
        
        # Apply form factor
        strength *= (0.8 + 0.4 * performance['form'])  # Form can boost/reduce by 20%
        
        return strength
    
    def _generate_logical_score(self, home_strength: float, away_strength: float, 
                               home_perf: dict, away_perf: dict) -> Tuple[int, int]:
        """Generate realistic score based on team strengths."""
        
        # Base expectation from team averages
        home_expected = home_perf['avg_goals_scored']
        away_expected = away_perf['avg_goals_scored']
        
        # Adjust based on relative strength
        strength_ratio = home_strength / max(0.1, away_strength)
        
        if strength_ratio > 1.3:  # Home team much stronger
            home_expected *= 1.2
            away_expected *= 0.8
        elif strength_ratio < 0.7:  # Away team much stronger
            home_expected *= 0.8
            away_expected *= 1.2
        
        # Use team names for consistent randomness
        seed_value = hash(f"{home_strength}{away_strength}") % 1000000
        np.random.seed(seed_value)
        
        # Generate using Poisson distribution
        home_goals = max(0, int(np.random.poisson(max(0.1, home_expected))))
        away_goals = max(0, int(np.random.poisson(max(0.1, away_expected))))
        
        return home_goals, away_goals
    
    def _calculate_smart_confidence(self, strength_diff: float, home_perf: dict, away_perf: dict) -> float:
        """Calculate confidence based on multiple factors."""
        
        # Base confidence from strength difference
        if strength_diff > 0.3:  # Large difference
            base_confidence = 85.0
        elif strength_diff > 0.15:  # Moderate difference
            base_confidence = 75.0
        elif strength_diff > 0.05:  # Small difference
            base_confidence = 65.0
        else:  # Very close match
            base_confidence = 55.0
        
        # Adjust for data quality
        avg_data_quality = (home_perf['data_quality'] + away_perf['data_quality']) / 2
        data_bonus = avg_data_quality * 15  # Up to 15% bonus for good data
        
        # Adjust for consistency
        avg_consistency = (home_perf['consistency'] + away_perf['consistency']) / 2
        consistency_bonus = avg_consistency * 10  # Up to 10% bonus for consistent teams
        
        final_confidence = base_confidence + data_bonus + consistency_bonus
        
        return min(95.0, max(50.0, final_confidence))  # Cap between 50-95%

# ========== CHART FUNCTIONS ==========
def create_goals_comparison_chart(home_team: str, away_team: str, data_manager: DataManager):
    """
    Create bar chart showing goals scored and goals conceded for each team.
    
    Args:
        home_team (str): Name of home team
        away_team (str): Name of away team  
        data_manager (DataManager): Data manager instance
    Returns:
        plotly.graph_objects.Figure: Goals scored/conceded comparison chart
    """
    # Get total goals scored and conceded for each team
    # Utilise les données filtrées par saison
    match_data = data_manager.filter_data_by_season(data_manager.match_data)

    def get_total_goals(team_name, home=True, scored=True):
        if home:
            team_matches = match_data[match_data['HomeTeam'] == team_name]
            if scored:
                return int(team_matches['FTHG'].sum())
            else:
                return int(team_matches['FTAG'].sum())
        else:
            team_matches = match_data[match_data['AwayTeam'] == team_name]
            if scored:
                return int(team_matches['FTAG'].sum())
            else:
                return int(team_matches['FTHG'].sum())

    home_goals_scored = get_total_goals(home_team, home=True, scored=True)
    home_goals_conceded = get_total_goals(home_team, home=True, scored=False)
    away_goals_scored = get_total_goals(away_team, home=False, scored=True)
    away_goals_conceded = get_total_goals(away_team, home=False, scored=False)

    teams = [home_team, away_team]
    scored = [home_goals_scored, away_goals_scored]
    conceded = [home_goals_conceded, away_goals_conceded]

    fig = go.Figure()

    # Goals scored bars
    fig.add_trace(go.Bar(
        name='Wins',
        x=teams,
        y=scored,
        marker_color='#4CAF50',  # Green for wins
        text=[str(val) for val in scored],
        textposition='auto',
        textfont=dict(size=14, color='white'),
        hovertemplate='<b>%{x}</b><br>Wins: %{y}<extra></extra>'
    ))

    fig.add_trace(go.Bar(
        name='Losses',
        x=teams,
        y=conceded,
        marker_color='#f44336',  # Red for losses
        text=[str(val) for val in conceded],
        textposition='auto',
        textfont=dict(size=14, color='white'),
        hovertemplate='<b>%{x}</b><br>Losses: %{y}<extra></extra>'
    ))

    fig.update_layout(
        title={
            'text': ' Total Goals Scored & Conceded',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18, 'color': 'white', 'family': 'Arial, sans-serif'}
        },
        xaxis_title='Teams',
        yaxis_title='Total Goals',
        barmode='group',
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white', family='Arial, sans-serif'),
        height=400,
        margin=dict(l=50, r=50, t=80, b=50),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.5,
            xanchor="center",
            x=0.5
        )
    )

    return fig

def create_head_to_head_pie_chart(home_team: str, away_team: str, data_manager: DataManager):
    """
    Create a pie chart showing head-to-head results breakdown for selected teams and season.
    Returns:
        plotly.graph_objects.Figure: Pie chart of win/draw percentages
    """
    match_data = data_manager.filter_data_by_season(data_manager.match_data)
    h2h_matches = match_data[
        ((match_data['HomeTeam'] == home_team) & (match_data['AwayTeam'] == away_team)) |
        ((match_data['HomeTeam'] == away_team) & (match_data['AwayTeam'] == home_team))
    ]
    total = len(h2h_matches)
    if total == 0:
        labels = ['No Data']
        values = [1]
        colors = ['#888']
    else:
        home_wins = len(h2h_matches[
            ((h2h_matches['HomeTeam'] == home_team) & (h2h_matches['FTHG'] > h2h_matches['FTAG'])) |
            ((h2h_matches['AwayTeam'] == home_team) & (h2h_matches['FTAG'] > h2h_matches['FTHG']))
        ])
        away_wins = len(h2h_matches[
            ((h2h_matches['HomeTeam'] == away_team) & (h2h_matches['FTHG'] > h2h_matches['FTAG'])) |
            ((h2h_matches['AwayTeam'] == away_team) & (h2h_matches['FTAG'] > h2h_matches['FTHG']))
        ])
        draws = len(h2h_matches[h2h_matches['FTHG'] == h2h_matches['FTAG']])
        labels = [f'{home_team} Wins', 'Draws', f'{away_team} Wins']
        values = [home_wins, draws, away_wins]
        colors = ['#667eea', '#FF9800', '#f093fb']
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        marker_colors=colors,
        textinfo='label+percent',
        textfont=dict(size=12, color='white'),
        hovertemplate='<b>%{label}</b><br>Percentage: %{percent}<extra></extra>'
    )])
    fig.update_layout(
        title={
            'text': f' Head-to-Head Results ({home_team} vs {away_team})',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18, 'color': 'white', 'family': 'Arial, sans-serif'}
        },
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white', family='Arial, sans-serif'),
        height=400,
        margin=dict(l=50, r=50, t=80, b=50),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.5,
            xanchor="center",
            x=0.5
        )
    )
    return fig

def create_placeholder_pie_chart():
    """
    Create placeholder pie chart for future implementation.
    
    Returns:
        plotly.graph_objects.Figure: Placeholder pie chart
    """
    # Placeholder data
    labels = ['Data A', 'Data B', 'Data C']
    values = [40, 35, 25]
    colors = ['#667eea', '#f093fb', '#4CAF50']
    
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        marker_colors=colors,
        textinfo='label+percent',
        textfont=dict(size=12, color='white'),
        hovertemplate='<b>%{label}</b><br>Percentage: %{percent}<extra></extra>'
    )])
    
    fig.update_layout(
        title={
            'text': '📊 Percentage Analysis',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18, 'color': 'white', 'family': 'Arial, sans-serif'}
        },
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white', family='Arial, sans-serif'),
        height=400,
        margin=dict(l=50, r=50, t=80, b=50),
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="middle",
            y=0.5,
            xanchor="left",
            x=1.05
        )
    )
    
    return fig

def create_placeholder_bar_chart():
    """
    Create placeholder bar chart for future implementation.
    
    Returns:
        plotly.graph_objects.Figure: Placeholder bar chart
    """
    # Placeholder data
    categories = ['Category 1', 'Category 2', 'Category 3']
    values = [65, 80, 45]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=categories,
        y=values,
        marker_color='#4CAF50',
        text=values,
        textposition='auto',
        textfont=dict(size=14, color='white'),
        hovertemplate='<b>%{x}</b><br>Value: %{y}<extra></extra>',
        showlegend=False
    ))
    
    fig.update_layout(
        title={
            'text': '📈 Future Analysis',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18, 'color': 'white', 'family': 'Arial, sans-serif'}
        },
        xaxis_title='Categories',
        yaxis_title='Values',
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white', family='Arial, sans-serif'),
        height=400,
        margin=dict(l=50, r=50, t=80, b=50)
    )
    
    return fig

def create_home_away_performance_chart(home_team: str, away_team: str, data_manager: DataManager):
    """
    Bar chart comparing home and away performance (wins/losses) for each team in the selected season.
    """
    match_data = data_manager.filter_data_by_season(data_manager.match_data)
    def get_results(team, home):
        if home:
            matches = match_data[match_data['HomeTeam'] == team]
            wins = len(matches[matches['FTHG'] > matches['FTAG']])
            losses = len(matches[matches['FTHG'] < matches['FTAG']])
        else:
            matches = match_data[match_data['AwayTeam'] == team]
            wins = len(matches[matches['FTAG'] > matches['FTHG']])
            losses = len(matches[matches['FTAG'] < matches['FTHG']])
        return wins, losses
    home_home_wins, home_home_losses = get_results(home_team, True)
    home_away_wins, home_away_losses = get_results(home_team, False)
    away_home_wins, away_home_losses = get_results(away_team, True)
    away_away_wins, away_away_losses = get_results(away_team, False)
    teams = [home_team, away_team]
    home_wins = [home_home_wins, away_home_wins]
    away_wins = [home_away_wins, away_away_wins]
    home_losses = [home_home_losses, away_home_losses]
    away_losses = [home_away_losses, away_away_losses]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name='Home Wins',
        x=teams,
        y=home_wins,
        marker_color='#4CAF50',
        text=home_wins,
        textposition='auto',
        textfont=dict(size=14, color='white'),
        hovertemplate='<b>%{x}</b><br>Home Wins: %{y}<extra></extra>'
    ))
    fig.add_trace(go.Bar(
        name='Away Wins',
        x=teams,
        y=away_wins,
        marker_color='#2196F3',
        text=away_wins,
        textposition='auto',
        textfont=dict(size=14, color='white'),
        hovertemplate='<b>%{x}</b><br>Away Wins: %{y}<extra></extra>'
    ))
    fig.add_trace(go.Bar(
        name='Home Losses',
        x=teams,
        y=home_losses,
        marker_color='#f44336',
        text=home_losses,
        textposition='auto',
        textfont=dict(size=14, color='white'),
        hovertemplate='<b>%{x}</b><br>Home Losses: %{y}<extra></extra>'
    ))
    fig.add_trace(go.Bar(
        name='Away Losses',
        x=teams,
        y=away_losses,
        marker_color='#FF9800',
        text=away_losses,
        textposition='auto',
        textfont=dict(size=14, color='white'),
        hovertemplate='<b>%{x}</b><br>Away Losses: %{y}<extra></extra>'
    ))
    fig.update_layout(
        title={
            'text': 'paHome vs Away Performance',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18, 'color': 'white', 'family': 'Arial, sans-serif'}
        },
        xaxis_title='Teams',
        yaxis_title='Number of Matches',
        barmode='group',
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white', family='Arial, sans-serif'),
        height=420,
        margin=dict(l=50, r=50, t=80, b=50),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.5,
            xanchor="center",
            x=0.5,
            font=dict(size=12),
            itemclick="toggleothers",
            itemdoubleclick="toggle"
        )
    )
    return fig

def render_analytics_charts(home_team: str, away_team: str, data_manager: DataManager):
    """
    Render comprehensive analytics charts section.
    
    Args:
        home_team (str): Name of home team
        away_team (str): Name of away team
        data_manager (DataManager): Data manager instance
    """
    if not (home_team and away_team):
        return
    
    st.markdown("---")
    st.markdown(
        "<h2 style='text-align: center; color: #667eea; margin-bottom: 30px;'> Team Analytics Dashboard</h2>", 
        unsafe_allow_html=True
    )
    
    # Create three columns for charts
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Goals comparison chart
        fig1 = create_goals_comparison_chart(home_team, away_team, data_manager)
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        # Head-to-head results pie chart
        fig2 = create_head_to_head_pie_chart(home_team, away_team, data_manager)
        st.plotly_chart(fig2, use_container_width=True)
    
    with col3:
        # Home vs Away performance bar chart
        fig3 = create_home_away_performance_chart(home_team, away_team, data_manager)
        st.plotly_chart(fig3, use_container_width=True)

# ========== UI FUNCTIONS ==========
def render_team_selection(data_manager: DataManager) -> Tuple[str, str]:
    """
    Render team selection interface with home/away dropdown menus.
    
    Creates a three-column layout with home team selector on the left,
    empty space in the middle, and away team selector on the right.
    Ensures the same team cannot be selected for both home and away.
    
    Args:
        data_manager (DataManager): DataManager instance containing team list
        
    Returns:
        Tuple[str, str]: Selected team names as (home_team, away_team)
        
    UI Components:
        - Home team selectbox with all available teams
        - Away team selectbox with all teams except the selected home team
        - Responsive column layout [2, 1, 2] proportions
    """
    col1, col2, col3 = st.columns([2, 1, 2])
    
    with col1:
        st.markdown("#### Home Team")
        home_team = st.selectbox("Choose home team", data_manager.teams, key="home")

    with col3:
        st.markdown("#### Away Team")
        away_options = [team for team in data_manager.teams if team != home_team]
        away_team = st.selectbox("Choose away team", away_options, key="away")
    
    return home_team, away_team

def render_team_preview(home_team: str, away_team: str, data_manager: DataManager):
    """Render team preview with logos and stats."""
    if not (home_team and away_team):
        return
    
    col1, col2, col3 = st.columns([2, 1, 2])
    
    # Home team
    with col1:
        render_team_card(home_team, data_manager, "🏠")
    
    # VS section
    with col2:
        st.markdown('<div class="vs-divider">VS</div>', unsafe_allow_html=True)
        render_prediction_results(home_team, away_team)
    
    # Away team
    with col3:
        render_team_card(away_team, data_manager, "✈️")

def render_team_card(team_name: str, data_manager: DataManager, default_emoji: str):
    """
    Render a team card with logo, emoji, and key team statistics.
    
    Displays team information in a compact card format including team logo,
    default emoji indicator, and core performance metrics (attack, midfield, defense).
    
    Args:
        team_name (str): Full name of the team to display
        data_manager (DataManager): DataManager instance for retrieving team data
        default_emoji (str): Emoji to display alongside team name (🏠 for home, ✈️ for away)
        
    UI Components:
        - Team logo image (if available) or placeholder
        - Team name with emoji indicator
        - Attack, Midfield, Defense ratings display
        - Responsive card layout with proper spacing
        
    Data Flow:
        1. Retrieves team data via data_manager.get_team_data()
        2. Displays logo image or uses placeholder
        3. Shows team name with contextual emoji
        4. Presents core team statistics in readable format
    """
    att, mid, def_, logo_url = data_manager.get_team_data(team_name)
    
    # Logo
    if logo_url:
        st.markdown(f'<img src="{logo_url}" class="team-logo" alt="{team_name} logo">', unsafe_allow_html=True)
    
    # Team name
    st.markdown(f"<div class='team-name'>{team_name}</div>", unsafe_allow_html=True)
    
    # Last 5 matches
    try:
        historical_data = load_csv(filedir="prepared", filename="B1_old.csv")
        team_matches = historical_data[
            (historical_data['HomeTeam'] == team_name) | 
            (historical_data['AwayTeam'] == team_name)
        ].tail(5).sort_values('Date', ascending=False)
        
        if not team_matches.empty:
            st.markdown("<div class='recent-matches-title'>Last 5 matches:</div>", unsafe_allow_html=True)
            
            # Build HTML string with all circles in one row
            circles_list = []
            
            for idx, (_, match) in enumerate(team_matches.iterrows()):
                if idx < 5:  # Ensure we don't exceed 5 matches
                    home_team_match = match['HomeTeam']
                    away_team_match = match['AwayTeam']
                    result = match['FTR']
                    
                    # Determine result for this team
                    if team_name == home_team_match:
                        if result == 1:  # Home win
                            result_text = "W"
                            result_color = "#4CAF50"
                        elif result == -1:  # Home loss
                            result_text = "L"
                            result_color = "#f44336"
                        else:  # Draw
                            result_text = "D"
                            result_color = "#FF9800"
                    else:  # Away team
                        if result == -1:  # Away win
                            result_text = "W"
                            result_color = "#4CAF50"
                        elif result == 1:  # Away loss
                            result_text = "L"
                            result_color = "#f44336"
                        else:  # Draw
                            result_text = "D"
                            result_color = "#FF9800"
                    
                    circle_html = f'<div style="width: 24px; height: 24px; border-radius: 50%; background-color: {result_color}; color: white; display: flex; align-items: center; justify-content: center; font-weight: bold; font-size: 0.7rem; flex-shrink: 0;">{result_text}</div>'
                    circles_list.append(circle_html)
            
            # Combine all circles
            all_circles = ''.join(circles_list)
            final_html = f'<div style="display: flex; justify-content: center; gap: 10px; margin-bottom: 15px; align-items: center;">{all_circles}</div>'
            
            st.markdown(final_html, unsafe_allow_html=True)
    except Exception as e:
        st.markdown("<div style='text-align: center; font-size: 0.8rem; color: #ff6b6b; margin: 10px 0; font-style: italic;'>No recent matches available</div>", unsafe_allow_html=True)
    
    # Team stats
    st.markdown(f"""
    <div class="fifa-stats-container">
        <div class="fifa-stats-row">
            <div class="fifa-stat-item">
                <div class="fifa-stat-label">ATT</div>
                <div class="fifa-stat-value">{att}</div>
            </div>
            <div class="fifa-stat-item">
                <div class="fifa-stat-label">MID</div>
                <div class="fifa-stat-value">{mid}</div>
            </div>
            <div class="fifa-stat-item">
                <div class="fifa-stat-label">DEF</div>
                <div class="fifa-stat-value">{def_}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_prediction_results(home_team: str, away_team: str):
    """
    Render match prediction results with detailed outcome analysis.
    
    Displays comprehensive prediction results including match outcome,
    confidence levels, predicted scores, and betting recommendations.
    Only renders if all required prediction data is available in session state.
    
    Args:
        home_team (str): Name of the home team
        away_team (str): Name of the away team
        
    Session State Requirements:
        - 'prediction': Match outcome ('Home Win', 'Draw', 'Away Win')
        - 'confidence': Prediction confidence percentage (0-100)
        - 'home_goals': Predicted goals for home team
        - 'away_goals': Predicted goals for away team
        
    UI Components:
        - Match outcome header with team names
        - Confidence indicator with visual styling
        - Predicted score display
        - Betting recommendation section
        - Color-coded outcome styling (green/blue/red for win/draw/loss)
        
    Visual Elements:
        - High confidence (>70%): Green styling
        - Medium confidence (50-70%): Orange styling  
        - Low confidence (<50%): Red styling
        - Responsive layout with centered content
    """
    if not all(key in st.session_state for key in ["prediction", "confidence", "home_goals", "away_goals"]):
        return
    
    outcome = st.session_state['prediction']
    confidence = st.session_state['confidence']
    home_goals = st.session_state["home_goals"]
    away_goals = st.session_state["away_goals"]
    
    st.markdown(f"""
    <div style="margin-top: 20px; padding: 15px; text-align: center;">
        <h2 style="margin: 0; font-size: 1.8rem; color: white; font-weight: bold;">
            {outcome}
        </h2>
        <h3 style="margin: 15px 0; font-size: 4.2rem; color: #FFD700; font-weight: bold;">
            {home_goals} - {away_goals}
        </h3>
        <div style="font-size: 1.1rem; color: #ccc; font-weight: bold;">
            Confidence: {confidence:.1f}%
        </div>
    </div>
    """, unsafe_allow_html=True)

def clear_prediction_if_teams_changed(home_team: str, away_team: str):
    """Clear prediction results if teams changed."""
    if (st.session_state.get("last_pred_home") != home_team or 
        st.session_state.get("last_pred_away") != away_team):
        
        for key in ["prediction", "confidence", "home_goals", "away_goals"]:
            if key in st.session_state:
                del st.session_state[key]

# ========== SIDEBAR FUNCTIONS ==========
def render_sidebar():
    """
    Render application sidebar with season selection and filtering options.
    
    Creates sidebar interface for selecting football seasons and applying
    various filters to the prediction model. Provides user controls for
    customizing the prediction experience.
    
    UI Components:
        - Season selector dropdown with available seasons
        - Additional filter options (if implemented)
        - Navigation and settings controls
        
    Data Flow:
        1. Creates temporary DataManager to retrieve available seasons
        2. Renders season selection widget in sidebar
        3. Updates session state with selected filters
        4. Provides context for main application filtering
        
    Session State Effects:
        - Updates selected season for data filtering
        - Maintains filter preferences across page refreshes
        - Influences prediction model data scope
    """
    
    # Get seasons using temporary data manager
    temp_data_manager = DataManager("All Seasons")
    available_seasons = temp_data_manager.get_available_seasons()
    
    # Season selection
    st.sidebar.markdown("### Sélection de Saison")
    selected_season = st.sidebar.selectbox(
        "Choisir une saison:",
        available_seasons,
        index=0,  # "All Seasons" par défaut
        key="season_selector",
        help="Filtrer les statistiques par saison. 'All Seasons' affiche toutes les données disponibles."
    )
    
    # Season statistics (only if different from "All Seasons")
    if selected_season != "All Seasons":
        season_data_manager = DataManager(selected_season)
        filtered_data = season_data_manager.filter_data_by_season(season_data_manager.match_data)
        
        if not filtered_data.empty:
            total_matches = len(filtered_data)
            st.sidebar.metric("Matchs dans la saison", total_matches)
            
            # Most goals in a match
            if 'FTHG' in filtered_data.columns and 'FTAG' in filtered_data.columns:
                max_goals = (filtered_data['FTHG'] + filtered_data['FTAG']).max()
                st.sidebar.metric("Plus grand nombre de buts", int(max_goals))
    
    return selected_season

# ========== MAIN APP ==========
def main():
    """Main application function."""
    # Page setup
    st.set_page_config(**APP_CONFIG)
    AppStyle.apply_custom_css()
    
    # Render sidebar and get selected options
    selected_season = render_sidebar()
    
    # Initialize data manager with selected season
    data_manager = DataManager(selected_season)
    
    # Initialize session state
    for var in ["last_pred_home", "last_pred_away"]:
        if var not in st.session_state:
            st.session_state[var] = None
    
    # Main header with season info
    if selected_season != "All Seasons":
        season_text = f" - Saison {selected_season}"
    else:
        season_text = " - Toutes Saisons"
    
    st.markdown(f"""
    <div style="text-align: center; margin-bottom: 30px;">
        <h1 style="color: #667eea; font-family: 'Poppins', sans-serif; font-weight: 700; font-size: 3rem; margin-bottom: 10px;">
             Jupiler Pro League Match Predictions
        </h1>
        <p style="color: rgba(255,255,255,0.8); font-size: 1.2rem; margin: 0;">
            Analyse Prédictive Professionnelle{season_text}
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Team selection
    home_team, away_team = render_team_selection(data_manager)
    
    # Clear results if teams changed
    clear_prediction_if_teams_changed(home_team, away_team)
    
    # Team preview
    render_team_preview(home_team, away_team, data_manager)
    
    # Prediction button
    st.markdown('<div class="predict-container">', unsafe_allow_html=True)
    _, col2, _ = st.columns([1, 1, 1])
    with col2:
        if st.button(" Predict Match", key="predict_btn", use_container_width=True):
            try:
                outcome, confidence = data_manager.predict_match(home_team, away_team)
                
                st.session_state.update({
                    'prediction': outcome,
                    'confidence': confidence,
                    'last_pred_home': home_team,
                    'last_pred_away': away_team
                })
                
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ **Prediction Error:** {str(e)}")
                st.info("💡 Try selecting different teams or refresh the page")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Analytics charts section
    render_analytics_charts(home_team, away_team, data_manager)
    
    # Footer
    AppStyle.add_footer(author="Santo, Konstantin and RV a.k.a The Dream Team", year="2025")

if __name__ == "__main__":
    main()
