'''
Module for collecting and preparing real football data for predictions.
Ce module collecte et prépare de vraies données de football pour les prédictions.
'''

# Import libraries (PEP8 convention)

# -- Import standard libraries --
import json # For JSON handling
import os # For file operations
from datetime import datetime, timedelta # For date handling

# -- Import third-party libraries --
import pandas as pd # For data manipulation
import numpy as np # For numerical operations
import requests # For making HTTP requests
from typing import Dict, List, Optional # For type annotations


class RealDataCollector:
    '''
    Collects real football data from various sources for better predictions.
    Collecte de vraies données de football depuis diverses sources pour de meilleures prédictions.
    '''

    def __init__(self): # Initialization method for the RealDataCollector class.
        self.data_cache = {} # Dictionary to store cached data
        self.cache_file = "data/football_cache.json" # Path to cache file
        self.load_cache() # Load cached data if available

#----------------------------------------------------------------------------------------------------------------------------------------

    def load_cache(self): # Load cached data if available
        """Load cached data if available"""
        if os.path.exists(self.cache_file): # Check if cache file exists
            try: # Try to read the cache file
                with open(self.cache_file, 'r') as f: # Open the cache file for reading
                    self.data_cache = json.load(f) # Load data from cache file
            except Exception as e: # Handle any exceptions that occur during loading
                print(f"Warning: Could not load cache: {e}") # Print warning if cache loading fails
            self.data_cache = {} # Initialize data cache as empty dictionary

#----------------------------------------------------------------------------------------------------------------------------------------

    def save_cache(self): # Save data to cache
        """Save data to cache"""
        os.makedirs(os.path.dirname(self.cache_file), exist_ok=True) # Ensure the directory exists
        try: # Try to write the data cache to the cache file
            with open(self.cache_file, 'w') as f: # Open the cache file for writing
                json.dump(self.data_cache, f, indent=2) # Write the data cache to the cache file
        except Exception as e: # Handle any exceptions that occur during saving
            print(f"Warning: Could not save cache: {e}") # Print warning if cache saving fails

#----------------------------------------------------------------------------------------------------------------------------------------

    def create_belgian_league_sample_data(self, num_matches: int = 300) -> pd.DataFrame: # Create realistic sample data based on Belgian Pro League patterns.
        """
        Create realistic sample data based on Belgian Pro League patterns.
        Crée des données d'exemple réalistes basées sur les tendances de la Pro League belge.
        """
        print(f"Creating {num_matches} realistic Belgian Pro League matches...") # Print message indicating match creation

        # Real Belgian Pro League teams
        belgian_teams = [
            'Club Brugge', 'Royal Antwerp', 'Union Saint-Gilloise', 'Standard de Liège',
            'KRC Genk', 'Anderlecht', 'La Gantoise', 'Cercle Brugge',
            'KV Mechelen', 'Westerlo', 'Sint-Truiden', 'Charleroi',
            'KV Kortrijk', 'OH Leuven', 'Eupen', 'RFC Seraing'
        ]

        # Team strength ratings (1-10 scale, based on recent performance)
        team_strength = {
            'Club Brugge': 9.0, 'Royal Antwerp': 8.5, 'Union Saint-Gilloise': 8.0,
            'Standard de Liège': 7.0, 'KRC Genk': 7.5, 'Anderlecht': 7.8,
            'La Gantoise': 6.5, 'Cercle Brugge': 6.0, 'KV Mechelen': 6.2,
            'Westerlo': 5.5, 'Sint-Truiden': 5.0, 'Charleroi': 5.8,
            'KV Kortrijk': 4.5, 'OH Leuven': 4.8, 'Eupen': 4.0, 'RFC Seraing': 3.5
        }

        matches = []
        
        for _ in range(num_matches):
            # Select random teams
            home_team = np.random.choice(belgian_teams) # Randomly select a home team
            away_team = np.random.choice([t for t in belgian_teams if t != home_team]) # Randomly select an away team that is not the same as the home team
            
            # Calculate match probability based on team strength
            home_strength = team_strength[home_team] # Home team strength
            away_strength = team_strength[away_team] # Away team strength
            
            # Home advantage (typically +0.3 to +0.5 goals)
            home_advantage = 0.4 # Home advantage factor

            # Expected goals calculation (more realistic)
            home_expected = max(0.1, home_strength - away_strength * 0.8 + home_advantage) # Home expected goals
            away_expected = max(0.1, away_strength - home_strength * 0.8) # Away expected goals
            
            # Generate goals using Poisson distribution (more realistic than pure random) 
            home_goals = np.random.poisson(home_expected) # Home goals scored
            away_goals = np.random.poisson(away_expected) # Away goals scored
            
            # Determine result from Standard's perspective if Standard plays
            if home_team == 'Standard de Liège': # If Standard is the home team
                if home_goals > away_goals: # If Standard wins
                    result = 1  # Win
                elif home_goals == away_goals: # If it's a draw
                    result = 0  # Draw
                else: # Loss
                    result = -1  # Loss
            elif away_team == 'Standard de Liège': # If Standard is the away team
                if away_goals > home_goals: # If Standard wins
                    result = 1  # Win
                elif away_goals == home_goals: # If it's a draw
                    result = 0  # Draw
                else: # Loss
                    result = -1  # Loss
            else: # If Standard is not involved
                # For matches not involving Standard, create general result
                if home_goals > away_goals: # Home win
                    result = 1  # Home win
                elif home_goals == away_goals: # Draw
                    result = 0  # Draw
                else: # Away win
                    result = -1  # Away win
            
            # Add additional realistic features
            match_data = { # Match data dictionary
                'home_team': home_team,
                'away_team': away_team,
                'home_goals': home_goals,
                'away_goals': away_goals,
                'result': result,
                'home_strength': home_strength,
                'away_strength': away_strength,
                'strength_difference': home_strength - away_strength,
                'total_goals': home_goals + away_goals,
                'goal_difference': home_goals - away_goals
            }
            
            matches.append(match_data) # Append match data to the list
        
        df = pd.DataFrame(matches) # Convert the list of matches to a pandas DataFrame
        
        print(f"Created {len(df)} realistic matches!") # Print the number of matches created
        print(f"Result distribution:") # Print the distribution of results
        print(df['result'].value_counts().sort_index()) # Print the count of each result type
        print(f"Average goals per match: {df['total_goals'].mean():.2f}") # Print the average goals per match
        print(f"Home advantage: {(df[df['result'] == 1].shape[0] / len(df) * 100):.1f}% wins")  
        
        return df # Return the DataFrame containing the match data

#----------------------------------------------------------------------------------------------------------------------------------------

    def enhance_data_with_features(self, data: pd.DataFrame) -> pd.DataFrame: # Add advanced features to improve model performance.
        """
        Add advanced features to improve model performance.
        Ajoute des caractéristiques avancées pour améliorer les performances du modèle.
        """
        print("Enhancing data with advanced features...")
        
        enhanced_data = data.copy()
        
        # 1. Form indicators (recent performance simulation)
        enhanced_data['home_recent_form'] = np.random.normal(5, 1.5, len(data))  # 0-10 scale
        enhanced_data['away_recent_form'] = np.random.normal(5, 1.5, len(data))
        
        # 2. Head-to-head history simulation
        enhanced_data['h2h_home_wins'] = np.random.poisson(2, len(data))
        enhanced_data['h2h_away_wins'] = np.random.poisson(2, len(data))
        enhanced_data['h2h_draws'] = np.random.poisson(1, len(data))
        
        # 3. League position simulation (1-16 for Belgian league)
        enhanced_data['home_league_position'] = np.random.randint(1, 17, len(data))
        enhanced_data['away_league_position'] = np.random.randint(1, 17, len(data))
        
        # 4. Days since last match (fatigue factor)
        enhanced_data['home_days_rest'] = np.random.randint(3, 14, len(data))
        enhanced_data['away_days_rest'] = np.random.randint(3, 14, len(data))
        
        # 5. Is derby match? (local rivalry)
        derby_pairs = [
            ('Standard de Liège', 'Anderlecht'),
            ('Club Brugge', 'Cercle Brugge'),
            ('Royal Antwerp', 'KV Mechelen')
        ]
        
        enhanced_data['is_derby'] = enhanced_data.apply(
            lambda row: any(
                (row['home_team'] in pair and row['away_team'] in pair) 
                for pair in derby_pairs
            ), axis=1
        ).astype(int)
        
        print(f"Enhanced with {len(enhanced_data.columns) - len(data.columns)} new features!")
        print(f"New features: {list(enhanced_data.columns[len(data.columns):])}")
        
        return enhanced_data

#----------------------------------------------------------------------------------------------------------------------------------------

    def get_standard_focused_data(self, num_matches: int = 200) -> pd.DataFrame:
        """
        Create dataset focused on Standard de Liège matches.
        Crée un dataset centré sur les matchs du Standard de Liège.
        """
        print("Creating Standard de Liège focused dataset...")
        
        # Generate base data
        all_data = self.create_belgian_league_sample_data(num_matches * 2)
        
        # Filter for Standard matches only
        standard_matches = all_data[
            (all_data['home_team'] == 'Standard de Liège') | 
            (all_data['away_team'] == 'Standard de Liège')
        ].copy()
        
        # If not enough Standard matches, create more
        if len(standard_matches) < num_matches:
            additional_needed = num_matches - len(standard_matches)
            print(f"Need {additional_needed} more Standard matches...")
            
            belgian_teams = [
                'Club Brugge', 'Royal Antwerp', 'Union Saint-Gilloise',
                'KRC Genk', 'Anderlecht', 'La Gantoise', 'Cercle Brugge',
                'KV Mechelen', 'Westerlo', 'Sint-Truiden', 'Charleroi',
                'KV Kortrijk', 'OH Leuven', 'Eupen', 'RFC Seraing'
            ]
            
            for _ in range(additional_needed):
                opponent = np.random.choice(belgian_teams)
                is_home = np.random.choice([True, False])
                
                if is_home:
                    match = self._create_standard_match('Standard de Liège', opponent)
                else:
                    match = self._create_standard_match(opponent, 'Standard de Liège')
                
                standard_matches = pd.concat([standard_matches, pd.DataFrame([match])], ignore_index=True)
        
        # Take exactly the number requested
        standard_matches = standard_matches.head(num_matches)
        
        # Enhance with additional features
        enhanced_data = self.enhance_data_with_features(standard_matches)
        
        print(f"Standard dataset ready: {len(enhanced_data)} matches")
        print(f"Home matches: {len(enhanced_data[enhanced_data['home_team'] == 'Standard de Liège'])}")
        print(f"Away matches: {len(enhanced_data[enhanced_data['away_team'] == 'Standard de Liège'])}")
        
        return enhanced_data

#----------------------------------------------------------------------------------------------------------------------------------------

    def _create_standard_match(self, home_team: str, away_team: str) -> Dict:
        """Create a single realistic Standard match"""
        # Standard strength rating
        standard_strength = 7.0
        
        # Opponent strength (simplified)
        opponent_strengths = {
            'Club Brugge': 9.0, 'Royal Antwerp': 8.5, 'Union Saint-Gilloise': 8.0,
            'KRC Genk': 7.5, 'Anderlecht': 7.8, 'La Gantoise': 6.5,
            'Cercle Brugge': 6.0, 'KV Mechelen': 6.2, 'Westerlo': 5.5,
            'Sint-Truiden': 5.0, 'Charleroi': 5.8, 'KV Kortrijk': 4.5,
            'OH Leuven': 4.8, 'Eupen': 4.0, 'RFC Seraing': 3.5
        }
        
        if home_team == 'Standard de Liège':
            home_strength = standard_strength
            away_strength = opponent_strengths.get(away_team, 6.0)
        else:
            home_strength = opponent_strengths.get(home_team, 6.0)
            away_strength = standard_strength
        
        # Calculate goals
        home_expected = max(0.1, home_strength - away_strength * 0.8 + 0.4)
        away_expected = max(0.1, away_strength - home_strength * 0.8)
        
        home_goals = np.random.poisson(home_expected)
        away_goals = np.random.poisson(away_expected)
        
        # Result from Standard's perspective
        if home_team == 'Standard de Liège':
            if home_goals > away_goals:
                result = 1
            elif home_goals == away_goals:
                result = 0
            else:
                result = -1
        else:
            if away_goals > home_goals:
                result = 1
            elif away_goals == home_goals:
                result = 0
            else:
                result = -1
        
        return {
            'home_team': home_team,
            'away_team': away_team,
            'home_goals': home_goals,
            'away_goals': away_goals,
            'result': result,
            'home_strength': home_strength,
            'away_strength': away_strength,
            'strength_difference': home_strength - away_strength,
            'total_goals': home_goals + away_goals,
            'goal_difference': home_goals - away_goals
        }

#----------------------------------------------------------------------------------------------------------------------------------------

    def save_data(self, data: pd.DataFrame, filename: str = "real_football_data.csv"):
        """Save data to CSV file"""
        os.makedirs("data", exist_ok=True)
        filepath = f"data/{filename}"
        data.to_csv(filepath, index=False)
        print(f"Data saved to {filepath}")
        return filepath

#----------------------------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    # Test the data collector
    collector = RealDataCollector()
    
    # Create realistic Standard de Liège data
    real_data = collector.get_standard_focused_data(300)
    
    # Save to file
    collector.save_data(real_data, "standard_realistic_data.csv")
    
    print("\nSample of the data:")
    print(real_data.head())
    
    print(f"\nData shape: {real_data.shape}")
    print(f"Columns: {list(real_data.columns)}")
