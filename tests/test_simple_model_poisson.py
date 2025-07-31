'''
Simple test for the SimplePredictor model.
'''

import sys
import os
# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.real_data_collector import RealDataCollector
from model.simple_predictor_poisson import SimplePredictorPoisson
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
import pandas as pd
import numpy as np

def test_simple_predictor():
    """
    Detailed and comprehensive test of the SimplePredictor
    """
    print("DETAILED TEST OF SIMPLE PREDICTOR")
    print("=" * 50)

    #----------------------------------------------------------------------------------------------------------------------------------------
    
    # 1. DATA GENERATION
    print("\n1. DATA GENERATION")
    print("-" * 30)
    
    collector = RealDataCollector()
    data = collector.get_standard_focused_data(125000)
    
    print(f"Dataset created: {len(data)} matches")
    print(f"Available columns: {len(data.columns)}")
    print(f"Home matches: {len(data[data['home_team'] == 'Standard de Liège'])}")
    print(f"Away matches: {len(data[data['away_team'] == 'Standard de Liège'])}")
    
    # Result distribution
    result_counts = data['result'].value_counts().sort_index()
    print(f"\nResult distribution:")
    print(f"   Defeats (-1): {result_counts.get(-1, 0)} ({result_counts.get(-1, 0)/len(data)*100:.1f}%)")
    print(f"   Draws (0):    {result_counts.get(0, 0)} ({result_counts.get(0, 0)/len(data)*100:.1f}%)")
    print(f"   Wins (1):     {result_counts.get(1, 0)} ({result_counts.get(1, 0)/len(data)*100:.1f}%)")

    # --- Sélectionne uniquement les features réalistes pour l'entraînement et la prédiction ---
    realistic_features = [
        'home_recent_form', 'away_recent_form',
        'h2h_home_wins', 'h2h_away_wins',
        'home_league_position', 'away_league_position'
    ]
    # On garde aussi les cibles
    data = data[realistic_features + ['home_goals', 'away_goals', 'result']]
    
    #----------------------------------------------------------------------------------------------------------------------------------------

    # 2. MODEL INITIALIZATION
    print("\n2. MODEL INITIALIZATION")
    print("-" * 30)
    
    predictor = SimplePredictorPoisson()
    print("SimplePredictor initialized")
    
    #----------------------------------------------------------------------------------------------------------------------------------------

    # 3. DATA PREPARATION
    print("\n3. DATA PREPARATION")
    print("-" * 30)
    

    # --- Poisson Regression pipeline ---
    # 3. DATA PREPARATION
    print("\n3. DATA PREPARATION")
    print("-" * 30)
    y_home = data['home_goals']
    y_away = data['away_goals']
    # On ne passe que les features réalistes au préprocessing
    X = predictor.preprocess_features(data[realistic_features])
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_home_train, y_home_test = train_test_split(X, y_home, test_size=0.2, random_state=42)
    _, _, y_away_train, y_away_test = train_test_split(X, y_away, test_size=0.2, random_state=42)

    print(f"Data split:")
    print(f"   Training: {len(X_train)} samples")
    print(f"   Test: {len(X_test)} samples")
    print(f"   Features used: {len(X_train.columns)}")
    print(f"   Feature list: {list(X_train.columns)}")

    # 4. TRAINING
    print("\n4. MODEL TRAINING")
    print("-" * 30)
    predictor.train(X_train, y_home_train, y_away_train)
    print("Poisson Regression models trained successfully.")

    # 5. PREDICTION & EVALUATION
    print("\n5. MODEL EVALUATION")
    print("-" * 30)
    home_pred, away_pred = predictor.predict_scores(X_test)
    home_pred_rounded = np.round(home_pred).astype(int)
    away_pred_rounded = np.round(away_pred).astype(int)
    result_pred = np.sign(home_pred_rounded - away_pred_rounded)
    result_true = np.sign(y_home_test.values - y_away_test.values)
    accuracy = (result_pred == result_true).mean()
    print(f"Test accuracy (victoire/nul/défaite): {accuracy:.2%}")

    # Analyse détaillée des erreurs de prédiction
    test_results = X_test.copy()
    test_results['true_result'] = result_true
    test_results['predicted_result'] = result_pred
    errors = test_results[test_results['true_result'] != test_results['predicted_result']]
    print(f"\nNombre de matchs mal prédits : {len(errors)} / {len(test_results)}")
    print("Exemples de matchs mal prédits :")
    print(errors.head(10))
    print("\nRépartition des erreurs par vrai résultat :")
    print(errors['true_result'].value_counts())
    print("\nRépartition des erreurs par prédiction :")
    print(errors['predicted_result'].value_counts())

    # 6. TEST ON NEW MATCHES
    print("\n6. TEST ON NEW MATCHES")
    print("-" * 30)
    new_matches = pd.DataFrame(000
        'home_team': ['Standard de Liège', 'Club Brugge', 'Standard de Liège', 'Anderlecht'],
        'away_team': ['Anderlecht', 'Standard de Liège', 'RFC Seraing', 'Standard de Liège'],
        'home_recent_form': [0.6, 0.4, 0.7, 0.8],
        'away_recent_form': [0.8, 0.6, 0.3, 0.5],
        'h2h_home_wins': [5, 2, 8, 3],
        'h2h_away_wins': [3, 6, 1, 5],
        'home_league_position': [8, 3, 8, 2],
        'away_league_position': [2, 8, 15, 8]
    })
    print("Matches to predict (realistic features only):")
    try:
        X_new = predictor.preprocess_features(new_matches)
        home_pred_new, away_pred_new = predictor.predict_scores(X_new)
        home_pred_rounded = np.round(home_pred_new).astype(int)
        away_pred_rounded = np.round(away_pred_new).astype(int)
        result_pred_new = np.sign(home_pred_rounded - away_pred_rounded)
        result_names = {-1: 'LOSE', 0: 'DRAW', 1: 'WIN'}
        for i, (idx, match) in enumerate(new_matches.iterrows()):
            prediction = result_pred_new[i]
            print(f"   {match['home_team']} vs {match['away_team']} → Standard should {result_names[prediction]}")
    except Exception as e:
        print(f"   Error in prediction: {e}")
        print("   This might indicate issues with feature preprocessing")

    print("\nSIMPLE PREDICTOR TEST COMPLETED SUCCESSFULLY!")
    return {
        'accuracy': accuracy,
        'data_size': len(data),
        'features': list(X_train.columns),
        'model_type': 'SimplePredictorPoisson',
        'predictions': result_pred
    }

if __name__ == "__main__":
    try:
        results = test_simple_predictor()
        print("\nTest completed successfully!")
        print(f"Final accuracy: {results['accuracy']:.1%}")
    except Exception as e:
        print(f"\nError occurred during testing: {e}")
        print("Please check that all files are present and correct.")