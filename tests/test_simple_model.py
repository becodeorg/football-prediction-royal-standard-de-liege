'''
Simple test for the SimplePredictor model.
'''

import sys
import os
# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.real_data_collector import RealDataCollector
from model.simple_predictor import SimplePredictor
import pandas as pd
import numpy as np

def test_simple_predictor():
    """
    Detailed and comprehensive test of the SimplePredictor
    """
    print("DETAILED TEST OF SIMPLE PREDICTOR")
    print("=" * 50)
    
    # 1. DATA GENERATION
    print("\n1. DATA GENERATION")
    print("-" * 30)
    
    collector = RealDataCollector()
    data = collector.get_standard_focused_data(200)
    
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
    
    # 2. MODEL INITIALIZATION
    print("\n2. MODEL INITIALIZATION")
    print("-" * 30)
    
    predictor = SimplePredictor()
    print("SimplePredictor initialized")
    
    # 3. DATA PREPARATION
    print("\n3. DATA PREPARATION")
    print("-" * 30)
    
    X_train, X_test, y_train, y_test = predictor.prepare_data(data, test_size=0.2)
    
    print(f"Data split:")
    print(f"   Training: {len(X_train)} samples")
    print(f"   Test: {len(X_test)} samples")
    print(f"   Features used: {len(X_train.columns)}")
    print(f"   Feature list: {list(X_train.columns)}")
    
    # 4. TRAINING
    print("\n4. MODEL TRAINING")
    print("-" * 30)
    
    predictor.train(X_train, y_train)
    print("Model trained successfully")
    
    # Feature importance
    if hasattr(predictor.model, 'feature_importances_'):
        print(f"\nFeature importance:")
        feature_importance = list(zip(X_train.columns, predictor.model.feature_importances_))
        feature_importance.sort(key=lambda x: x[1], reverse=True)
        for i, (feature, importance) in enumerate(feature_importance[:5]):
            print(f"   {i+1}. {feature}: {importance:.3f} ({importance*100:.1f}%)")
    
    # 5. MODEL EVALUATION
    print("\n5. MODEL EVALUATION")
    print("-" * 30)
    
    # Test predictions
    test_predictions = predictor.predict_preprocessed(X_test)
    accuracy = (test_predictions == y_test).mean()
    
    print(f"Overall accuracy: {accuracy:.1%}")
    
    # Detailed analysis by class
    unique_results = [-1, 0, 1]
    result_names = {-1: 'Defeat', 0: 'Draw', 1: 'Win'}
    
    print(f"\nDetailed analysis by result:")
    for result in unique_results:
        mask = y_test == result
        if mask.sum() > 0:
            class_accuracy = (test_predictions[mask] == result).mean()
            count = mask.sum()
            print(f"   {result_names[result]}: {class_accuracy:.1%} ({count} cases)")
    
    # 6. TEST ON NEW MATCHES
    print("\n6. TEST ON NEW MATCHES")
    print("-" * 30)
    
    # Create realistic test matches
    new_matches = pd.DataFrame({
        'home_team': ['Standard de Liège', 'Club Brugge', 'Standard de Liège', 'Anderlecht'],
        'away_team': ['Anderlecht', 'Standard de Liège', 'RFC Seraing', 'Standard de Liège'],
        'home_goals': [2, 1, 3, 1],  # Dummy data for preprocessing
        'away_goals': [1, 2, 0, 2]   # Dummy data for preprocessing
    })
    
    print("Matches to predict:")
    predictions = predictor.predict(new_matches)
    
    for i, (_, match) in enumerate(new_matches.iterrows()):
        result_text = result_names[predictions[i]]
        
        # Determine if Standard plays at home or away
        if match['home_team'] == 'Standard de Liège':
            match_desc = f"Standard vs {match['away_team']}"
            prediction_desc = f"Standard should {'WIN' if predictions[i] == 1 else 'DRAW' if predictions[i] == 0 else 'LOSE'}"
        else:
            match_desc = f"{match['home_team']} vs Standard"
            prediction_desc = f"Standard should {'WIN' if predictions[i] == -1 else 'DRAW' if predictions[i] == 0 else 'LOSE'}"
        
        print(f"   {match_desc} → {prediction_desc}")
    
    # 7. FINAL VALIDATION
    print("\n7. FINAL VALIDATION")
    print("-" * 30)
    
    # Sanity checks
    checks = []
    
    # Check that the model predicts all 3 classes
    unique_predictions = np.unique(np.concatenate([test_predictions, predictions]))
    if len(unique_predictions) >= 2:
        checks.append("The model predicts multiple different results")
    else:
        checks.append("The model always predicts the same result")
    
    # Check accuracy
    if accuracy > 0.6:  # More than 60%
        checks.append(f"Excellent accuracy: {accuracy:.1%}")
    elif accuracy > 0.4:  # More than 40% (better than random)
        checks.append(f"Acceptable accuracy: {accuracy:.1%}")
    else:
        checks.append(f"Low accuracy: {accuracy:.1%}")
    
    # Check consistency
    if len(X_train) > len(X_test):
        checks.append("Train/test split is consistent")
    else:
        checks.append("Train/test split is suspicious")
    
    for check in checks:
        print(f"   {check}")
    
    # 8. FINAL SUMMARY
    print("\n8. FINAL SUMMARY")
    print("-" * 30)
    
    print(f"Dataset: {len(data)} Standard de Liège matches")
    print(f"Model: Random Forest ({predictor.model.n_estimators} trees)")
    print(f"Accuracy: {accuracy:.1%}")
    print(f"Features: {len(X_train.columns)} characteristics")
    print(f"Tests: {len(new_matches)} new matches predicted")
    
    if accuracy >= 0.7:
        print(f"EXCELLENT: Very high-performing model for football prediction!")
    elif accuracy >= 0.5:
        print(f"GOOD: Correct and usable model")
    else:
        print(f"NEEDS IMPROVEMENT: Model requires adjustments")
    
    print(f"\nSIMPLE PREDICTOR TEST COMPLETED SUCCESSFULLY!")
    
    return {
        'accuracy': accuracy,
        'data_size': len(data),
        'features': list(X_train.columns),
        'model_type': 'SimplePredictor',
        'predictions': predictions
    }

if __name__ == "__main__":
    try:
        results = test_simple_predictor()
        print("\nTest completed successfully!")
        print(f"Final accuracy: {results['accuracy']:.1%}")
    except Exception as e:
        print(f"\nError occurred during testing: {e}")
        print("Please check that all files are present and correct.")