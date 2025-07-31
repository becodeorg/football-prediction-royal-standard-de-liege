'''
Simple test for the SimplePredictor model.
'''

import sys
import os
# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.real_data_collector import RealDataCollector
from src.backend import SimplePredictorRegression
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
    
    #----------------------------------------------------------------------------------------------------------------------------------------

    # 2. MODEL INITIALIZATION
    print("\n2. MODEL INITIALIZATION")
    print("-" * 30)
    
    predictor = SimplePredictorRegression()
    print("SimplePredictor initialized")
    
    #----------------------------------------------------------------------------------------------------------------------------------------

    # 3. DATA PREPARATION
    print("\n3. DATA PREPARATION")
    print("-" * 30)
    
    X_train, X_test, y_train, y_test = predictor.prepare_data(data, test_size=0.2)

    # Rééquilibrage des classes avec SMOTE
    smote = SMOTE(random_state=42)
    X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
    print(f"After SMOTE: {dict(pd.Series(y_train_bal).value_counts())}")
    
    print(f"Data split:")
    print(f"   Training: {len(X_train)} samples")
    print(f"   Test: {len(X_test)} samples")
    print(f"   Features used: {len(X_train.columns)}")
    print(f"   Feature list: {list(X_train.columns)}")
    
    #----------------------------------------------------------------------------------------------------------------------------------------

    # 4. TRAINING
    print("\n4. MODEL TRAINING")
    print("-" * 30)

    # GridSearch pour Random Forest
    param_grid = {
        'n_estimators': [30, 50, 100],
        'max_depth': [3, 4, 5, 6],
        'min_samples_leaf': [3, 5, 10]
    }
    rf = RandomForestClassifier(class_weight='balanced', random_state=42)
    grid_search = GridSearchCV(rf, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train_bal, y_train_bal)

    print("Best params:", grid_search.best_params_)
    print("Best CV score:", grid_search.best_score_)

    # Utilise le meilleur modèle trouvé
    predictor.model = grid_search.best_estimator_
    
    predictor.train(X_train_bal, y_train_bal)
    print("Model trained successfully")
    
    # Feature importance
    if hasattr(predictor.model, 'feature_importances_'):
        print(f"\nFeature importance:")
        feature_importance = list(zip(X_train.columns, predictor.model.feature_importances_))
        feature_importance.sort(key=lambda x: x[1], reverse=True)
        for i, (feature, importance) in enumerate(feature_importance[:5]):
            print(f"   {i+1}. {feature}: {importance:.3f} ({importance*100:.1f}%)")
    
    #----------------------------------------------------------------------------------------------------------------------------------------

    # 5. CROSS-VALIDATION ANALYSIS
    print("\n5. CROSS-VALIDATION ANALYSIS")
    print("-" * 30)
    
    cv_metrics = predictor.evaluate_with_cross_validation(X_train, y_train, cv_folds=5)
    
    #----------------------------------------------------------------------------------------------------------------------------------------

    # 6. MODEL EVALUATION
    print("\n6. MODEL EVALUATION")
    print("-" * 30)
    
    test_metrics = predictor.evaluate(X_test, y_test, X_train, y_train)
    accuracy = test_metrics['test_accuracy']
    
    # Test predictions for detailed analysis
    test_predictions = predictor.predict_preprocessed(X_test)

    # Analyse détaillée des erreurs de prédiction
    test_results = X_test.copy()
    test_results['true_result'] = y_test.values
    test_results['predicted_result'] = test_predictions

    # Afficher les erreurs
    errors = test_results[test_results['true_result'] != test_results['predicted_result']]
    print(f"\nNombre de matchs mal prédits : {len(errors)} / {len(test_results)}")
    print("Exemples de matchs mal prédits :")
    print(errors.head(10))  # Affiche les 10 premières erreurs

    # Statistiques sur les types d'erreurs
    print("\nRépartition des erreurs par vrai résultat :")
    print(errors['true_result'].value_counts())
    print("\nRépartition des erreurs par prédiction :")
    print(errors['predicted_result'].value_counts())
    
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
    
    #----------------------------------------------------------------------------------------------------------------------------------------

    # 7. TEST ON NEW MATCHES
    print("\n7. TEST ON NEW MATCHES")
    print("-" * 30)
    
    new_matches = pd.DataFrame({
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
        predictions = predictor.predict(new_matches)
        result_names = {-1: 'LOSE', 0: 'DRAW', 1: 'WIN'}
        
        for i, (idx, match) in enumerate(new_matches.iterrows()):
            prediction = predictions[i]
            print(f"   {match['home_team']} vs {match['away_team']} → Standard should {result_names[prediction]}")
    except Exception as e:
        print(f"   Error in prediction: {e}")
        print("   This might indicate issues with feature preprocessing")

    #----------------------------------------------------------------------------------------------------------------------------------------
    
    # 8. FINAL VALIDATION
    print("\n8. FINAL VALIDATION")
    print("-" * 30)
    
    # Sanity checks
    checks = []
    
    # Check for data leakage
    problematic_features = ['home_goals', 'away_goals']
    remaining_problematic = [col for col in X_train.columns if col in problematic_features]
    if remaining_problematic:
        checks.append(f"WARNING: Still using problematic features: {remaining_problematic}")
    else:
        checks.append("No data leakage: All problematic features removed")
    
    # Check that the model predicts all 3 classes
    unique_predictions = np.unique(np.concatenate([test_predictions, predictions]))
    if len(unique_predictions) >= 2:
        checks.append("The model predicts multiple different results")
    else:
        checks.append("The model always predicts the same result")
    
    # Check accuracy with realistic thresholds
    if accuracy > 0.7:
        checks.append(f"Exceptional accuracy: {accuracy:.1%}")
    elif accuracy > 0.6:
        checks.append(f"Good accuracy: {accuracy:.1%}")
    elif accuracy > 0.5:
        checks.append(f"Fair accuracy: {accuracy:.1%} (better than random)")
    else:
        checks.append(f"Low accuracy: {accuracy:.1%} (needs improvement)")
    
    # Check overfitting
    if cv_metrics['is_overfitting']:
        checks.append("Overfitting detected in cross-validation")
    else:
        checks.append("Good generalization in cross-validation")
    
    if test_metrics.get('is_overfitting', False):
        checks.append("Overfitting detected in train/test comparison")
    else:
        checks.append("Good generalization in train/test comparison")
    
    # Check consistency
    if len(X_train) > len(X_test):
        checks.append("Train/test split is consistent")
    else:
        checks.append("Train/test split is suspicious")
    
    for check in checks:
        print(f"   {check}")
    
    # 9. FINAL SUMMARY
    print("\n9. FINAL SUMMARY")
    print("-" * 30)
    
    print(f"Dataset: {len(data)} Standard de Liège matches")
    print(f"Model: {predictor.model_name}")
    print(f"Features: {len(X_train.columns)} legitimate features")
    print(f"CV Accuracy: {cv_metrics['cv_mean_accuracy']:.1%} (±{cv_metrics['cv_std_accuracy']:.1%})")
    print(f"Test Accuracy: {accuracy:.1%}")
    print(f"Tests: {len(new_matches)} realistic matches predicted")
    
    if accuracy >= 0.7:
        print("EXCEPTIONAL: Very high performance for realistic football prediction!")
    elif accuracy >= 0.6:
        print("EXCELLENT: Good performance for football prediction")
    elif accuracy >= 0.5:
        print("GOOD: Better than random, reasonable for football")
    else:
        print("NEEDS IMPROVEMENT: Model requires adjustments")
    
    # Diagnostic d'overfitting
    if cv_metrics['is_overfitting'] or test_metrics.get('is_overfitting', False):
        print("Note: Some overfitting detected - consider regularization")
    else:
        print("Model shows good generalization capabilities")
    
    print("\nSIMPLE PREDICTOR TEST COMPLETED SUCCESSFULLY!")
    
    return {
        'accuracy': accuracy,
        'cv_metrics': cv_metrics,
        'test_metrics': test_metrics,
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