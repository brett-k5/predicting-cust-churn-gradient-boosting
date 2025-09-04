# Core third-party libraries
import pandas as pd
import joblib 

# Local application imports
def in_colab():
    try:
        import google.colab
        # Colab utilities
        from google.colab import files
        return True
    except ImportError:
        return False
    
if in_colab():
    from model_utils import accuracy_threshold_shap
else:
    from src.model_utils import accuracy_threshold_shap

if __name__ == '__main__':
    
    features_train = pd.read_parquet('features_train.parquet')
    target_train = pd.read_parquet('target_train.parquet')

    if in_colab():
        grid_search_xgb = joblib.load('grid_search_xgb.joblib')
    else:
        grid_search_xgb = joblib.load('cv_tuning_results/grid_search_xgb.joblib')

    model_xgb = grid_search_xgb.best_estimator_

    accuracy_threshold_shap(features_train, 
                            target_train,  
                            model_xgb,
                            'xgboost',
                            3)