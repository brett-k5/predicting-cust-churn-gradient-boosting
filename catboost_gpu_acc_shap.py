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

    cat_cols = features_train.select_dtypes(include='category').columns.tolist()
    for col in cat_cols:
        features_train[col] = features_train[col].astype(str).fillna('missing')
    
    cat_cols = features_train.select_dtypes(include='object').columns.tolist()

    if in_colab():
        grid_search_cat = joblib.load('grid_search_cat.pkl')
    else:
        grid_search_cat = joblib.load('cv_tuning_results/grid_search_cat.pkl')

    model_cat = grid_search_cat.best_estimator_

    accuracy_threshold_shap(features_train, 
                  target_train,  
                  model_cat,
                  'catboost',
                  3,
                  cat_cols)
