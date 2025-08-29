# third-party libraries
import pandas as pd

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
    from model_utils import tuning_cv
else:
    from src.model_utils import tuning_cv



if __name__ == '__main__':

    features_train = pd.read_parquet('features_train.parquet')
    target_train = pd.read_parquet('target_train.parquet')

    param_grid = {
    'n_estimators': [100, 300],
    'max_depth': [3, 5],
    'learning_rate': [0.05, 0.1],
    'subsample': [0.7, 1.0],
    'colsample_bytree': [0.7, 1.0],
    'scale_pos_weight': [3, 4]}  # imbalance adjustment

    tuning_cv(features_train, target_train, 
              param_grid, 'xgb')
