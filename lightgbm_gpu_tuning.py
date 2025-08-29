# third party libraries
import pandas as pd
import lightgbm as lgb
from lightgbm import LGBMClassifier

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
    # Local Utilities
    from src.model_utils import tuning_cv


if __name__ == '__main__':

    features_train = pd.read_parquet('features_train.parquet')
    target_train = pd.read_parquet('target_train.parquet').iloc[:, 0]

    # We need to make sure we have a version of LGBMClassifier that can run on GPUs
    print("LightGBM version:", lgb.__version__)

    try:
        model_light = lgb.LGBMClassifier(device='gpu', eval_metric='auc', random_state=12345)
        print("GPU support is enabled for LightGBM.")
    except Exception as e:
        print("GPU support is NOT enabled:", e)

    param_grid = {
        'num_leaves': [15, 31],                 
        'max_depth': [5, -1],                   
        'learning_rate': [0.01, 0.1],           
        'n_estimators': [100, 300],             
        'subsample': [0.8, 1.0],                
        'colsample_bytree': [0.8, 1.0],         
        'class_weight': [None, 'balanced']      
    }

    tuning_cv(features_train, target_train, param_grid, 'light')

    
