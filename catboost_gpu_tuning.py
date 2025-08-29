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

    cat_cols = features_train.select_dtypes(include='category').columns.tolist()
    for col in cat_cols:
        features_train[col] = features_train[col].astype(str).fillna('missing')
    
    cat_cols = features_train.select_dtypes(include='object').columns.tolist()
    
    param_grid = {
    'iterations': [100, 300],               # Keep it light but effective
    'depth': [4, 6],                        # Balanced tree depth options
    'learning_rate': [0.05, 0.1],           # Two sensible learning rates
    'l2_leaf_reg': [3, 7],                  # Light regularization options
    'border_count': [32, 50],               # Numerical feature binning splits
    'random_strength': [1, 2],              # Control randomness in splitting
    'bagging_temperature': [0, 1],          # Simple bagging temps to test
    'scale_pos_weight': [1, 3]}             # Basic class imbalance adjustment

    tuning_cv(features_train, target_train, 
              param_grid, 'cat',
              cat_features=cat_cols)
    

