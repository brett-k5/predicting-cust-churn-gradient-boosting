# Core third-party libraries
import pandas as pd
from catboost import CatBoostClassifier
import lightgbm as lgb
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

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
    from model_utils import load_threshold, print_save_metrics, shap_eval, load_grid_search, model_selection
else:
    # Local Utilities
    from src.model_utils import load_threshold, print_save_metrics, shap_eval, load_grid_search, model_selection



if __name__ == '__main__':

    features_train = pd.read_parquet('features_train.parquet')
    target_train = pd.read_parquet('target_train.parquet')

    features_test = pd.read_parquet('features_test.parquet')
    target_test = pd.read_parquet('target_test.parquet')

    # We need to make sure we have a version of LGBMClassifier that can run on GPUs
    print("LightGBM version:", lgb.__version__)

    try:
        model_light = lgb.LGBMClassifier(device='gpu', eval_metric='auc', random_state=12345)
        print("GPU support is enabled for LightGBM.")
    except Exception as e:
        print("GPU support is NOT enabled:", e)

    model_light = load_grid_search('grid_search_light.joblib').best_estimator_
    model_cat = load_grid_search('grid_search_cat.joblib').best_estimator_
    model_xgb = load_grid_search('grid_search_xgb.joblib').best_estimator_

    model_light_roc_auc = load_grid_search('grid_search_light.joblib').best_score_
    model_cat_roc_auc = load_grid_search('grid_search_cat.joblib').best_score_
    model_xgb_roc_auc = load_grid_search('grid_search_xgb.joblib').best_score_

    best_model, model_type = model_selection(model_cat,
                                             model_light,
                                             model_xgb,
                                             model_cat_roc_auc,
                                             model_light_roc_auc,
                                             model_xgb_roc_auc,
                                             override_model = model_light)
    

    print(f"Best Model Type: {type(best_model).__name__}")

    optimal_threshold = load_threshold(best_model)

    print_save_metrics(best_model, 
                       features_train, 
                       target_train, 
                       features_test, 
                       target_test,
                       optimal_threshold)


    shap_eval(best_model, 
              model_type, 
              features_test,
              None,
              False)


