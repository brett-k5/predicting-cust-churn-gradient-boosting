

# For type hints
from typing import Optional

# Colab utilities
from google.colab import files

# Core third-party libraries
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import joblib

# XGBoost
import xgboost
from xgboost import XGBClassifier

# Scikit-learn: model selection and evaluation and type hints
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score

# Local Utilities
from src.model_utils import tuning_cv, accuracy_calc, shap_eval



if __name__ == '__main__':

    # Make sure we have xgboost version that can handle categorical variables natively
    print(xgboost.__version__)

    features_train = pd.read_parquet('features_train.parquet')
    target_train = pd.read_parquet('target_train.parquet').iloc[:, 0]

    model_xgb = XGBClassifier(
    tree_method="gpu_hist",
    enable_categorical=True,        # enables native handling of categorical features
    predictor="gpu_predictor",      # optional for faster inference
    use_label_encoder=False,
    eval_metric='auc',
    random_state=12345)

    param_grid = {
    'n_estimators': [100, 300],
    'max_depth': [3, 5],
    'learning_rate': [0.05, 0.1],
    'subsample': [0.7, 1.0],
    'colsample_bytree': [0.7, 1.0],
    'scale_pos_weight': [3, 4]}  # imbalance adjustment


    model_xgb = tuning_cv(features_train, 
                                   target_train, 
                                   model_xgb, 
                                   param_grid)

    accuracy_calc(features_train, 
                  target_train,  
                  model_xgb)



    shap_eval(model_xgb, 
              features_train, 
              target_train)

    # Save model to project directory
    joblib.dump(model_xgb, 'model_xgb.pkl')

