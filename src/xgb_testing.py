

# Standard library imports
import os

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
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict

# Local Utilities
from src.model_utils import threshold_calc, print_metrics, shap_eval



if __name__ == '__main__':

    # Make sure we have xgboost version that can handle categorical variables natively
    print(xgboost.__version__)

    features_train = pd.read_parquet('features_train.parquet')
    target_train = pd.read_parquet('target_train.parquet').iloc[:,0]

    features_test = pd.read_parquet('features_test.parquet')
    target_test = pd.read_parquet('target_test.parquet')


    # Load model if the file exists
    model_path = "model_xgb.pkl"
    if os.path.exists(model_path):
        model_xgb = joblib.load(model_path)
        print("Loaded model_xgb from file.")
    else:
        raise FileNotFoundError(f"{model_path} not found. Make sure the model was trained and saved.")


    optimal_threshold = threshold_calc(model_xgb, 
                                    features_train, 
                                    target_train)

    print_metrics(model_xgb, 
                  features_train, 
                  target_train, 
                  features_test, 
                  target_test,
                  optimal_threshold)

    shap_eval(model_xgb, 
              features_test, 
              target_test)


