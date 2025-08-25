
# ============================
# COLAB SETUP (Run only if in Colab)
# ============================
import sys
import os
from pathlib import Path

try:
    import google.colab
    from google.colab import drive
    drive.mount('/content/drive')

    # Adjust this path if needed
    project_path = Path('/content/drive/MyDrive/predicting_cust_churn')
    sys.path.append(str(project_path))
    os.chdir(project_path)

except ImportError:
    pass  # Not running in Colab




# For type hints
from typing import Optional

# Colab utilities
from google.colab import files

# Core third-party libraries
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

# LightGBM
import lightgbm as lgb
from lightgbm import LGBMClassifier

# Scikit-learn: model selection and evaluation
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score

# Local Utilities
from src.model_utils import tuning_cv, accuracy_calc, shap_eval



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

    model_light = tuning_cv(features_train, 
                            target_train, 
                            model_light, 
                            param_grid)

    accuracy_calc(features_train, 
                  target_train,  
                  model_light)

    shap_eval(model_light, 
              features_train, 
              target_train)

