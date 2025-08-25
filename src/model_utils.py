
from typing import Optional
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt

from sklearn.base import BaseEstimator
from sklearn.model_selection import GridSearchCV, cross_val_score, cross_val_predict
from sklearn.metrics import roc_auc_score, accuracy_score, recall_score, roc_curve

# Define a function that returns best hyperparameters (roc_auc) during cross validation
def tuning_cv(features_train: pd.DataFrame, 
              target_train: pd.Series, 
              model: BaseEstimator, 
              param_grid: dict[str, list], 
              scoring: str = 'roc_auc',
              cv: int = 3,
              cat_features: Optional[list[str]] = None) -> BaseEstimator:
    """
    Returns the model with the hyperparameters that performed the best, prints the hyperparameters,
    and prints the best accuracy score (default accuracy metric is roc_auc).
    """
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=cv,
        scoring=scoring,
        verbose=1)

    if cat_features:
        grid_search.fit(features_train, target_train, cat_features=cat_features)
    else:
        grid_search.fit(features_train, target_train)

    print("Best parameters found: ", grid_search.best_params_)
    print("Best cross-validation score: ", grid_search.best_score_)

    return grid_search.best_estimator_

# Define a function that calculates the accuracy score for the given model 
def accuracy_calc(features_train: pd.DataFrame, 
                  target_train: pd.Series, 
                  returned_estimator: BaseEstimator, 
                  cv: int = 3, 
                  scoring: str = 'accuracy') -> None:
    """
    Calculates and prints the accuracy score.
    """
    scores = cross_val_score(
        returned_estimator,
        features_train,
        target_train,
        cv=cv,
        scoring=scoring)

    print("Mean cross validated accuracy of best estimator during cross validation:", scores.mean())

# Define a function that performs a SHAP analysis for the given model
def shap_eval(returned_estimator: BaseEstimator, 
              features: pd.DataFrame, 
              target: pd.Series) -> None:
    """
    Calculates and prints SHAP values for each feature and prints SHAP plot.
    """
    returned_estimator.fit(features, target)
    explainer = shap.TreeExplainer(returned_estimator)
    shap_values = explainer.shap_values(features)

    feature_names = list(features.columns)
    mean_abs_impacts = np.mean(np.abs(shap_values), axis=0)
    for name, val in zip(feature_names, mean_abs_impacts):
        print(f"{name}: {val}")
    shap.summary_plot(shap_values, features)
    plt.show()

# Define a function to calculate the best decision threshold for the given model 
def threshold_calc(returned_estimator: BaseEstimator, 
                   features_train: pd.DataFrame, 
                   target_train: pd.Series, 
                   method: str = 'predict_proba', 
                   cv: int = 5) -> float: 
    """
    Calculate the optimal threshold for predictions. Utilizes 5 folds during cross validation 
    instead of 3 as we used during hyperparameter tuning to minimize overfitting as much as 
    possible. Prints and returns optimal threshold value. 
    """
    probs = cross_val_predict(
        returned_estimator,
        features_train,
        target_train,
        method=method,
        cv=cv)[:, 1]

    fpr, tpr, thresholds = roc_curve(target_train, probs)
    youden_j = tpr - fpr
    optimal_idx = np.argmax(youden_j)
    optimal_threshold = thresholds[optimal_idx]
    print(f"Optimal threshold: {optimal_threshold:.4f}")
    return optimal_threshold

# Define a function that prints metrics for model performance on test set
def print_metrics(returned_estimator: BaseEstimator, 
                  features_train: pd.DataFrame, 
                  target_train: pd.Series, 
                  features_test: pd.DataFrame, 
                  target_test: pd.Series,
                  optimal_threshold: float) -> None:
    """
    1. Trains model on full training set
    2. Then makes predictions on test set
    3. Calculates and prints roc_auc, accuracy, and recall scores for the model's predictions
    """
    returned_estimator.fit(features_train, target_train)
    pred_prob = returned_estimator.predict_proba(features_test)[:, 1]

    roc_auc = roc_auc_score(target_test, pred_prob)
    print(f"Roc_Auc score for model: {roc_auc:.4f}")

    preds = pred_prob > optimal_threshold
    accuracy = accuracy_score(target_test, preds)
    print(f"Accuracy score for model: {accuracy:.4f}")

    recall = recall_score(target_test, preds)
    print(f"Recall score for model: {recall:.4f}")
