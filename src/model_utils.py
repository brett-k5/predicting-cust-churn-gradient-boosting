# Standard library
import os
from typing import Optional

# Third-party libraries
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from catboost import CatBoostClassifier
import lightgbm as lgb
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.base import BaseEstimator, clone
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score, roc_curve
from sklearn.model_selection import (GridSearchCV, StratifiedKFold,
                                     cross_val_predict, cross_val_score)

# Local application imports
def in_colab():
    try:
        import google.colab
        # Colab utilities
        from google.colab import files
        return True
    except ImportError:
        return False

# Define a function that selects best hyperparameters (roc_auc) during cross validation
def tuning_cv(features_train: pd.DataFrame, 
              target_train: pd.Series, 
              param_grid: dict[str, list],
              model_type: str, 
              scoring: str = 'roc_auc',
              cv: int = 3,
              cat_features: Optional[list[str]] = None) -> None:
    """
    Selects the hyperparameters that performed the best with the model, prints the hyperparameters,
    prints the best cross validation score (default accuracy metric is roc_auc), 
    and saves the grid search object
    """
    
    if model_type == 'cat':
        model = CatBoostClassifier(random_seed=12345, eval_metric='AUC', task_type='GPU')
    elif model_type == 'light':
        try:
            model = LGBMClassifier(random_state=12345, num_leaves=31, n_estimators=200, device='gpu')
        except Exception as e:
            print("GPU support is NOT enabled:", e)
    elif model_type == 'xgb':
        model = XGBClassifier(tree_method="gpu_hist", enable_categorical=True,        # enables native handling of categorical features
                                          predictor="gpu_predictor",  # optional for faster inference
                                          use_label_encoder=False, eval_metric='auc', random_state=12345)
    else:
        raise TypeError(f"Invalid type: {model_type}, function expects 'cat' (catboost), "
                        f"'light' (lightgbm), or 'xgb' (xgboost) instead.")

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=cv,
        scoring=scoring,
        verbose=1)

    if cat_features is not None:
        grid_search.fit(features_train, target_train, cat_features=cat_features)
    else:
        grid_search.fit(features_train, target_train)

    print("Best parameters found: ", grid_search.best_params_)
    print("Best cross-validation score: ", grid_search.best_score_)

    os.makedirs("cv_tuning_results", exist_ok=True)
    grid_search_path = f'cv_tuning_results/grid_search_{model_type}.joblib'
    joblib.dump(grid_search, grid_search_path)
    if in_colab():
        from google.colab import files
        files.download(grid_search_path)



# Define a function that performs a SHAP analysis for the given model
def shap_eval(returned_estimator: BaseEstimator,
              model_type: str, 
              features: pd.DataFrame, 
              fold: int = None,
              cv: bool = True) -> None:
    """
    Calculates and prints SHAP values for each feature and prints SHAP plot.
    """

    # Calculate shap values
    explainer = shap.TreeExplainer(returned_estimator)
    shap_values = explainer.shap_values(features)

    # Calculate mean shap values per feature
    feature_names = list(features.columns)
    mean_abs_impacts = np.mean(np.abs(shap_values), axis=0)

    # Save mean shap values to csv
    shap_values_df = pd.DataFrame([mean_abs_impacts], columns=feature_names)
    if cv:
        # Create cv_tuning_results directory if it doesn't already exist
        os.makedirs("cv_tuning_results", exist_ok=True)
        shap_values_df_path = f'cv_tuning_results/{model_type}_avg_shap_fold{fold}.csv'
    else:
        # Create test_results directory if it doesn't already exist
        os.makedirs("test_results", exist_ok=True)
        shap_values_df_path = f'test_results/{model_type}_test_shap.csv'
    shap_values_df.to_csv(shap_values_df_path, index=False)

    # Print mean shap values per feature
    for name, val in zip(feature_names, mean_abs_impacts):
        print(f"{name}: {val}")

    # Generate and save SHAP summary plot
    plt.figure()
    shap.summary_plot(shap_values, features, show=False)
    if cv:
        plot_path = f'cv_tuning_results/{model_type}_shap_summary_plot_fold{fold}.png'
    else:
        plot_path = f'test_results/{model_type}_test_shap_plot.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.show
    plt.close()

    if in_colab():
        from google.colab import files
        files.download(shap_values_df_path)
        files.download(plot_path)


def accuracy_threshold_shap(features_train: pd.DataFrame, 
                  target_train: pd.Series, 
                  returned_estimator: BaseEstimator,
                  model_type: str,
                  cv: int = 3, 
                  cat_features: list[str] = None) -> None:
    """
    Calculates and prints the accuracy score and SHAP analysis using manual cross-validation.
    """

    # Create cv_tuning_results if it doesn't already exist
    os.makedirs("cv_tuning_results", exist_ok=True)

    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=12345)
    acc_scores = []
    recall_scores = []
    probs = []
    y_val_all = []
    for i, (train_idx, val_idx) in enumerate(skf.split(features_train, target_train)):
        X_train, X_val = features_train.iloc[train_idx], features_train.iloc[val_idx]
        y_train, y_val = target_train.iloc[train_idx], target_train.iloc[val_idx]
        
        model = clone(returned_estimator)
        if cat_features:
            model.fit(X_train, y_train, cat_features=cat_features)
        else:
            model.fit(X_train, y_train)

        preds = model.predict(X_val)
        pred_prob = model.predict_proba(X_val)[:, 1]
        probs.extend(pred_prob)

        acc_score = accuracy_score(y_val, preds)
        acc_scores.append(acc_score)

        rec_score = recall_score(y_val, preds)
        recall_scores.append(rec_score)

        y_val_all.extend(np.array(y_val))

        shap_eval(model, model_type, X_val, i)
    
    # Calculate Youden's J
    fpr, tpr, thresholds = roc_curve(y_val_all, probs)
    youden_j = tpr - fpr
    optimal_idx = np.argmax(youden_j)
    optimal_threshold = thresholds[optimal_idx]
    print(f"Optimal threshold: {optimal_threshold:.4f}")

    # Calculate mean accuracy and recall scores
    mean_accuracy = np.mean(acc_scores)
    mean_recall = np.mean(recall_scores)
    
    # Save the mean accuracy score and threshold
    accuracy_path = f"cv_tuning_results/{model_type}_mean_accuracy.joblib"
    recall_path = f"cv_tuning_results/{model_type}_mean_recall.joblib"
    threshold_path = f"cv_tuning_results/{model_type}_optimal_threshold.joblib"
    joblib.dump(mean_accuracy, accuracy_path)
    joblib.dump(mean_recall, recall_path)
    joblib.dump(optimal_threshold, threshold_path)
    if in_colab():
        from google.colab import files
        files.download(accuracy_path)
        files.download(recall_path)
        files.download(threshold_path)
    else:
        print(f"Saved mean accuracy to: {accuracy_path}")
        print(f"saved mean recall to: {recall_path}")
        print(f"saved optimal decision threshold to {threshold_path}")
    print(f"Mean cross-validation accuracy score of best estimator: {mean_accuracy:.4f}")
    print(f"Mean cross-validation recall score of best estimator: {mean_recall:.4f}")



# Define a function that prints metrics for model performance on test set
def print_save_metrics(returned_estimator: BaseEstimator, 
                  features_train: pd.DataFrame, 
                  target_train: pd.Series, 
                  features_test: pd.DataFrame, 
                  target_test: pd.Series,
                  optimal_threshold: float,
                  ignore_youden_j: bool = False) -> None:
    """
    1. Trains model on full training set
    2. Then makes predictions on test set
    3. Calculates and prints roc_auc, accuracy, and recall scores for the model's predictions
    """
    if isinstance(returned_estimator, CatBoostClassifier):
        cat_cols = features_train.select_dtypes(include='category').columns.tolist()
        for col in cat_cols:
            features_train[col] = features_train[col].astype(str).fillna('missing')
            features_test[col] = features_test[col].astype(str).fillna('missing')
        cat_cols = features_train.select_dtypes(include='object').columns.tolist()
        returned_estimator.fit(features_train, target_train, cat_features=cat_cols)

    else:
        returned_estimator.fit(features_train, target_train)
    pred_prob = returned_estimator.predict_proba(features_test)[:, 1]
    preds = returned_estimator.predict(features_test)
    preds_opt = pred_prob > optimal_threshold

    roc_auc = roc_auc_score(target_test, pred_prob)
    print(f"Roc_Auc score for model: {roc_auc:.4f}")
    
    if ignore_youden_j:
        accuracy = accuracy_score(target_test, preds)
        recall = recall_score(target_test, preds)
    else: 
        accuracy = accuracy_score(target_test, preds_opt)
        recall = recall_score(target_test, preds_opt)
    print(f"Accuracy score for model: {accuracy:.4f}")
    print(f"Recall score for model: {recall:.4f}\n")

    scores_df = pd.DataFrame([{'roc_auc': roc_auc, 
                               'accuracy_score': accuracy, 
                               'recall_score': recall}])
    

    # Create test_results directory if it doesn't already exist
    os.makedirs("test_results", exist_ok=True)
    scores_df.to_csv('test_results/test_scores.csv')
    if in_colab():
        from google.colab import files
        files.download('test_results/test_scores.csv')



def load_grid_search(grid_search_path):
    if os.path.exists(grid_search_path):
        grid_search = joblib.load(grid_search_path)
        return grid_search
    else:
        raise FileNotFoundError("Check to make sure grid_search_path passed to function matches actual grid_Search_path")



def string_selection(model):
        if isinstance(model, CatBoostClassifier):
            model_type = 'cat'
        elif isinstance(model, LGBMClassifier):
            model_type = 'light'
        elif isinstance(model, XGBClassifier):
            model_type = 'xgb'
        else:
            print("Model is of an unexpexted type")
        return model_type



def model_selection(model_cat,
                    model_light,
                    model_xgb,
                    model_cat_roc_auc,
                    model_light_roc_auc,
                    model_xgb_roc_auc,
                    override_model = None):
    if override_model:
        return override_model, string_selection(override_model)
    else:
        best_roc_auc = max(model_cat_roc_auc, model_light_roc_auc, model_xgb_roc_auc)
        if best_roc_auc == model_cat_roc_auc:
            return model_cat, 'cat' 
        elif best_roc_auc == model_light_roc_auc:
            return model_light, 'light'
        else:
            return model_xgb, 'xgb'
    
def load_threshold(best_model: BaseEstimator) -> float:
    if isinstance(best_model, CatBoostClassifier):
        filename = 'catboost_optimal_threshold.joblib'
    elif isinstance(best_model, LGBMClassifier):
        filename = 'lightgbm_optimal_threshold.joblib'
    elif isinstance(best_model, XGBClassifier):
        filename = 'xgboost_optimal_threshold.joblib'
    else:
        raise TypeError("Function expects an instance of either CatBoostClassifier, "
                        "LGBMClassifier, or XGBClassifier as the best_model input.")
    if in_colab():
        optimal_threshold = joblib.load(filename)
    else:
        optimal_threshold = joblib.load(f'cv_tuning_results/{filename}')
    return optimal_threshold

        

    



