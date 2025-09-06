# 🧠 Customer Churn Prediction with Gradient Boosting

This project is an end-to-end machine learning pipeline designed to predict customer churn using powerful gradient boosting algorithms. We leveraged feature engineering, model tuning, cross-validation, and SHAP explainability tools to build a robust and interpretable solution. The final model is trained to identify customers likely to churn with high recall and precision.

---

## 🚀 Project Goals
- Evaluate XGBoost, LightGBM, and CatBoost for best performance.
- Engineer time- and subscription-based features while minimizing data leakage risk.
- Use SHAP analysis for model interpretability.
- Optimize for **ROC-AUC** while preserving strong **recall**.

---

## 🛠️ Feature Engineering

We engineered a comprehensive set of temporal and categorical features, including:

- A custom `p_i_or_b` feature indicating customer service type (`phone`, `internet`, or `both`).
- A refined `customer_duration` variable, adjusted with randomized floats to prevent data leakage tied to uniform contract start dates.
- Multiple datetime-derived features:
  - `start_year`, `start_month`, `start_dayofmonth`, `start_dayofweek`, `start_dayofyear`
  - Integer, categorical, and **cyclical (sin/cos)** encodings of temporal features

To prevent leakage or redundant signal:

- **Cyclical encodings and redundant date features were dropped** after iterative CV testing because the led to data leakage when paired with 'customer_duration' and were not as predictive as 'customer_duration'
- Features were evaluated using SHAP values across CV folds, allowing data-driven pruning of noisy or misleading predictors.

---

## 🧪 Model Selection & Tuning

We evaluated three gradient boosting models:

- [XGBoost](https://xgboost.readthedocs.io/)
- [LightGBM](https://lightgbm.readthedocs.io/)
- [CatBoost](https://catboost.ai/)

### 🔧 Hyperparameter Tuning:
- Performed using `GridSearchCV`, optimizing for **ROC-AUC**.
- A secondary CV loop (outside GridSearchCV) was used for:
  - Computing average **accuracy**, **recall**, and **optimal decision thresholds** across all folds.
  - Generating **SHAP plots** for interpretability and leakage detection.

---

## 📈 Cross-Validation Results

| Model       | ROC-AUC | Accuracy | Recall  |
|-------------|---------|----------|---------|
| CatBoost    | 0.850   | 0.807    | 0.528   |
| LightGBM    | 0.850   | 0.805    | 0.489   |
| **XGBoost** | **0.852** | 0.717  | **0.852** |

> Note: While CatBoost and LightGBM achieved slightly higher accuracy, XGBoost had **substantially better recall**, making it more suitable for identifying churn risk.

---

## 🏁 Final Model Test Results

The final model selected was **XGBoost**, trained on the full dataset and evaluated on a holdout test set.

| Metric      | Score   |
|-------------|---------|
| ROC-AUC     | 0.870   |
| Accuracy    | 0.766   |
| Recall      | 0.807   |

When tested with `ignore_youden_j=True` (i.e., **no Youden’s J decision threshold adjustment**) in custom model_selection() function, results were:

| Metric      | Score   |
|-------------|---------|
| Accuracy    | 0.720   |
| Recall      | **0.890** |

> ⚠️ These alternate results suggest **improved recall**, but should be interpreted cautiously, as they were not validated through rigorous CV.

---

## 📋 Key Custom Function Features

- `model_selection(override_model=None)`  
  Allows manual override of the model selected based on `roc_auc` if another model demonstrates better real-world performance traits (e.g. balance of roc_auc vs. recall).
  
- `print_save_metrics(ignore_youden_j=False)`  
  Enables bypassing the Youden's J statistic when recall is prioritized, which can be critical for churn problems where missing a churn-prone customer is costly.

---

## 🧠 Why We Modified `customer_duration`

Each customer contract began on the 1st of a month. The combination of contract type and unmodified `customer_duration` could allow a model to infer precise start dates — a form of **data leakage**. To avoid this:

- We **subtracted a random float between 12 and 15** days from each `customer_duration`, effectively fuzzing the signal without removing its predictive power.

---

## 📊 SHAP Interpretability

We used SHAP values to:

- Identify which features drove model predictions.
- Detect potential data leakage.
- Inform feature selection (drop vs. keep decisions).

---

## ✅ Summary

- 📌 **XGBoost** provided the best trade-off between ROC-AUC and recall.
- 📌 Feature engineering carefully balanced **predictive power** and **leakage prevention**.
- 📌 The project is fully end-to-end, from preprocessing and EDA to interpretability and final test results.
- 📌 Designed to be modular, interpretable, and production-adaptable.

---
## 📁 Project Structure Highlights  
```
project-root/
│
├── README.md
├── LICENSE
├── Requirements.txt
│
├── contract.csv
├── internet.csv
├── personal.csv
├── phone.csv
│
├── features_train.parquet
├── features_test.parquet
├── target_train.parquet
├── target_test.parquet
│
├── model_testing.py
├── catboost_gpu_tuning.py
├── lightgbm_gpu_tuning.py
├── xgb_gpu_tuning.py
├── catboost_gpu_acc_shap.py
├── lightgbm_gpu_acc_shap.py
├── xgboost_gpu_acc_shap.py
│
├── notebooks/
│ ├── EDA.ipynb
│ ├── pre_processing.ipynb
│ ├── results_and_analysis.ipynb
│ ├── model_test.ipynb
│ ├── catboost_gpu_acc_shap.ipynb
│ ├── lightgbm_gpu_acc_shap.ipynb
│ ├── xgboost_gpu_acc_shap.ipynb
│ └── catboost_cv_tuning_colab.ipynb
│
├── src/
│ └── model_utils.py
│
├── cv_tuning_results/
│ ├── grid_search_cat.joblib
│ ├── grid_search_light.joblib
│ ├── grid_search_xgb.joblib
│ ├── catboost_shap_summary_plot_fold*.png
│ ├── lightgbm_shap_summary_plot_fold*.png
│ ├── xgboost_shap_summary_plot_fold*.png
│ └── _avg_shap_fold.csv
│
└── test_results/
├── test_scores.csv
├── xgb_test_shap.csv
└── xgb_test_shap_plot.png
```

---

## 🧠 Authors

- Developed by Brett Kunkel | [www.linkedin.com/in/brett-kunkel](www.linkedin.com/in/brett-kunkel) | brttkunkel@gmail.com

---

## 📜 License

This project is licensed under the MIT License.