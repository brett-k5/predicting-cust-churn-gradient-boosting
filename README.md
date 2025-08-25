# predicting-customer-churn-gradient-boosting
This project uses customer data from an insurance company to predict churn using advanced feature engineering and gradient boosting models. The goal was to develop a model that could effectively identify customers likely to leave, allowing the business to take proactive retention measures.

🔍 **Project Overview**  
The raw data was provided across four separate datasets:

Customer personal info

Phone service details

Internet service details

Contract information

These were merged on a shared customer_id column to create a unified dataset for modeling.

🧠 **Feature Engineering**  
Several new features were engineered to enhance model performance:

p_i_or_b: Categorized customers as Phone-only, Internet-only, or Both.

customer_duration: A feature representing how long the customer had been with the company at the time of the snapshot.

Cyclical time features (e.g., using sine/cosine transforms of start dates) and categorical date encodings were tested, but excluded in the final model due to data leakage.

⚠️ **Preventing Data Leakage**  
It was clear at the outset that most of the engineered begin_date features would lead to data leakage when used alongside the customer_duration feature, since combining both would allow the model to easily infer the churn label. However, it seemed plausible that certain features, such as start_dayofweek, might avoid this problem because they do not directly indicate the start date of the customer's subscription. During experimentation, though, it became evident that even the start_dayofweek features allowed the model to narrow the range of potential start dates with enough precision to create data lekage when combined with customer_duration. As a result, these features were also excluded when customer_duration was used in modeling.  

🧪 **Models Used**  
XGBoost

LightGBM

CatBoost

📊**Evaluation Metrics**  
Performance was evaluated using:

ROC AUC (primary metric)

Accuracy

Recall

🔎 **Model Interpretability**  
SHAP (SHapley Additive exPlanations) was used to interpret model predictions and understand feature importance.

💼**Potential Business Applications**  
Though no business-specific recommendations are included in this version, the SHAP visualizations provide actionable insights that could inform retention strategies, such as:

Identifying at-risk customer segments

Targeting promotional offers

📁**Repo Structure**
```
predicting_cust_churn/
│
├── contract.csv
├── internet.csv
├── personal.csv
├── phone.csv
├── test_data_cat.csv
├── test_data_light_xgb.csv
├── train_data_light_xgb.csv
├── train_data_cat.csv
│
├── notebooks/
│   ├── sprint_17_project_model_prep_test.ipynb   # Data cleaning, FE, and testing
│   └── sprint_17_project_report.ipynb            # EDA, model results, SHAP, summary
│
├── src/
│   ├── model_utils.py           # Common utility functions: metrics, preprocessing, etc.
│   ├── lightgbm_gpu_tuning.py       # Hyperparameter tuning for LightGBM
│   ├── xgboost_gpu_tuning.py        # Hyperparameter tuning for XGBoost
│   ├── catboost_gpu_tuning.py       # Hyperparameter tuning for CatBoost
│   └── xgb_testing.py    # Final testing using best model from tuning
│
├── README.md                    # Project overview, instructions, findings
├── requirements.txt             # List of required Python packages
└── .gitignore                   # Ignore compiled files, .ipynb_checkpoints, etc.
