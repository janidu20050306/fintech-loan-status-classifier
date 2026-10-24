import ast
import datetime
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


df = pd.read_csv('sampleDataset.csv')


def parse_employment(emp_str):
    try:
        data = ast.literal_eval(emp_str) if isinstance(emp_str, str) else emp_str

        if isinstance(data, list) and len(data) > 0:
            return pd.Series([
                np.mean([item.get('income', 0) for item in data]),
                np.sum([item.get('years', 0) for item in data]),
                len(data),
            ])
    except Exception:
        pass
    return pd.Series([np.nan, np.nan, np.nan])


emp_features = df['Customer Profile.employment_history'].apply(parse_employment)
emp_features.columns = ['emp_avg_income', 'emp_total_years', 'emp_job_changes']


def parse_transactions(tx_str):
    try:
        data = (
            eval(tx_str, {'datetime': datetime, '__builtins__': {}})
            if isinstance(tx_str, str)
            else tx_str
        )
        if isinstance(data, list):
            return pd.Series([len(data), np.sum([item.get('amount', 0) for item in data])])
    except Exception:
        pass
    return pd.Series([0, 0.0])


tx_features = df['Transaction Description'].apply(parse_transactions)
tx_features.columns = ['flagged_tx_count', 'flagged_tx_total_amount']


processed_df = pd.concat([df, emp_features, tx_features], axis=1)

processed_df['income_level_clean'] = processed_df['Customer Profile.income_level'].fillna(processed_df['emp_avg_income'])
processed_df['dti_ratio'] = processed_df['Loan Application Summary.loan_amount'] / (processed_df['income_level_clean'] * 12 + 1e-5)
processed_df['liquidity_buffer'] = processed_df['Account Activity.balance'] / (processed_df['Loan Application Summary.loan_amount'] + 1e-5)
processed_df['net_cash_flow'] = processed_df['Account Activity.deposits'] - processed_df['Account Activity.withdrawals']


num_cols = [
    'Account Activity.balance', 'income_level_clean', 'Loan Application Summary.interest_rate',
    'Account Activity.deposits', 'Account Activity.withdrawals', 'Loan Application Summary.loan_amount',
    'Loan Application Summary.loan_term', 'emp_avg_income', 'emp_total_years', 'emp_job_changes',
    'flagged_tx_count', 'flagged_tx_total_amount', 'dti_ratio', 'liquidity_buffer', 'net_cash_flow',
]
cat_cols = [
    'Loan Application Summary.employment_status', 'Customer Profile.marital_status',
    'Customer Profile.education_level', 'Customer Profile.region',
]

x = processed_df[num_cols + cat_cols].copy()
le = LabelEncoder()
y = le.fit_transform(processed_df['Loan Application Summary.loan_status'].fillna('Unknown'))

preprocessor = ColumnTransformer(
    transformers=[
        ('num', SimpleImputer(strategy='mean'), num_cols),
        ('cat', Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore'))]), cat_cols)
    ]
)

model = Pipeline(steps=[('preprocessor', preprocessor), ('clf', GradientBoostingClassifier(random_state=42))])

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)
print(classification_report(y_test, y_pred, target_names=le.classes_))
print(f"ROC AUC (macro, OVR): {roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='macro'):.4f}")
