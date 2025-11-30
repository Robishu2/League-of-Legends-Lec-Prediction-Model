import numpy as np
import pandas as pd

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold

from DataGathering import request_data
from DataPrep import add_lags, data_prep, rolling_window

if __name__ == '__main__':
    lec_df = request_data()

    lec_df = data_prep(lec_df)

    home_cols = [x for x in lec_df.columns if 'home' in x.lower() and lec_df[x].dtype == float]
    away_cols = [x for x in lec_df.columns if 'away' in x.lower() and lec_df[x].dtype == float]

    temp = rolling_window(lec_df, home_cols, away_cols)
    lec_df = add_lags(lec_df, home_cols, away_cols, number_of_lags=3)

    y = lec_df['result_home']
    X = lec_df.drop(columns=['result_home']).copy()

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3, random_state=1231)

    best_params = {
    'n_estimators': 500,
    'max_depth': 5,
    'gamma': 1,
    'colsample_bytree': 0.7,
    'learning_rate': 0.03,
    'reg_lambda': 1.5,
    'reg_alpha': 0.5,
    'n_jobs': -1,
    'enable_categorical': True,
    'tree_method': 'hist',
}

model = XGBClassifier(**best_params)
skf = StratifiedKFold(n_splits=10, random_state=1231, shuffle=True)

accuracy_list = []

for train_idx, test_idx in skf.split(X, y):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    model.fit(X_train, y_train)
    preds_proba = model.predict_proba(X_test)[:, 1]

    preds = (preds_proba >= 0.5).astype(int)
    acc = (preds == y_test).mean()
    accuracy_list.append(acc)

print(np.mean(accuracy_list))
