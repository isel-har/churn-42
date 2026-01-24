# from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split#, GridSearchCV
from sklearn.metrics import accuracy_score, auc
from sklearn.naive_bayes import BernoulliNB
import pandas as pd
import numpy as np
import joblib


preprocessor = joblib.load("preprocessor.joblib")

#scaling params
mean          = preprocessor['mean']
std_dev       = preprocessor['std']
kept_num_cols = preprocessor['ketp_num_cols']

#encoding
kept_cat_cols = preprocessor['ketp_cat_cols']
encoders      = preprocessor['encoders']

dtypes = preprocessor['dtypes']

naive = BernoulliNB()

chunks = pd.read_csv('data/bank_data_train.csv', dtypes=dtypes, chunksize=100_000)
for chunk in chunks:
    Y = chunk['TARGET'].to_numpy()
    
    for _ in range(len(kept_cat_cols)):
        encoder = encoders[_]
        chunk[kept_cat_cols[_]] = encoder.transform(chunk[kept_cat_cols[_]])

    chunk[kept_num_cols] = (chunk[kept_num_cols].to_numpy() - mean) / std_dev

    X_train, X_test, y_train, y_test = train_test_split(
        chunk[kept_num_cols].to_numpy(),
        Y,
        stratify=Y,
        random_state=42
    )

    print(f'naive solution partial fit...')
    naive.partial_fit(X_train, y_train)
    # print score for each chunk using test data split


df = pd.read_csv('data/bank_data_test.csv', dtypes=dtypes)

df[kept_num_cols] = (df[kept_num_cols] - mean) / std_dev
for _ in range(len(kept_cat_cols)):
        encoder = encoders[_]
        df[kept_cat_cols[_]] = encoder.transform(df[kept_cat_cols[_]])


y_pred = naive.predict(df[kept_num_cols + kept_cat_cols].to_numpy())

acc = accuracy_score(y_true=df['TARGET'].to_numpy(), y_pred=y_pred)
print('naive accuracy :', acc)
auc_m = auc(y_true=df['TARGET'].to_numpy(), y_pred=y_pred)
print('naive AUC :', auc_m)

