import pandas as pd
import joblib
import sklearn


from sklearn.metrics import  roc_auc_score, accuracy_score
from sklearn.model_selection import GridSearchCV
from lib.nn_wrapper import NNetWrapper
import numpy as np


sklearn.set_config(transform_output="pandas")
pd.set_option('display.max_colwidth', None)

preprocessor = joblib.load("transformer.pkl")

df = pd.read_csv("data/balanced_train_data.csv")
val_df = pd.read_csv("data/split_data_val.csv")

y  = df['TARGET'].copy().astype(np.float32)
df = df.drop(columns=['TARGET'])
X_transformed = preprocessor.transform(df).astype(np.float32)

y_val = val_df['TARGET'].copy().astype(np.float32)
val_df = val_df.drop(columns=['TARGET'])
X_val_transformed = preprocessor.transform(val_df).astype(np.float32)

del df, val_df


param_grid = {
    "batch_size": [256],
    "epochs": [100]
}

grid = GridSearchCV(
    estimator=NNetWrapper(),
    param_grid=param_grid,
    scoring='roc_auc_ovr',
    n_jobs=-1,
    cv=3
)

grid.fit(X_transformed, y, validation_data=(X_val_transformed, y_val))

del X_transformed, y

test_split_df = pd.read_csv('data/split_data_test.csv')

y_test        = test_split_df['TARGET'].astype(np.float32)
test_split_df = test_split_df.drop(columns=['TARGET'])
X_test_split  = preprocessor.transform(test_split_df).astype(np.float32)

model = grid.best_estimator_

y_prob    = model.predict_proba(X_test_split)
y_pred    = (y_prob > 0.5).astype(int)

result = pd.DataFrame({
    'library': ["scikit-learn, tensorflow"],
    'algorithm': ['Multilayer-perceptron'],
    'hyerparamters': ['swa'],
    'accuracy': [accuracy_score(y_test, y_pred)],
    'auc': [roc_auc_score(y_test, y_prob)]
}, index=None)

print(result.T)


# del test_split_df, X_test_split, y_test, y_pred

# df = pd.read_csv("data/bank_data_test.csv")

# ids = df['ID'].values.tolist()

# X_test = preprocessor.transform(df).astype(np.float32)

# y_pred = grid.predict(X_test)

# preds_df = pd.DataFrame(data={'ID':ids, 'TARGET':y_pred.ravel().tolist()}, columns=['ID', 'TARGET'])

# preds_df.to_csv("prediction_file.csv", index=False)
# print("prediction_file.csv saved")