
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import GridSearchCV
import pandas as pd
import joblib
import numpy as np
import sklearn
from lib.tnn_wrapper import TNetWrapper


sklearn.set_config(transform_output="pandas")
pd.set_option('display.max_colwidth', None)


transformer = joblib.load("transformer.pkl")
train_df = pd.read_csv("data/balanced_train_data.csv")
y = train_df['TARGET'].copy().astype(np.float32)
train_df = train_df.drop(columns=['TARGET'])
X_transformed = transformer.transform(train_df).astype(np.float32)
del train_df


param_grid = {
    "hidden_units": [8, 16, 32],
    "learning_rate": [0.01, 0.001],
    "epochs": [10, 20]
}

grid = GridSearchCV(TNetWrapper(), param_grid=param_grid, cv=3, n_jobs=-1, scoring='roc_auc_ovr')
grid.fit(X_transformed, y)


del X_transformed
del y

test_df = pd.read_csv("data/split_data_test.csv")
y_test = test_df['TARGET'].astype(np.float32)
X_test = test_df.drop(columns=['TARGET'])
X_transformed_test = transformer.transform(X_test).astype(np.float32)

del X_test

model = grid.best_estimator_

y_prob = model.predict(X_transformed_test)
y_pred = model.predict_classes(X_transformed_test)

result = pd.DataFrame({
    'library': ["scikit-learn, tensorflow"],
    'algorithm': ['Multilayer-perceptron'],
    'hyerparamters': ['hidden layers : (64, 32, 16) + relu activation, epochs 10, learning rate 0.001 batch_size 32'],
    'Accuracy': [accuracy_score(y_test, y_pred)],
    'AUC': [roc_auc_score(y_test, y_prob)]
}, index=None)

print(result.T)
