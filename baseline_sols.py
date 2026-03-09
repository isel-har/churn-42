
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import roc_auc_score
import pandas as pd
import joblib
import numpy as np
import sys
# import gc

transformer = joblib.load("transformer.pkl")
train_df = pd.read_csv(sys.argv[1])
y = train_df['TARGET'].copy()
train_df = train_df.drop(columns=['TARGET'])

transformed_df = transformer.transform(train_df)

transformed_df = transformed_df.astype(np.float32)
y = y.astype(np.float32)

del train_df

naive = BernoulliNB()

param_grid = {
    'n_estimators': [100],      # Number of trees in the forest
    'max_depth': [None],           # Maximum depth of the tree
    'min_samples_split': [2, 5],      # Min samples required to split a node
    'min_samples_leaf': [1, 2],        # Min samples required at a leaf node
}

rclr = RandomForestClassifier(random_state=42)
# cv=5 means 5-fold cross-validation
cv = GridSearchCV(estimator=rclr, param_grid=param_grid, cv=2, n_jobs=-1, verbose=2)


naive.fit(X=transformed_df, y=y)
cv.fit(X=transformed_df, y=y)

del transformed_df
del y


test_df = pd.read_csv(sys.argv[2])
y_test = test_df['TARGET']
X_test = test_df.drop(columns=['TARGET'])
transformed_test_df = transformer.transform(X_test)
transformed_test_df = transformed_test_df.astype(np.float32)
y_test = y_test.astype(np.float32)
del X_test


naive_y_pred = naive.predict(X=transformed_test_df).ravel()
rcv_y_pred   = cv.predict(X=transformed_test_df).ravel()

naive_auc_score = roc_auc_score(y_test, naive_y_pred)
rcv_auc_score   = roc_auc_score(y_test, rcv_y_pred)

print(f"naive AUC score: {naive_auc_score}")
print(f"random forst + grid seach AUC score: {naive_auc_score}")
