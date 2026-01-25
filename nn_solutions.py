import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import pandas as pd
import numpy as np
import joblib


from sklearn.neural_network import MLPClassifier
from keras.layers import Dense
from keras import Sequential
from keras.losses import BinaryCrossentropy
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score#, log_loss

CAT_NA_VALUE = "__MISSING__"

preprocessor = joblib.load("preprocessor.joblib")

mean          = preprocessor['mean']
std_dev       = preprocessor['std']
kept_num_cols = preprocessor['kept_num_cols']


kept_cat_cols = preprocessor['kept_cat_cols']
encoder       = preprocessor['encoder']

dtype = preprocessor['dtypes']








sklearn_sol = MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu', solver='sgd', batch_size=128)
classes     = np.array([0, 1])

keras_sol   = Sequential([
        Dense(64, activation='relu', input_shape=(56,)),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
])

keras_sol.compile(
    optimizer='adam',
    loss=BinaryCrossentropy(from_logits=False),
    metrics=['accuracy']
)


chunks = pd.read_csv('data/bank_data_train.csv', dtype=dtype, chunksize=100_000)
for i, chunk in enumerate(chunks, start=1):
    chunk[kept_num_cols] = chunk[kept_num_cols].fillna(0.0)
    chunk[kept_cat_cols] = chunk[kept_cat_cols].fillna(CAT_NA_VALUE)

    X_cat_enc = encoder.transform(chunk[kept_cat_cols])
    X_cat_enc = pd.DataFrame(X_cat_enc, columns=encoder.get_feature_names_out(kept_cat_cols), index=chunk.index)
    X_num = (chunk[kept_num_cols] - mean) / std_dev

    X    = pd.concat([X_num, X_cat_enc], axis=1)
    y    = chunk['TARGET']

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y)

    print(f"chunk __________[{i}]__________")

    print("fit keras sequential solution")
    keras_sol.fit(
        X_train, y_train,
        epochs=10,
        batch_size=128,
        validation_data=(X_val, y_val)
    )
    # print("fit Sklearn MLPClassifier solution")
    # sklearn_sol.partial_fit(X_train, y_train, classes=classes)


test_df = pd.read_csv("data/bank_data_test.csv")


test_df[kept_num_cols] = test_df[kept_num_cols].fillna(0.0)
test_df[kept_cat_cols] = test_df[kept_cat_cols].fillna(CAT_NA_VALUE)

X_cat_enc = encoder.transform(test_df[kept_cat_cols])

X_cat_enc = pd.DataFrame(X_cat_enc, columns=encoder.get_feature_names_out(kept_cat_cols), index=test_df.index)
X_num = (test_df[kept_num_cols] - mean) / std_dev

X    = pd.concat([X_num, X_cat_enc], axis=1)
y_pred = keras_sol.predict(X)
## use numpy argmax
print(y_pred)
