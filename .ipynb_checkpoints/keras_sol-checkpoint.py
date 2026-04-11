import os

import pandas as pd
import numpy as np
import sklearn
import joblib
import tensorflow as tf

import keras_tuner as kt

from keras.losses import BinaryCrossentropy
from keras.layers import Dense, Dropout, Input
from keras.models import Sequential
from keras.optimizers import Adam
from keras.models import Sequential
from keras.metrics import AUC
from keras.callbacks import EarlyStopping



from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.utils import class_weight

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
tf.get_logger().setLevel('ERROR')
# tf.config.optimizer.set_jit(True)
sklearn.set_config(transform_output="pandas")
pd.set_option('display.max_colwidth', None)




transformer   = joblib.load("transformer.pkl")
train_df      = pd.read_csv("data/balanced_train_data.csv")
val_df        = pd.read_csv("data/split_data_val.csv")

y_train       = train_df['TARGET'].copy().astype(np.float32)
train_df      = train_df.drop(columns=['TARGET'])
X_transformed = transformer.transform(train_df).astype(np.float32)

y_val  = val_df['TARGET'].copy().astype(np.float32)
val_df = val_df.drop(columns=['TARGET'])
X_val  = transformer.transform(val_df).astype(np.float32)
del train_df, val_df



weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)

dict_weights = dict(enumerate(weights))

early_stop = EarlyStopping(
    monitor='val_loss', 
    patience=20, 
    restore_best_weights=True
)

def build_model(hp):
    model = Sequential([
        Input(shape=(X_transformed.shape[1],)),
        Dense(
            units=hp.Int('units', min_value=32, max_value=128, step=32),
            activation=hp.Choice('activation', values=['relu']),
        ),
        Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=[AUC(name='auc'), 'accuracy']
    )
    return model


tuner = kt.GridSearch(
    build_model,
    objective=kt.Objective("val_auc", direction="max"),
    directory="tuner_dir",
    project_name="auc_search"
)


tuner.search(
    X_transformed, 
    y_train, 
    batch_size=64, 
    epochs=200, 
    validation_split=0.2,
    class_weight=dict_weights,
    callbacks=[early_stop],
    validation_data=(X_val, y_val)
)

best_hps = tuner.get_best_hyperparameters()[0]
model    = tuner.get_best_models(num_models=1)[0]

del X_transformed, y_train

test_df = pd.read_csv("data/split_data_test.csv")
y_test = test_df['TARGET'].astype(np.float32)
X_test = test_df.drop(columns=['TARGET'])
X_transformed_test = transformer.transform(X_test).astype(np.float32)

del X_test

y_prob = model.predict(X_transformed_test).ravel()
auc    = roc_auc_score(y_test, y_prob)
y_pred = (y_prob > 0.5).astype(int)

result = pd.DataFrame({
    'library': ["scikit-learn, keras"],
    'algorithm': ['Multilayer-perceptron'],
    'Accuracy': [accuracy_score(y_test, y_pred)],
    'AUC': [roc_auc_score(y_test, y_prob)]
}, index=None)


model.summary()

print(result.T)

