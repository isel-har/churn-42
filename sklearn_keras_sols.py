import sys

import pandas as pd
import numpy as np
import joblib

import tensorflow as tf
from keras.layers import Dense, Dropout, BatchNormalization, LeakyReLU, GaussianNoise#, Activation
from keras.losses import BinaryCrossentropy
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.models import Sequential
from keras.optimizers import Adam


import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score#, precision_score#, accuracy_score
from sklearn.utils import class_weight
# from sklearn.neural_network import MLPClassifier

sklearn.set_config(transform_output="pandas")

tf.config.optimizer.set_jit(True)

def main():
    
    if len(sys.argv) < 3:
        print("Usage: python script.py data/*_train.csv data/*_test.csv")
        return
    try:

        transformer = joblib.load("transformer.pkl")
        train_df = pd.read_csv(sys.argv[1])
        y = train_df['TARGET'].astype(np.float32)
        train_df = train_df.drop(columns=['TARGET'])

        transformed_df = transformer.transform(train_df).astype(np.float32)
        del train_df

        keras = Sequential([
            GaussianNoise(0.05, input_shape=(transformed_df.shape[1],)),

            Dense(256),
            BatchNormalization(),
            LeakyReLU(0.1),
            Dropout(0.25),

            Dense(128),
            BatchNormalization(),
            LeakyReLU(0.1),
            Dropout(0.15),

            Dense(64),
            BatchNormalization(),
            LeakyReLU(0.1),
            Dropout(0.1),

            Dense(1, activation="sigmoid")
        ])

        # lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        #     initial_learning_rate=0.001,
        #     decay_steps=10000
        # )

        optimizer=Adam(learning_rate=0.0005)
        reduce_lr = ReduceLROnPlateau(
            monitor='val_auc',
            mode='max',
            factor=0.3,
            patience=5,
            min_lr=1e-6
        )
        keras.compile(
                loss=BinaryCrossentropy(label_smoothing=0.01),
                optimizer=optimizer,
                metrics=[tf.keras.metrics.AUC(curve='ROC', name='auc')]
        )
        weights = class_weight.compute_class_weight(
                class_weight='balanced',
                classes=np.unique(y),
                y=y
        )

        dict_weights = dict(enumerate(weights))
        early_stop = EarlyStopping(
            monitor='val_auc',
            mode='max',
            patience=15,
            restore_best_weights=True
        )
    
        X_train, X_val, y_train, y_val = train_test_split(
            transformed_df, y, test_size=0.15, stratify=y, random_state=42
        )
        train_dataset = (
            tf.data.Dataset
            .from_tensor_slices((X_train, y_train))
            .shuffle(10000)
            .batch(1024)
            .prefetch(tf.data.AUTOTUNE)
        )
        val_dataset = (
            tf.data.Dataset
            .from_tensor_slices((X_val, y_val))
            .batch(1024)
            .prefetch(tf.data.AUTOTUNE)
        )

        keras.fit(
            train_dataset, 
            epochs=200,
            batch_size=128,
            validation_data=val_dataset,
            callbacks=[early_stop, reduce_lr],
            class_weight=dict_weights
        )

        del transformed_df
        del y

        test_df = pd.read_csv(sys.argv[2])
        y_test = test_df['TARGET'].astype(np.float32)
        X_test = test_df.drop(columns=['TARGET'])
        transformed_test_df = transformer.transform(X_test).astype(np.float32)
        del X_test

        y_pred = keras.predict(transformed_test_df).ravel()
        auc    = roc_auc_score(y_test, y_pred)

        with open('scores.txt', 'a') as file:
            file.write(f"AUC: {auc}\n")


    except Exception as e:
        print("error:", str(e))

if __name__ == "__main__":
    main()