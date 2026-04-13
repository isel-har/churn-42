from sklearn.base import BaseEstimator, ClassifierMixin
from .tf_modules import Dense, TNet
import numpy as np
import tensorflow as tf


class TNetWrapper(BaseEstimator, ClassifierMixin):
    _estimator_type = "classifier"
    def __init__(self, hidden_units=10, epochs=10, learning_rate=0.001, batch_size=32):
        self.hidden_units = hidden_units
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.model_ = None  # will hold trained model

    def _build_model(self, input_dim):
        return TNet(layers=[
            Dense(input_dim, self.hidden_units, activation=tf.nn.relu),
            # Dense(self.hidden_units, self.hidden_units, activation=tf.nn.relu),
            Dense(self.hidden_units, 1, activation=tf.nn.sigmoid)
        ])

    def fit(self, X, y):
        X = tf.convert_to_tensor(X, dtype=tf.float32)
        y = tf.convert_to_tensor(y, dtype=tf.float32)

        self.model_ = self._build_model(X.shape[1])

        self.model_.fit(
            X, y,
            epochs=self.epochs,
            learning_rate=self.learning_rate,
            batch_size=self.batch_size
        )
        return self

    def predict_proba(self, X):
        X = tf.convert_to_tensor(X, dtype=tf.float32)
        preds = self.model_.predict(X)
        return preds.numpy().flatten()


    # def predict_proba(self, X):
    #     return self.model_.predict_classes(X)


    def score(self, X, y):
        preds = self.predict(X)
        return np.mean(preds == y)
