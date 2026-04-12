from sklearn.base import BaseEstimator, ClassifierMixin
from .class_weight import compute_class_weight
from sklearn.metrics import roc_auc_score
from .earlystopping import EarlyStopping
from .layers import DenseLayer, Dropout
from .losses import BinaryCrossEntropy
from .activations import Relu, Sigmoid
from .optimizers import Adam
from .nnet import NNet
import numpy as np



class NNetWrapper(BaseEstimator, ClassifierMixin):
    _estimator_type = "classifier"
    def __init__(self, learning_rate=0.001, batch_size=256, epochs=300):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs

    def build_model(self, input_dim):

        return NNet(
            layers=[
                DenseLayer(input_size=input_dim, output_size=64, activation=Relu()),
                Dropout(0.3),
                DenseLayer(output_size=32, activation=Relu()),
                Dropout(0.2),
                DenseLayer(output_size=16, activation=Relu()),
                # Dropout(0.1),
                # DenseLayer(output_size=8, activation=Relu()),
                DenseLayer(output_size=1, activation=Sigmoid(),  weights_initializer='xavier')
            ],
            earlystopping=EarlyStopping(patience=40, restore_best_weights=True),
            optimizer=Adam(learning_rate=self.learning_rate),
            loss=BinaryCrossEntropy()
        )

    def fit(self, X, y, validation_data=None):
        self.model_ = self.build_model(X.shape[1])

        class_weight = compute_class_weight(np.unique(y), y)

        self.model_.fit(
            X, y,
            epochs=self.epochs,
            batch_size=self.batch_size,
            class_weight=class_weight,
            validation_data=validation_data
        )
        return self

    def predict(self, X):
        probs = self.model_.predict(X)
        return (probs > 0.5).astype(int)

    def predict_proba(self, X):
        return self.model_.predict(X)

    def score(self, X, y):
        from sklearn.metrics import roc_auc_score
        return roc_auc_score(y, self.model_.predict(X))
