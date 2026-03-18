import pandas as pd
import joblib
import sklearn

from sklearn.metrics import roc_auc_score

from lib.earlystopping import EarlyStopping
from lib.losses import BinaryCrossEntropy
from lib.activations import Relu, Sigmoid
from lib.layers import DenseLayer
from lib.optimizers import Adam
from lib.nnet import NNet

sklearn.set_config(transform_output="pandas")

try:
    preprocessor = joblib.load("transformer.pkl")
    df = pd.read_csv("data/balanced_train_data.csv")
    val_df = pd.read_csv("data/split_data_val.csv")
    y  = df['TARGET'].copy()
    df = df.drop(columns=['TARGET'])
    X_transformed = preprocessor.transform(df)

    y_val = val_df['TARGET'].copy()
    val_df = val_df.drop(columns=['TARGET'])
    X_val_transformed = preprocessor.transform(val_df)

    del df, val_df

    model = NNet(
        layers=[
            DenseLayer(input_size=X_transformed.shape[1], output_size=64, activation=Relu()),
            DenseLayer(output_size=32, activation=Relu()),
            DenseLayer(output_size=16, activation=Relu()),

            DenseLayer(
                output_size=1,
                activation=Sigmoid(),
                weights_initializer='xavier'
            )
        ],
        earlystopping=EarlyStopping(patience=30),
        optimizer=Adam(learning_rate=0.0001),
        loss=BinaryCrossEntropy()
    )

    model.fit(
        X_transformed,
        y,
        batch_size=16,
        epochs=250,
        validation_data=(X_val_transformed, y_val)
    )
    del X_transformed, y

    df = pd.read_csv("data/split_data_test.csv")
    y_test  = df['TARGET'].to_numpy()
    df = df.drop(columns=['TARGET'])
    X_test = preprocessor.transform(df)
    y_pred = model.predict(X_test)

    auc_score = roc_auc_score(y_test, y_pred)
    print("auc score :", auc_score)

except Exception as e:
    print(str(e))