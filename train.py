import joblib
import pandas as pd
import numpy as np

from sklearn.metrics import roc_auc_score, precision_score#, accuracy_score
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.losses import BinaryCrossentropy
from keras.optimizers import Adam
import sys


def main():
    
    if len(sys.argv) < 4:
        print("Usage: python script.py data/*_train.csv data/*_test.csv")
        return
    try:
        preprocessor = joblib.load(sys.argv[3])
        selector     = joblib.load("selector.pkl")

        model = Sequential([
            Dense(64, activation='relu', input_shape=(len(selector.columns_to_keep_) - 1,), kernel_initializer='he_uniform'),
            Dropout(0.2),
            Dense(24, activation='relu', kernel_initializer='he_uniform'),
            Dense(1, activation='sigmoid')
        ])

        model.compile(
            loss=BinaryCrossentropy(),
            optimizer=Adam(learning_rate=0.001),
            metrics=['Precision', 'AUC']
        )


        df_train = pd.read_csv(sys.argv[1])

        y = df_train['TARGET']
        X = df_train.drop(columns=['TARGET'])
        X_transformed = preprocessor.transform(X)

        model.fit(X_transformed, y, epochs=50, batch_size=16)

        del df_train
        del X_transformed
        del y
        del X
        
        df_test = pd.read_csv(sys.argv[2])
        y_test = df_test['TARGET']
        X_test = df_test.drop(columns=['TARGET'])
        X_test_transformed = preprocessor.transform(X_test)

        # y_pred = model.predict(X_test_transformed)
        # y_pred = y_pred.ravel()

        # for chunk in pd.read_csv(sys.argv[1], chunksize=50_000):
        #     y = chunk['TARGET']
        #     chunk = chunk.drop(columns=['TARGET'])

        #     transformed = preprocessor.transform(chunk)
        #     model.fit(transformed, y, epochs=50, batch_size=16)


        # test_dataset = pd.read_csv(sys.argv[2])
        # y_test = test_dataset['TARGET']
        # X_test = test_dataset.drop(columns=['TARGET'])

        y_pred = model.predict(X_test_transformed)
        auc = roc_auc_score(y_test, y_pred)

        y_pred_class = (y_pred >= 0.5).astype(int)
        prec = precision_score(y_true=y_test, y_pred=y_pred_class)

        try:
            with open('scores.txt', 'a') as file:
                file.write(f"auc : {auc}\nprecision: {prec}\n\n")
        except IOError as e:
            print(e)

    except Exception as e:
        print("error:", str(e))

if __name__ == "__main__":
    main()