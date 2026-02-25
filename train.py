import joblib
import pandas as pd
import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score#, roc_auc_score

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.losses import BinaryCrossentropy
from keras.optimizers import Adam
import sys
# np.set_printoptions(threshold=np.inf)
# pd.set_option('display.max_columns', None)
def main():
    
    if len(sys.argv) != 2:
        return
    try:
        preprocessor = joblib.load("preprocessor.pkl")
        selector     = joblib.load("selector.pkl")

        model = Sequential([
            Dense(128, activation='relu', input_shape=(len(selector.columns_to_keep_) - 1,)),
            Dropout(0.3),
            Dense(64, activation='relu'),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(1, activation='sigmoid')
        ])

        model.compile(
            loss=BinaryCrossentropy(),
            optimizer=Adam(learning_rate=0.001),
            metrics=['Precision', 'Recall', 'AUC', 'accuracy']
        )

        for chunk in pd.read_csv(sys.argv[1], chunksize=50_000):
            y = chunk['TARGET']
            chunk = chunk.drop(columns=['TARGET'])

            transformed = preprocessor.transform(chunk)
            model.fit(transformed, y, epochs=50, batch_size=32)


        print("done!")

    except Exception as e:
        print("error:", str(e))

if __name__ == "__main__":
    main()