# import os
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import sys

import pandas as pd
import joblib
import numpy as np


from sklearn.neural_network import MLPClassifier
from chunked_preprocessor import ChunkedPreprocessor

# from keras.layers import Dense
# from keras import Sequential
# from keras.losses import BinaryCrossentropy
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score#, log_loss


def main():
    if len(sys.argv) != 3:
        sys.exit(1)
    try:
        preprocessor : ChunkedPreprocessor = joblib.load("preprocessor.pkl")
        x_shape = len(preprocessor.imputer.kept_columns)

        # usecols = preprocessor.imputer.kept_columns.index.to_list().copy()
        # usecols.append('TARGET')

        # sklearn_sol = MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu', solver='sgd', batch_size=128)
        # classes     = np.array([0, 1])

        # keras_sol   = Sequential([
        #         Dense(64, activation='relu', input_shape=(x_shape,)),
        #         Dense(32, activation='relu'),
        #         Dense(1, activation='sigmoid')
        # ])

        # keras_sol.compile(
        #     optimizer='adam',
        #     loss=BinaryCrossentropy(from_logits=False),
        #     metrics=['accuracy'],
        # )

        # xn = None
        # for chunk in pd.read_csv(sys.argv[1], chunksize=int(sys.argv[2]), usecols=usecols):

        #     X = preprocessor.transform(chunk)
        #     if xn is None:
        #         xn = X.isnull().sum()
        #     else:
        #         xn += X.isnull().sum()
        #     # y = chunk['TARGET'].to_numpy()
     

        # print("sum of x nan :", xn)

    except Exception as e:
        print(str(e))




if __name__ == '__main__':
    main()