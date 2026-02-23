import joblib
import pandas as pd
import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score#, roc_auc_score

# from keras.models import Sequential
# from keras.layers import Dense
# from keras.losses import BinaryCrossentropy

import sys
np.set_printoptions(threshold=np.inf)

def main():
    
    if len(sys.argv) != 2:
        return
    try:
        pipline = joblib.load("pipline.pkl")

        df = pd.read_csv(sys.argv[1], nrows=100)
        # y = df['TARGET']
        X = df.drop(columns=['ID', 'TARGET'])


        X = pipline.transform(X)
        print(X)


        # usecols = preprocessor.imputer.columns.index.tolist()
        # usecols.append('TARGET')
        # sample = pd.read_csv(sys.argv[1], usecols=usecols, nrows=120000)

        # X = preprocessor.transform(sample[preprocessor.imputer.columns.index.tolist()])
        # y = sample['TARGET']

        # X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42, test_size=0.2)

        # model = Sequential([
        #     Dense(64, activation='relu', input_shape=(len(usecols)-1, )),
        #     Dense(32, activation='relu'),
        #     Dense(1, activation='sigmoid')
        # ])
        # model.compile(optimizer='adam')

        # model.fit(X_train.to_numpy(), y.to_numpy(), epochs=50)


        # y_pred = model.predict(X_test.to_nompy())

        # print(y_pred)
        # accuracy = accuracy_score(y_true=y_test.to_numpy(), y_pred=y_pred)

        print("done!")
    except Exception as e:
        print(str(e))

if __name__ == "__main__":
    main()