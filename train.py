import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score#, roc_auc_score

from keras.models import Sequential
from keras.layers import Dense
from keras.losses import BinaryCrossentropy

import sys
np.set_printoptions(threshold=np.inf)

def main():
    
    if len(sys.argv) != 2:
        return
    try:
        pipline = joblib.load("pipline.pkl")

        df = pd.read_csv(sys.argv[1])
        y = df['TARGET']
        X = df.drop(columns=['ID', 'TARGET'])
        X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

        pipline.fit(X_train)

        X_train = pipline.transform(X_train)
        X_test  = pipline.transform(X_test)




        print(X.shape)

    except Exception as e:
        print(str(e))

if __name__ == "__main__":
    main()