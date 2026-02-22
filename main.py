# import os
from chunked_preprocessor import ChunkedPreprocessor
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
import sys
import pandas as pd
# from imputer import Imputer
# from encoder import Encoder
# import joblib

# from keras.models import Sequential
# from keras.layers import Dense
# from keras.losses import BinaryCrossentropy

pd.set_option('display.max_rows', None)
# pd.set_option('display.max_rows', None)


def main():

    if len(sys.argv) != 2:
        sys.exit(1)
    
    strategies = {
        'imputers':{
            (0, 0.25):(SimpleImputer(strategy='mean'), SimpleImputer(strategy="most_frequent")),
            (0.25, 0.50): (KNNImputer(n_neighbors=6), SimpleImputer(strategy="constant", fill_value='MISSING'))
        },
        'encoders':{
            'CLNT_JOB_POSITION':OrdinalEncoder(),
            'PACK':OrdinalEncoder()
        }
    }
    try:

        preprocessor = ChunkedPreprocessor(chunksize=100_000)

        preprocessor.fit(sys.argv[1], to_drop=['ID', 'TARGET'], strategies=strategies)

        # x_nans = None

        sample = pd.read_csv(sys.argv[1], nrows=1000, usecols=preprocessor.imputer.columns.index.to_list())

        X = preprocessor.transform(sample)

        x_nans = X.isnull().sum()
        print(x_nans)

    except Exception as e:
        print("exception :", str(e))



if __name__ == "__main__":
    main()