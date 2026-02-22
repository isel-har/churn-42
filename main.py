# import os
# from chunked_preprocessor import ChunkedPreprocessor
from sklearn.impute import KNNImputer, SimpleImputer
# from sklearn.preprocessing import OrdinalEncoder
import sys
import pandas as pd
from imputer import Imputer
# import joblib

# from keras.models import Sequential
# from keras.layers import Dense
# from keras.losses import BinaryCrossentropy

pd.set_option('display.max_rows', None)


def main():

    if len(sys.argv) != 2:
        sys.exit(1)
    
    # strategies = {
    #     'imputation':{
    #         'samplesize':10000,
    #         'num': {
    #             (0, 0.20): SimpleImputer(strategy='mean'),
    #             (0.20, 0.30): KNNImputer(n_neighbors=6),
    #         },
    #         'cat':{
    #             (0.31, 0.50):SimpleImputer(strategy='constant', fill_value='MISSING')
    #         }
    #     },
    #     'encoding': [['CLNT_JOB_POSITION', OrdinalEncoder()], ['PACK', OrdinalEncoder()]]
    # }
    try:

        imputer = Imputer(chunksize=100000)

        imputers_ = {
            (0, 0.25):(SimpleImputer(strategy='mean'), SimpleImputer(strategy="most_frequent")),
            (0.25, 0.50): (KNNImputer(n_neighbors=6), SimpleImputer(strategy="constant", fill_value='MISSING'))
        }
        imputer.fit(sys.argv[1], cols_to_drop=['ID', 'TARGET'], imputers=imputers_)


        # preprocessor = ChunkedPreprocessor(missing_threshold=0.5, chunksize=10_000)
        # preprocessor.fit(sys.argv[1], to_drop=['ID', 'TARGET'], strategies=strategies)

        sample = pd.read_csv(sys.argv[1], nrows=100000, usecols=imputer.columns.index.to_list())

        X = imputer.transform(sample)

        xn = X.isnull().sum()

        print(xn)        

    except Exception as e:
        print("exception :", str(e))



if __name__ == "__main__":
    main()