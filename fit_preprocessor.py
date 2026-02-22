from chunked_preprocessor import ChunkedPreprocessor
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import OrdinalEncoder

import pandas as pd
import joblib
import sys
# from imputer import Imputer
# from encoder import Encoder

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
            (0, 0.25):(SimpleImputer(strategy='mean'), SimpleImputer(strategy="constant", fill_value='MISSING')),
            (0.25, 0.50): (KNNImputer(n_neighbors=6), SimpleImputer(strategy="constant", fill_value='MISSING'))
        },
        'encoders':{
            # 'CLNT_JOB_POSITION':OrdinalEncoder(),
            'PACK':OrdinalEncoder()
        }
    }
    try:

        preprocessor = ChunkedPreprocessor(chunksize=100_000)

        preprocessor.fit(sys.argv[1], to_drop=['ID', 'TARGET', 'CLNT_JOB_POSITION'], strategies=strategies)

        joblib.dump(preprocessor, "preprocessor.pkl")
        
    except Exception as e:
        print("exception :", str(e))


if __name__ == "__main__":
    main()