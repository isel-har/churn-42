from sklearn.compose import ColumnTransformer

from chunked_preprocessor import ChunkedPreprocessor
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
import joblib
import sys
import pandas as pd

# import


# pd.set_option('display.max_columns', None)

def main():

    if len(sys.argv) != 2:
        sys.exit(1)
    
    strategies = {
        'imputation':{
            'samplesize':10000,
            'num':{
                (0, 0.20) :SimpleImputer(strategy='median'),
                (0.21, 0.30):KNNImputer(n_neighbors=6),
            },
            'cat':{
                (0.31, 0.50):SimpleImputer(strategy='constant', fill_value='MISSING')
            }
        },
        'encoding': [['CLNT_JOB_POSITION', OrdinalEncoder()], ['PACK', OrdinalEncoder()]]
    }
    try:
        preprocessor = ChunkedPreprocessor(missing_threshold=0.5, chunksize=10_000)
        preprocessor.fit(sys.argv[1], to_drop=['ID', 'TARGET'], strategies=strategies)



        usecols = preprocessor.imputer.kept_columns.index.tolist()
        sample =  pd.read_csv(sys.argv[1], nrows=1000, usecols=usecols)
        
        print("transforming sample ...")
        sample = preprocessor.transform(sample)
        # print(sample.head(50))




        # joblib.dump(preprocessor, "preprocessor.pkl")
    except Exception as e:
        print("exception :", str(e))






if __name__ == "__main__":
    main()