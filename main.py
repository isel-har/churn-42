from sklearn.compose import ColumnTransformer

from chunked_preprocessor import ChunkedPreprocessor
from sklearn.impute import KNNImputer, SimpleImputer
import joblib
import sys

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
        }
    }
    try:
        preprocessor = ChunkedPreprocessor(missing_threshold=0.5, chunksize=100_000)
        preprocessor.fit(sys.argv[1], to_drop=['ID'], strategies=strategies)

        # joblib.dump(preprocessor, "preprocessor.pkl")
    except Exception as e:
        print("exception :", str(e))






if __name__ == "__main__":
    main()