from chunked_preprocessor import ChunkedPreprocessor
import sys
# import joblib

# pd.set_option('display.max_columns', None)

def main():

    if len(sys.argv) != 2:
        return
    
    preprocessor = ChunkedPreprocessor(missing_threshold=0.5, chunksize=100_000)

    preprocessor.fit(sys.argv[1], to_drop=['ID'])

    # print(preprocessor.impute_values_)

    # joblib.dump(preprocessor, "preprocessor.pkl")







if __name__ == "__main__":
    main()