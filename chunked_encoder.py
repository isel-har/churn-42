import pandas as pd

class ChunkedEncoder:

    def __init__(self, chunksize=10000):
        self.chunksize = chunksize
        self.unique_categories = {}


    def fit(self, filepath, imputer=None, strategies=None):

        cols = imputer.columns_to_encode
        for col in cols:
            self.unique_categories[col] = set()

        for chunk in pd.read_csv(filepath, chunksize=self.chunksize, usecols=cols):

            X = chunk
            for key, value in imputer.cat_imputers.items():
                obj = value['imputer']
                i_cols  = value['cols']
                X[i_cols] = obj.transform(X[i_cols])

            for col in cols:
                self.unique_categories[col].update(list(X[col].astype(str).unique()))


            


    def transform(self):
        ...