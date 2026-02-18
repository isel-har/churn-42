from chunked_imputer import ChunkedImputer
from chunked_encoder import ChunkedEncoder
# from chunked_scaler import ChunkedScaler
# import numpy as np


class ChunkedPreprocessor:

    def __init__(self, 
        missing_threshold=0.5,
        chunksize=10000):
        self.imputer = ChunkedImputer(missing_threshold, chunksize)
        self.encoder = ChunkedEncoder(chunksize)
        # self.scaler  = ChunkedScaler(chunksize)

    def fit(self, filepath, to_drop=[], strategies=None):

        self.imputer.fit(filepath, to_drop=to_drop, strategies=strategies['imputation'])

        self.encoder.fit(filepath, imputer=self.imputer, strategies=None)

        # self.scaler.fit(filepath, cols=self.imputer.nem_cols, strategies=strategies['scaler'])

        return self






    def transform(self, filepath, output_path=None):
        ...

