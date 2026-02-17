import numpy as np
from chunked_imputer import ChunkedImputer
from chunked_encoder import ChunkedEncoder
from chunked_scaler import ChunkedScaler


class ChunkedPreprocessor:

    def __init__(self, 
        missing_threshold=0.5,
        chunksize=10000):
        self.imputer = ChunkedImputer(missing_threshold, chunksize)
        self.encoder = ChunkedEncoder()
        self.scaler  = ChunkedScaler()
        return self

    def fit(self, filepath, to_drop, strategies=None):

        self.imputer.fit(filepath, to_drop, strategies['imputation'])
        # self.encoder.fit(filepath, to_drop, strategies['encoder'])
        # self.scaler.fit(filepath, to_drop, strategies['scaler'])

    def transform(self, filepath, output_path=None):
        ...

