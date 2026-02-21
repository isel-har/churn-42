from chunked_scaler import ChunkedStandardScaler
from chunked_imputer import ChunkedImputer
from chunked_encoder import ChunkedEncoder


class ChunkedPreprocessor:

    def __init__(self, missing_threshold=0.5, chunksize=10000):

        self.imputer = ChunkedImputer(missing_threshold, chunksize)
        self.encoder = ChunkedEncoder(chunksize)
        self.scaler  = ChunkedStandardScaler(chunksize)


    def fit(self, filepath, to_drop=[], strategies=None):

        self.imputer.fit(filepath, to_drop=to_drop, strategies=strategies['imputation'])
        self.encoder.fit(filepath, imputer=self.imputer, strategies=strategies['encoding'])
        self.scaler.fit(filepath, imputer=self.imputer)

        return self

    # def transform(self, filepath, output_path=None):
    #     ...

    def transform(self, chunk):

        print("transforming...")
        cols_encode = self.imputer.columns_to_encode
        cols_scale  = self.imputer.columns_to_scale

        chunk = self.imputer.transform(chunk)

        """
            transform/change only selected indexes
        # """
        chunk[cols_encode] = self.encoder.transform(chunk[cols_encode])

        """
            the problem probably here (columns problem)
        """

        chunk[cols_scale] = self.scaler.transform(chunk[cols_scale])
        return chunk

