from scaler import Scaler
from encoder import Encoder
from imputer import Imputer


class ChunkedPreprocessor:

    def __init__(self, missing_threshold=0.5, chunksize=10000):

        self.imputer = Imputer(missing_threshold, chunksize)
        self.encoder = Encoder(chunksize)
        self.scaler  = Scaler(chunksize=10000)


    def fit(self, filepath, to_drop=[], strategies=None, sample_size=10_000):
        if not strategies:
            raise Exception("strategies dict required.")


        self.imputer.fit(filepath, to_drop, strategies['imputers'], sample_size)
        self.encoder.fit(filepath, imputer=self.imputer, encoders=strategies['encoders'])
        self.scaler.fit(filepath, imputer=self.imputer)

        ## scaler fitting takes too long!

        return self

    def transform(self, chunk):

        scale_cols  = self.scaler.cols

        # print(chunk.columns.to_list())
        # print("_______________")
        # print(scale_cols)
        chunk = self.imputer.transform(chunk)
        chunk = self.encoder.transform(chunk)

        chunk[scale_cols] = self.scaler.transform(chunk[scale_cols])
        return chunk

