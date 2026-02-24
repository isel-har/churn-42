# from scaler import Scaler
# from encoder import Encoder
# from imputer import Imputer, ImputerMeta
# from sklearn.preprocessing import StandardScaler


# class ChunkedPreprocessor:

#     def __init__(self, missing_threshold=0.5, chunksize=10000):

#         self.imputer = Imputer(missing_threshold, chunksize)
#         self.encoder = Encoder(chunksize)
#         self.scaler  = Scaler(chunksize=10000)


#     def fit(self, filepath, to_drop=[], strategies=None, sample_size=10_000):
#         if not strategies:
#             raise Exception("strategies dict required.")


#         self.imputer.fit(filepath, to_drop, strategies['imputers'], sample_size)
#         self.encoder.fit(filepath, imputer=self.imputer, encoders=strategies['encoders'])
#         self.scaler.fit(filepath, imputer=self.imputer)

#         ## scaler fitting takes too long!

#         return self

#     def transform(self, chunk):

#         scale_cols  = self.scaler.cols

#         chunk = self.imputer.transform(chunk)
#         chunk = self.encoder.transform(chunk)

#         chunk[scale_cols] = self.scaler.transform(chunk[scale_cols])
#         return chunk



from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np


class MissingAwareColumnSelector(BaseEstimator, TransformerMixin):

    def __init__(self, missing_threshold=0.5):
        self.missing_threshold = missing_threshold

    def fit(self, X, y=None):
        # X = pd.DataFrame(X)

        self.missing_ratio_ = X.isnull().mean()

        self.columns_to_keep_ = self.missing_ratio_[
            self.missing_ratio_ < self.missing_threshold
        ].index.tolist()

        self.low_missing_ = self.missing_ratio_[
            (self.missing_ratio_ > 0) &
            (self.missing_ratio_ <= 0.25)
        ].index.tolist()

        self.mid_missing_ = self.missing_ratio_[
            (self.missing_ratio_ > 0.25) &
            (self.missing_ratio_ <= 0.5)
        ].index.tolist()

        self.not_missing_ = self.missing_ratio_[
            self.missing_ratio_ == 0
        ].index.tolist()


        return self

    def transform(self, X):
        X = pd.DataFrame(X)
        return X[self.columns_to_keep_]