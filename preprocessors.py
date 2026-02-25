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



# from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np


class MissingAwareColumnSelector:

    def __init__(self, missing_threshold=0.5, y_cols=[]):
        self.missing_threshold = missing_threshold
        self.y_cols = y_cols


    def fit(self, X, y=None):
        # X = pd.DataFrame(X)
        # X.drop(columns=to)
        num_cols = X.select_dtypes(include='number').columns.to_list()
        cat_cols = X.select_dtypes(exclude='number').columns.to_list()

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
            (self.missing_ratio_ <= 0.7)
        ].index.tolist()

        self.not_missing_ = self.missing_ratio_[
            self.missing_ratio_ == 0
        ].index.tolist()

        self.kept_num = [c for c in self.columns_to_keep_ if c in num_cols and c not in self.y_cols]
        self.kept_cat = [c for c in self.columns_to_keep_ if c in cat_cols and c not in self.y_cols]

        self.num_low = [c for c in self.low_missing_ if c in self.kept_num and c not in self.y_cols]
        self.num_mid = [c for c in self.mid_missing_ if c in self.kept_num and c not in self.y_cols]

        self.cat_low = [c for c in self.low_missing_ if c in self.kept_cat and c not in self.y_cols]
        self.cat_mid = [c for c in self.mid_missing_ if c in self.kept_cat and c not in self.y_cols]

        return self

    def transform(self, X):
        X = pd.DataFrame(X)
        return X[self.columns_to_keep_]