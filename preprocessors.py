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


# class Preprocessor(BaseEstimator, TransformerMixin):

#     def __init__(self, missing_threshold=0.5):
#         self.missing_threshold = missing_threshold
#         self.scaler = StandardScaler()

#         # Learned attributes
#         self.num_imputers = []
#         self.cat_imputers = []
#         self.encoders = {}
#         self.num_cols = []
#         self.cat_cols = []
#         self.columns_to_drop = []
#         self.fitted_columns = None


#     def fit(self, df, strategies):

#         if df is None or strategies is None:
#             raise ValueError("DataFrame and strategies are required.")

#         df = df.copy()
#         self.fitted_columns = df.columns

#         missing_ratio = df.isnull().mean()
#         self.columns_to_drop = missing_ratio[
#             missing_ratio >= self.missing_threshold
#         ].index.tolist()

#         df = df.drop(columns=self.columns_to_drop)

#         self.num_cols = df.select_dtypes(include=np.number).columns.tolist()
#         self.cat_cols = [c for c in df.columns if c not in self.num_cols]


#         for missing_range, imputer_pair in strategies['imputers'].items():

#             cols = missing_ratio[
#                 (missing_ratio > missing_range[0]) &
#                 (missing_ratio <= missing_range[1])
#             ].index.tolist()

#             cols = [c for c in cols if c not in self.columns_to_drop]

#             n_cols = [c for c in cols if c in self.num_cols]
#             c_cols = [c for c in cols if c in self.cat_cols]

#             if n_cols:
#                 num_imputer = imputer_pair[0]
#                 num_imputer.fit(df[n_cols])
#                 self.num_imputers.append((n_cols, num_imputer))

#             if c_cols:
#                 cat_imputer = imputer_pair[1]
#                 cat_imputer.fit(df[c_cols])
#                 self.cat_imputers.append((c_cols, cat_imputer))


#         for column, encoder in strategies['encoders'].items():
#             if column in df.columns:
#                 encoder.fit(df[[column]])
#                 self.encoders[column] = encoder

#         temp_df = self._apply_imputation(df.copy())
#         self.scaler.fit(temp_df[self.num_cols])

#         return self


#     def transform(self, df):

#         df = df.copy()

#         # Ensure same columns
#         df = df[self.fitted_columns]

#         # Drop same columns as during fit
#         df = df.drop(columns=self.columns_to_drop)

#         # Impute
#         df = self._apply_imputation(df)

#         # Encode
#         for column, encoder in self.encoders.items():
#             df[column] = encoder.transform(df[[column]])

#         # Scale numeric
#         df[self.num_cols] = self.scaler.transform(df[self.num_cols])

#         return df


#     def fit_transform(self, df, strategies):
#         return self.fit(df, strategies).transform(df)


#     def _apply_imputation(self, df):

#         for cols, imputer in self.num_imputers:
#             df[cols] = imputer.transform(df[cols])

#         for cols, imputer in self.cat_imputers:
#             df[cols] = imputer.transform(df[cols])

#         return df

from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np


class MissingAwareColumnSelector(BaseEstimator, TransformerMixin):

    def __init__(self, missing_threshold=0.5):
        self.missing_threshold = missing_threshold

    def fit(self, X, y=None):
        X = pd.DataFrame(X)

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

        return self

    def transform(self, X):
        X = pd.DataFrame(X)
        return X[self.columns_to_keep_]