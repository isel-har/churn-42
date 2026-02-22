import pandas as pd
import numpy as np


class Scaler:
    def __init__(self, chunksize=10000):
        self.chunksize = chunksize
        self.mean_ = None
        self.std_ = None
        self.columns_ = None
        self.total_rows_ = 0
        self.cols = []

    def _apply_imputer(self, X, imputer):

        if imputer is None:
            return X

        for imputer_ in imputer.num_imputers:
            cols = imputer_.cols
            obj  = imputer_.imputer
            X.loc[:, cols] = obj.transform(X[cols])

        return X



    def fit(self, filepath, imputer=None):

        self.cols = [c for c in imputer.columns.index.tolist() if c in imputer.num_cols]
        total_sum = None
        total_rows = 0

        for chunk in pd.read_csv(filepath, usecols=self.cols, chunksize=self.chunksize):

            X = chunk.copy()
            X = self._apply_imputer(X, imputer)
            X = X.astype(float)

            if total_sum is None:
                total_sum = X.sum(axis=0)
            else:
                total_sum += X.sum(axis=0)

            total_rows += len(X)

        self.mean_ = total_sum / total_rows
        self.total_rows_ = total_rows


        total_var = np.zeros(len(self.cols))

        for chunk in pd.read_csv(filepath, usecols=self.cols, chunksize=self.chunksize):
            X = chunk.copy()
            X = self._apply_imputer(X, imputer)
            X = X.astype(float)

            diff = X.values - self.mean_.values
            total_var += (diff ** 2).sum(axis=0)

        variance = total_var / self.total_rows_
        std__ = np.sqrt(variance)

        std__[std__ == 0] = 1.0

        self.std_ = pd.Series(std__, index=self.cols)
        return self

    def transform(self, chunk):
        if chunk is None:
            return None

        if self.mean_ is None or self.std_ is None:
            raise ValueError("The scaler has not been fitted yet.")

        chunk = chunk.copy()

        # Align columns explicitly
        chunk = chunk[self.mean_.index]

        return (chunk - self.mean_) / self.std_
