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

        self.cols = [c for c in imputer.columns.index.tolist()
                    if c in imputer.num_cols]

        n_total = 0
        mean = np.zeros(len(self.cols))
        M2 = np.zeros(len(self.cols))  # sum of squared deviations

        for chunk in pd.read_csv(filepath,
                                usecols=self.cols,
                                chunksize=self.chunksize):

            X = self._apply_imputer(chunk, imputer)

            # Avoid unnecessary copy
            values = X[self.cols].to_numpy(dtype=float, copy=False)

            for row in values:
                n_total += 1
                delta = row - mean
                mean += delta / n_total
                delta2 = row - mean
                M2 += delta * delta2

        variance = M2 / n_total
        std = np.sqrt(variance)
        std[std == 0] = 1.0

        self.mean_ = pd.Series(mean, index=self.cols)
        self.std_ = pd.Series(std, index=self.cols)
        self.total_rows_ = n_total

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
