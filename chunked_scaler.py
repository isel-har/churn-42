import pandas as pd
import numpy as np


class ChunkedScaler:
    def __init__(self, chunksize=10000):
        self.chunksize = chunksize
        self.mean_ = None
        self.std_ = None
        self.columns_ = None
        self.total_rows_ = 0


    def _apply_imputer(self, X, imputer):
        """Apply fitted imputers safely."""
        if imputer is None:
            return X

        for value in imputer.num_imputers.values():
            cols = value['cols']
            obj = value['imputer']
            X.loc[:, cols] = obj.transform(X[cols])

        return X

    def fit(self, filepath, imputer=None):

        cols = imputer.nem_cols if imputer else None

        total_sum = None
        total_rows = 0

        for chunk in pd.read_csv(filepath, usecols=cols, chunksize=self.chunksize):

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
        self.columns_ = self.mean_.index


        total_var = np.zeros(len(self.columns_))

        for chunk in pd.read_csv(filepath, usecols=self.columns_, chunksize=self.chunksize):

            X = chunk.copy()
            X = self._apply_imputer(X, imputer)
            X = X.astype(float)

            diff = X.values - self.mean_.values
            total_var += (diff ** 2).sum(axis=0)

        variance = total_var / self.total_rows_
        std__ = np.sqrt(variance)

        std__[std__ == 0] = 1.0

        self.std_ = pd.Series(std, index=self.columns_)

        return self

    def transform(self, filepath, output_path=None, imputer=None):
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("Scaler has not been fitted.")

        chunks = []

        for chunk in pd.read_csv(filepath, usecols=self.columns_, chunksize=self.chunksize):

            X = chunk.copy()
            X = self._apply_imputer(X, imputer)
            X = X.astype(float)

            X_scaled = (X - self.mean_) / self.std_

            if output_path:
                X_scaled.to_csv(
                    output_path,
                    mode="a",
                    header=not pd.io.common.file_exists(output_path),
                    index=False
                )
            else:
                chunks.append(X_scaled)

        if not output_path:
            return pd.concat(chunks, ignore_index=True)
