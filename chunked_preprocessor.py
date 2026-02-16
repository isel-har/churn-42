import pandas as pd
import numpy as np
import imputers as ipt

pd.set_option('display.max_rows', None)

class ChunkedPreprocessor:

    CAT_NA_VALUE = "__MISSING__"

    def __init__(self, 
        missing_threshold=0.5,
        chunksize=10000):

        self.missing_threshold = missing_threshold
        self.chunksize         = chunksize

        self.columns_to_drop_   = None
        self.columns_to_impute_ = None
        self.kept_columns       = None 
        self.columns_to_encode  = None
        self.nem_cols           = []


        self.impute_values_   = {}
        self.total_rows_      = 0
        self.missing_counts_  = None
        self.sum_             = None
        self.count_           = None



    def fit(self, filepath, to_drop=[]):

        for chunk in pd.read_csv(filepath, chunksize=self.chunksize):
            chunk = chunk.drop(columns=to_drop)

            if self.missing_counts_ is None:
                self.nem_cols = chunk.select_dtypes(include='number').columns.tolist()
                self.missing_counts_  = chunk.isnull().sum()
                self.sum_ = chunk.sum(numeric_only=True)
                self.count_ = chunk.count(numeric_only=True)
            else:
                self.missing_counts_ += chunk.isnull().sum()
                self.sum_ += chunk.sum(numeric_only=True)
                self.count_ += chunk.count(numeric_only=True)

            self.total_rows_ += len(chunk)


        missing_ratio = self.missing_counts_ / self.total_rows_
        # del missing_ratio
        self.kept_columns       = missing_ratio[missing_ratio < self.missing_threshold]
        self.columns_to_impute_ = self.kept_columns[(missing_ratio > 0) & (missing_ratio < self.missing_threshold)]
        self.columns_to_encode  = [c for c in self.kept_columns.index.tolist() if c not in self.nem_cols]

        #Missing Ratio	Features	Strategy
        # ~10-15%	PRC & Count columns	Median (to handle outliers) or 0 (if no activity).
        # ~28%	ATM Tendency 3M	KNN/MICE + Missingness Indicator.
        # ~40-45%	Job Position	Constant "Unknown" (Categorical).

        # ##fit numerical columns missing data
        impute_num_cols = [c for c in self.columns_to_impute_.index.tolist() if c in self.nem_cols]
        for chunk in pd.read_csv(filepath, chunksize=self.chunksize, usecols=impute_num_cols):### only columns to impute
            ...



        return self



    def transform(self, filepath, output_path=None):
        ...
        # processed_chunks = []

        # for chunk in pd.read_csv(filepath, chunksize=self.chunksize):

        #     chunk = chunk.drop(columns=self.columns_to_drop_, errors="ignore")

        #     chunk = chunk.fillna(self.impute_values_)

        #     if output_path:
        #         chunk.to_csv(output_path, mode="a", index=False, header=False)
        #     else:
        #         processed_chunks.append(chunk)

        # if not output_path:
        #     return pd.concat(processed_chunks)
