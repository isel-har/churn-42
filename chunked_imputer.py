import pandas as pd

class ChunkedImputer:
    def __init__(self, missing_threshold=0.5, chunksize=10000):
        self.missing_threshold = missing_threshold
        self.chunksize         = chunksize

        self.columns_to_drop_   = None
        self.columns_to_impute_ = None
        self.kept_columns       = None 
        self.columns_to_encode  = None
        self.nem_cols           = []

        self.num_imputers     = {}
        self.cat_imputers     = {}
        self.total_rows_      = 0
        self.missing_counts_  = None
        self.sum_             = None
        self.count_           = None

    def fit(self, filepath, to_drop, imputations_=None):

        for chunk in pd.read_csv(filepath, chunksize=self.chunksize):
            chunk = chunk.drop(columns=to_drop)

            if self.missing_counts_ is None:
                self.nem_cols = chunk.select_dtypes(include='number').columns.tolist()
                self.missing_counts_  = chunk.isnull().sum()
            else:
                self.missing_counts_ += chunk.isnull().sum()
            self.total_rows_ += len(chunk)


        missing_ratio = self.missing_counts_ / self.total_rows_
        self.kept_columns       = missing_ratio[missing_ratio < self.missing_threshold]
        self.columns_to_impute_ = self.kept_columns[(missing_ratio > 0) & (missing_ratio < self.missing_threshold)]
        self.columns_to_encode  = [c for c in self.kept_columns.index.tolist() if c not in self.nem_cols]
        del missing_ratio

        """
            imputation section
        """
        if imputations_ is None:
            raise Exception("imputations strategy required.")

        sample = pd.read_csv(filepath, usecols=self.kept_columns.index.tolist(), nrows=imputations_['samplesize'])
        imputations_.pop('samplesize', None)

        num_cols_impute = [c for c in self.columns_to_impute_.index.tolist() if c in self.nem_cols]
        cat_cols_impute = [c for c in self.columns_to_impute_.index.tolist() if c not in self.nem_cols]

        for index, (range, imputer) in enumerate(imputations_['num'].items()):
            num_ipt_series = self.columns_to_impute_[num_cols_impute]

            cols = num_ipt_series[
                (num_ipt_series > range[0]) & (num_ipt_series < range[1])
            ].index.tolist()

            if len(cols) > 0:
                imputer.fit(sample[cols])              
                self.num_imputers[index] = {'cols':cols, 'imputer':imputer}

        for index, (range, imputer) in enumerate(imputations_['cat'].items()):
            cat_ipt_series = self.columns_to_impute_[cat_cols_impute]

            cols = cat_ipt_series[
                (cat_ipt_series > range[0]) & (cat_ipt_series < range[1])
            ].index.tolist()

            if len(cols) > 0:
                imputer.fit(sample[cols])              
                self.num_imputers[index] = {'cols':cols, 'imputer':imputer}
        return self

    def tranform(self, filepath, output_path=None):
        ...