import pandas as pd

class ChunkedImputer:
    def __init__(self, missing_threshold=0.5, chunksize=10000):
        self.missing_threshold = missing_threshold
        self.chunksize         = chunksize

        self.columns_to_drop_   = None
        self.columns_to_impute_ = None
        self.kept_columns       = None 
        self.columns_to_encode  = None

        self.num_cols_impute = []
        self.cat_cols_impute = []

        self.nem_cols           = []

        self.num_imputers     = {}
        self.cat_imputers     = {}
        self.total_rows_      = 0
        self.missing_counts_  = None



    def fit(self, filepath, to_drop, strategies=None):

        for chunk in pd.read_csv(filepath, chunksize=self.chunksize):
            chunk = chunk.drop(columns=to_drop)

            if self.missing_counts_ is None:
                self.nem_cols = sorted(chunk.select_dtypes(include='number').columns.tolist())
                self.missing_counts_  = chunk.isnull().sum()
            else:
                self.missing_counts_ += chunk.isnull().sum()

            self.total_rows_ += len(chunk)

        missing_ratio           = self.missing_counts_ / self.total_rows_

        self.kept_columns       = missing_ratio[missing_ratio < self.missing_threshold]
        self.columns_to_impute_ = self.kept_columns[(self.kept_columns > 0) & (self.kept_columns < self.missing_threshold)]
        self.columns_to_encode  = [c for c in self.kept_columns.index.tolist() if c not in self.nem_cols]

        self.columns_to_scale   = [c for c in self.kept_columns.index.tolist() if c in self.nem_cols]# update

        del missing_ratio

        if strategies is None:
            raise Exception("imputations strategy required.")

        sample = pd.read_csv(filepath, usecols=self.columns_to_impute_.index.tolist(), nrows=strategies['samplesize'])
        strategies.pop('samplesize', None)

        self.num_cols_impute = [c for c in self.columns_to_impute_.index.tolist() if c in self.columns_to_scale]
        self.cat_cols_impute = [c for c in self.columns_to_impute_.index.tolist() if c not in self.columns_to_scale]


        for index, (range, imputer) in enumerate(strategies['num'].items()):

            num_ipt_series = self.columns_to_impute_[self.num_cols_impute]

            cols = num_ipt_series[
                (num_ipt_series > range[0] / 100) & (num_ipt_series < range[1])
            ].index.tolist()

            if len(cols) > 0:
                imputer.fit(sample[cols])              
                self.num_imputers[index] = {'cols':cols, 'imputer':imputer}


        for index, (range, imputer) in enumerate(strategies['cat'].items()):
            cat_ipt_series = self.columns_to_impute_[self.cat_cols_impute]

            cols = cat_ipt_series[
                (cat_ipt_series > range[0]) & (cat_ipt_series < range[1])
            ].index.tolist()
     
            if len(cols) > 0:
                imputer.fit(sample[cols])
                self.cat_imputers[index] = {'cols':cols, 'imputer':imputer}

        print("imputation step passed")
        return self


    def transform(self, chunk):
        if chunk is None:
            return None

        for key, value in self.cat_imputers.items():
                imputer = value['imputer']
                cols    = value['cols']
                chunk[cols] = imputer.transform(chunk[cols])

        for key, value in self.num_imputers.items():
                imputer = value['imputer']
                cols    = value['cols']
                chunk[cols] = imputer.transform(chunk[cols])


        print("impute transfrom completed")
        return chunk


        








        


