import pandas as pd

class ChunkedImputer:
    def __init__(self, missing_threshold=0.5, chunksize=10000):
        self.missing_threshold = missing_threshold
        self.chunksize         = chunksize

        self.columns_to_drop_   = None
        self.columns_to_impute_ = None
        self.kept_columns       = None 
        self.columns_to_encode  = None

        self.num_cols_impute  = []
        self.cat_cols_impute  = []
        self.columns_to_scale = []
        self.nem_cols         = []

        self.num_imputers     = {}
        self.cat_imputers     = {}
        self.total_rows_      = 0
        self.missing_counts_  = None

    def fit(self, filepath, to_drop, strategies=None):

        if strategies is None:
            raise Exception("imputations strategy required.")

        # Reset state (important if re-fitting same instance)
        self.missing_counts_ = None
        self.total_rows_ = 0
        self.num_imputers = {}
        self.cat_imputers = {}

        # -------- PASS 1: compute missing ratios --------  
        for chunk in pd.read_csv(filepath, chunksize=self.chunksize):
            chunk = chunk.drop(columns=to_drop)

            if self.missing_counts_ is None:
                self.nem_cols = sorted(
                    chunk.select_dtypes(include='number').columns.tolist()
                )
                self.missing_counts_ = chunk.isnull().sum()
            else:
                self.missing_counts_ += chunk.isnull().sum()

            self.total_rows_ += len(chunk)

        missing_ratio = self.missing_counts_ / self.total_rows_

        self.kept_columns = missing_ratio[
            missing_ratio < self.missing_threshold
        ]

        self.columns_to_impute_ = self.kept_columns[
            (self.kept_columns > 0) &
            (self.kept_columns < self.missing_threshold)
        ]

        self.columns_to_encode = [
            c for c in self.kept_columns.index if c not in self.nem_cols
        ]

        self.columns_to_scale = [
            c for c in self.kept_columns.index if c in self.nem_cols
        ]

        self.num_cols_impute = [
            c for c in self.columns_to_impute_.index if c in self.columns_to_scale
        ]

        self.cat_cols_impute = [
            c for c in self.columns_to_impute_.index if c not in self.columns_to_scale
        ]

        # -------- PASS 2: build unbiased random sample --------
        samplesize = strategies.get("samplesize", 10000)

        full_sample = pd.read_csv(
            filepath,
            usecols=self.columns_to_impute_.index.tolist()
        )

        sample = full_sample.sample(
            n=min(samplesize, len(full_sample)),
            random_state=42
        )

        # -------- FIT NUMERICAL IMPUTERS --------
        num_series = self.columns_to_impute_[self.num_cols_impute]

        for index, (range_, imputer) in enumerate(strategies['num'].items()):

            cols = num_series[
                (num_series >= range_[0]) &
                (num_series <= range_[1])
            ].index.tolist()

            # Remove columns that are completely NaN in sample
            cols = [c for c in cols if not sample[c].isna().all()]

            if len(cols) == 0:
                continue

            imputer.fit(sample[cols])

            self.num_imputers[index] = {
                "cols": cols,
                "imputer": imputer
            }

        # -------- FIT CATEGORICAL IMPUTERS --------
        cat_series = self.columns_to_impute_[self.cat_cols_impute]

        for index, (range_, imputer) in enumerate(strategies['cat'].items()):

            cols = cat_series[
                (cat_series >= range_[0]) &
                (cat_series <= range_[1])
            ].index.tolist()

            cols = [c for c in cols if not sample[c].isna().all()]

            if len(cols) == 0:
                continue

            imputer.fit(sample[cols])

            self.cat_imputers[index] = {
                "cols": cols,
                "imputer": imputer
            }

        return self

    # def fit(self, filepath, to_drop, strategies=None):

    #     for chunk in pd.read_csv(filepath, chunksize=self.chunksize):
    #         chunk = chunk.drop(columns=to_drop)

    #         if self.missing_counts_ is None:
    #             self.nem_cols = sorted(chunk.select_dtypes(include='number').columns.tolist())
    #             self.missing_counts_  = chunk.isnull().sum()
    #         else:
    #             self.missing_counts_ += chunk.isnull().sum()

    #         self.total_rows_ += len(chunk)


    #     missing_ratio           = self.missing_counts_ / self.total_rows_
    #     self.kept_columns       = missing_ratio[missing_ratio < self.missing_threshold]


    #     self.columns_to_impute_ = self.kept_columns[(self.kept_columns > 0) & (self.kept_columns < self.missing_threshold)]

    #     self.columns_to_encode  = [c for c in self.kept_columns.index.tolist() if c not in self.nem_cols]

    #     self.columns_to_scale   = [c for c in self.kept_columns.index.tolist() if c in self.nem_cols]# update


    #     if strategies is None:
    #         raise Exception("imputations strategy required.")



    #     nrows=strategies['samplesize']
    #     full_sample = pd.read_csv(filepath, usecols=self.columns_to_impute_.index.tolist())
    #     sample = full_sample.sample(
    #         n=min(strategies['samplesize'], len(full_sample)),
    #         random_state=42
    #     )


    #     cols = [
    #         c for c in cols
    #         if not sample[c].isna().all()
    #     ]

    #     self.num_cols_impute = [c for c in self.columns_to_impute_.index.tolist() if c in self.columns_to_scale]
    #     self.cat_cols_impute = [c for c in self.columns_to_impute_.index.tolist() if c not in self.columns_to_scale]


    #     for index, (range, imputer) in enumerate(strategies['num'].items()):

    #         num_ipt_series = self.columns_to_impute_[self.num_cols_impute]


    #         cols = num_ipt_series[
    #             (num_ipt_series >= range[0]) & (num_ipt_series <= range[1])
    #         ].index.tolist()

    #         if len(cols) > 0:
    #             imputer.fit(sample[cols])              
    #             self.num_imputers[index] = {'cols':cols, 'imputer':imputer}


    #     for index, (range, imputer) in enumerate(strategies['cat'].items()):
    #         cat_ipt_series = self.columns_to_impute_[self.cat_cols_impute]
    #         cols = cat_ipt_series[
    #             (cat_ipt_series >= range[0]) & (cat_ipt_series <= range[1])
    #         ].index.tolist()
     
    #         if len(cols) > 0:
    #             imputer.fit(sample[cols])
    #             self.cat_imputers[index] = {'cols':cols, 'imputer':imputer}

    #     return self


    def transform(self, chunk):
        if chunk is None:
            return None

        # Work on a copy to avoid side effects
        chunk = chunk.copy()

        def _apply_imputers(imputer_dict):
            for key, value in imputer_dict.items():
                imputer = value.get("imputer")
                cols = value.get("cols", [])
                if not imputer or not cols:
                    continue

                # Ensure columns exist
                missing_cols = [c for c in cols if c not in chunk.columns]
                if missing_cols:
                    raise ValueError(f"Missing columns in chunk: {missing_cols}")

                transformed = imputer.transform(chunk[cols])

                # Handle single-column case (sklearn returns 2D array)
                if len(cols) == 1:
                    chunk[cols[0]] = transformed.ravel()
                else:
                    chunk[cols] = transformed

        _apply_imputers(self.cat_imputers)
        _apply_imputers(self.num_imputers)

        return chunk
            








        


