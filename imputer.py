import pandas as pd

class ImputerMeta:

    def __init__(self, cols=[], imputer=None):
        self.cols = cols
        self.imputer = imputer


class Imputer:

    def __init__(self, missing_threshold=0.5, chunksize=10_000):
        
        self.missing_threshold = missing_threshold
        self.chunksize         = chunksize

        self.columns      = None
        self.num_imputers = []
        self.cat_imputers = []
        self.num_cols     = []


    def fit(self, filepath, cols_to_drop=[], imputers=None, samplesize=10000):
        if imputers is None:
            raise Exception("imputers dict required.")

        missing_count = None
        total_rows    = 0

        for chunk in pd.read_csv(filepath, chunksize=self.chunksize):

            chunk = chunk.drop(columns=cols_to_drop)

            if missing_count is None:

                self.num_cols = chunk.select_dtypes(include='number').columns.tolist()
                missing_count = chunk.isnull().sum()  
            else:
                missing_count += chunk.isnull().sum()
            total_rows += len(chunk)

        missing_ratio = missing_count / total_rows

        del missing_count

        self.columns = missing_ratio[missing_ratio < self.missing_threshold]

        for range, imputer_ in imputers.items():

            cols = self.columns[
                (self.columns > range[0]) & (self.columns <= range[1])
            ].index.tolist()
            
            n_cols = [c for c in cols if c in self.num_cols]
            c_cols = [c for c in cols if c not in self.num_cols]

            if len(n_cols) > 0:

                m = ImputerMeta(cols=n_cols.copy(), imputer=imputer_[0])
                self.num_imputers.append(m)

            if len(c_cols) > 0:
                m = ImputerMeta(cols=c_cols.copy(), imputer=imputer_[1])
                self.cat_imputers.append(m)


        sample = pd.read_csv(filepath, nrows=samplesize, usecols=self.columns.index.to_list())

        for o in self.num_imputers:
            cols = o.cols
            o.imputer.fit(sample[cols])

        for o in self.cat_imputers:
            cols = o.cols
            o.imputer.fit(sample[cols])

        return self
    

    def transform(self, chunk):
        if chunk is None:
            return None

        chunk = chunk.copy()

        def _apply_imputers(imputers):
            
            for o in imputers:

                cols = o.cols
                transformed = o.imputer.transform(chunk[cols])
                
                if len(cols) == 1:
                    chunk[cols[0]] = transformed.ravel()
                else:
                    chunk[cols] = transformed

            # for key, value in imputer_dict.items():
            #     imputer = value.get("imputer")
            #     cols = value.get("cols", [])
            #     if not imputer or not cols:
            #         continue

            #     # Ensure columns exist
            #     missing_cols = [c for c in cols if c not in chunk.columns]
            #     if missing_cols:
            #         raise ValueError(f"Missing columns in chunk: {missing_cols}")


                # Handle single-column case (sklearn returns 2D array)

        _apply_imputers(self.cat_imputers)
        _apply_imputers(self.num_imputers)


        print("imputation transformed.")
        return chunk