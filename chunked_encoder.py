import pandas as pd

class ChunkedEncoder:

    def __init__(self, chunksize=10000):
        self.chunksize = chunksize
        self.unique_categories = {}
        self.encoders_map = {}

    def fit(self, filepath, imputer=None, strategies=None):

        cols = imputer.columns_to_encode

        for col in cols:
            self.unique_categories[col] = set()

        for chunk in pd.read_csv(filepath, chunksize=self.chunksize, usecols=cols):

            X = chunk
            for key, value in imputer.cat_imputers.items():
                obj = value['imputer']
                i_cols  = value['cols']
                X[i_cols] = obj.transform(X[i_cols])

            for col in cols:
                self.unique_categories[col].update(list(X[col].astype(str).unique()))

        sample = pd.read_csv(filepath, nrows=100, usecols=cols)
        for key, value in imputer.cat_imputers.items():
                obj = value['imputer']
                i_cols  = value['cols']
                sample[i_cols] = obj.transform(sample[i_cols])


        for col, encoder in strategies:
            categories = sorted(self.unique_categories[col])

            encoder.set_params(categories=[categories])
            encoder.fit(sample[[col]])
            self.encoders_map[col] = encoder

        del sample
        
        print("encoding step passed")

        return self


    def transform(self, chunk):
    
        if chunk is None:
            return None


        cols = chunk.columns.to_list()


        for col in cols:
            encoder = self.encoders_map[col]
            encoded = encoder.transform(chunk[[col]])
            
            
            # If sparse, convert
            if hasattr(encoded, "toarray"):
                encoded = encoded.toarray()
            new_cols = encoder.get_feature_names_out([col])
            
            # Create column names
            
            encoded_df = pd.DataFrame(
                encoded,
                columns=new_cols,
                index=chunk.index
            )
            
            # Drop original column
            chunk = chunk.drop(columns=[col])
            
            # Add encoded columns
            chunk = pd.concat([chunk, encoded_df], axis=1)

        print("encode transform completed")
        return chunk