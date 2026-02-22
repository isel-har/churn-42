import pandas as pd

class EncoderMeta:
    def __init__(self, cols=[], encoder=None):
        self.cols = cols
        self.encoder = encoder


class Encoder:

    def __init__(self, chunksize=10_000):
        self.chunksize = chunksize
        self.unique_categories = {}
        self.encoders_ = []


    def fit(self, filepath, imputer=None, encoders=None):
        
        if imputer is None or encoders is None:
            raise Exception("imputer and encoders required.")

        self.cat_cols = [c for c in imputer.columns.index.tolist() if c not in imputer.num_cols]
        
        for col in self.cat_cols:
            self.unique_categories[col] = set()

        sample = None

        for chunk in pd.read_csv(filepath, chunksize=self.chunksize, usecols=self.cat_cols):

            for imputer_ in imputer.cat_imputers:

                cols = imputer_.cols
                transformed = imputer_.imputer.transform(chunk[cols])
                chunk[cols] = transformed
                # if len(cols) == 1:
                #     chunk[cols[0]] = transformed.ravel()
                # else:

            if sample is None:
                sample = chunk.head()

            for col in self.cat_cols:
                self.unique_categories[col].update(list(chunk[col].astype(str).unique()))



        for col, encoder in encoders.items():
            categories = sorted(self.unique_categories[col])
            encoder.set_params(categories=[categories])
            encoder.fit(chunk[[col]])
            em = EncoderMeta(cols=[col], encoder=encoder)
            self.encoders_.append(em)

        return self



    def transform(self, chunk):
        if chunk is None:
            return None


        for em in self.encoders_:
            cols    = em.cols
            encoded = em.encoder.transform(chunk[cols])

            if hasattr(encoded, "toarray"):
                encoded = encoded.toarray()
            new_cols = em.encoder.get_feature_names_out(cols)

            encoded_df = pd.DataFrame(
                encoded,
                columns=new_cols,
                index=chunk.index
            )
            for col in cols:
                chunk = chunk.drop(columns=[col])
                chunk = pd.concat([chunk, encoded_df], axis=1)


        print("encoding transformed")
        return chunk


