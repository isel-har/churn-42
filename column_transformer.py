import pandas as pd


class ColumnTransformer:
    def __init__(self, fit_transformers=[]):
        self.transformers = fit_transformers


    def transform(self, df: pd.DataFrame):
        transformed = df.copy()

        for cols, transformer in self.transformers:

            transformed[cols] = transformer.transform(transformed[cols])

            # If same number of columns → replace
            # if transformed_data.shape[1] == len(cols):
            #     output_df[cols] = transformed_data.values
            # else:
            #     # Drop old columns
            #     output_df = output_df.drop(columns=cols)

            #     # Generate new column names
            #     new_cols = [
            #         f"{name}_{i}" for i in range(transformed_data.shape[1])
            #     ]
            #     transformed_data.columns = new_cols

            #     # Concat
            #     output_df = pd.concat([output_df, transformed_data], axis=1)

        return transformed