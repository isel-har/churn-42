import pandas as pd


class ColumnTransformer:
    def __init__(self, transformers=None):
        self.transformers = transformers if transformers else []
        # Store fitted transformers in a dict for easy access
        self.fitted_transformers = {}

    def validate_transformers(self):
        for t in self.transformers:
            if len(t) != 3:
                return False
            name, obj, cols = t
            # Corrected isinstance order: (instance, type)
            if not isinstance(name, str):
                return False
            # Check for fit/transform methods rather than a specific class
            if not (hasattr(obj, "fit") and hasattr(obj, "transform")):
                return False
            if not isinstance(cols, list):
                return False
        return True

    def fit(self, df: pd.DataFrame):
        if not self.validate_transformers():
            raise Exception('Each step must be a tuple of (name, transformer_object, columns_list)')
        
        for name, transformer, cols in self.transformers:
            # Fit only on the subset of columns
            transformer.fit(df[cols])
            self.fitted_transformers[name] = transformer
    
        return self

    def transform(self, df: pd.DataFrame):
        # Work on a copy to avoid modifying the original input dataframe
        output_df = df.copy()
        
        for name, transformer, cols in self.transformers:
            # Use the fitted transformer from our dictionary
            fitted_obj = self.fitted_transformers[name]
            transformed_data = fitted_obj.transform(df[cols])
            
            # If the transformer returns a DataFrame, we can join/update
            if isinstance(transformed_data, pd.DataFrame):
                # Drop old columns and join new ones if column names changed
                output_df = output_df.drop(columns=cols)
                output_df = pd.concat([output_df, transformed_data], axis=1)
            else:
                # If it's a simple array (like a Scaler), overwrite the columns
                output_df[cols] = transformed_data
    
        return output_df