import pandas as pd

class Pipeline:
    def __init__(self, steps=[]):
        self.steps = steps

    def fit(self, df: pd.DataFrame):

        transformed = df

        for name, transformer in self.steps:
            transformer.fit(transformed)
            transformed = transformer.transform(transformed)

        return self

    def transform(self, df: pd.DataFrame):
    
        transformed = df
        for name, transformer in self.steps:
            transformed = transformer.transform(transformed)

        return transformed
