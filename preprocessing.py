from column_transformer import ColumnTransformer
from pipeline import Pipeline
import pandas as pd


import sys

df = pd.read_csv(sys.argv[1], nrows=1000)


preprocessor = ColumnTransformer(transformers=[
    (''),
    (''),
    ('')
])

