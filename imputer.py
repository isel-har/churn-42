from sklearn.experimental import enable_iterative_imputer  # Necessary
from sklearn.impute import IterativeImputer
import pandas as pd



def iterative_imputer(filepath, cols, chunksize=50_000):

    iter_imputer = IterativeImputer(max_iter=10, random_state=0)
    for chunk in pd.read_csv(filepath, usecols=cols, chunksize=chunksize):
        iter_imputer.fit(chunk)
    
    return iter_imputer


