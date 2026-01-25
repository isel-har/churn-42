# from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import  OneHotEncoder
import pandas as pd
import numpy as np
import joblib as jb



# pd.set_option('display.max_columns', None) ## global options
pd.set_option('display.max_rows', None)
CAT_NA_VALUE = "__MISSING__"

def main():

    df = pd.read_csv("data/bank_data_train.csv", nrows=1000)
    categorical_cols = df.select_dtypes(exclude="number").columns.to_list()
    numerical_cols   = df.select_dtypes(include='number').columns.to_list()
    numerical_cols   = [c for c in numerical_cols if c not in ("ID", "TARGET")]
    dtypes_dict      = df.dtypes.to_dict()
    nan_counts = pd.Series(0.0, index=df.columns.to_list())

    del df
    samples = 0

    chunks = pd.read_csv(
        "data/bank_data_train.csv",
        dtype=dtypes_dict,
        chunksize=100_000
    )
    ## first pass
    for chunk in chunks:
        nan_counts += chunk.isna().sum()
        samples    += len(chunk)

    nan_counts    = (nan_counts / samples) * 100
    kep_cols       = nan_counts[nan_counts <= 40].index
    # print("NaN percentage of each feature")
    # print(nan_counts)


    kept_num_cols = [n for n in numerical_cols if n in kep_cols]
    kept_cat_cols = [n for n in categorical_cols if n in kep_cols]

    chunks = pd.read_csv(
        "data/bank_data_train.csv",
        dtype=dtypes_dict,
        chunksize=100_000
    )


    unique_categories = {}
    for cat in kept_cat_cols:
        unique_categories[cat] = set()


    sum_x   = pd.Series(0.0, index=kept_num_cols)
    sum_x2  = pd.Series(0.0, index=kept_num_cols)
    for chunk in chunks:
        # ---- categorical stats ----

        for col in kept_cat_cols:
            unique_categories[col].add(CAT_NA_VALUE)
            unique_categories[col].update(
                chunk[col].fillna(CAT_NA_VALUE).astype(str).unique()
            )
        # ---- numerical stats ----
        chunk[kept_num_cols] = chunk[kept_num_cols].fillna(0.0)

        sum_x   += chunk[kept_num_cols].sum(axis=0)
        sum_x2  += chunk[kept_num_cols].pow(2).sum(axis=0)

    # print("==============================sum of all kept numerical features==============================")
    categories_for_ohe = [
        sorted(list(unique_categories[cat]))
        for cat in kept_cat_cols
    ]
    encoder = OneHotEncoder(
        categories=categories_for_ohe,
        handle_unknown="ignore",
        sparse_output=False
    )


    df = pd.read_csv('data/bank_data_train.csv', usecols=kept_cat_cols, nrows=100)
    encoder.fit(df)
    del df

    mean     = sum_x / samples
    variance = (sum_x2 / samples) - (mean ** 2)
    std_dev  = np.sqrt(variance)

    # kept_num_cols = variance[variance > 1e-4].index.to_list()



    preprocessor = {
        "mean":mean.to_numpy(),
        "std" : std_dev,
        "encoder":encoder,
        "kept_num_cols":kept_num_cols,
        "kept_cat_cols":kept_cat_cols,
        'dtypes':dtypes_dict
    }

    jb.dump(preprocessor, "preprocessor.joblib")
if __name__ == "__main__":
    main()
