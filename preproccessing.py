from sklearn.preprocessing import OrdinalEncoder, LabelBinarizer, OneHotEncoder
import pandas as pd
import numpy as np
import joblib as jb
# pd.set_option('display.max_columns', None) ## global options
# pd.set_option('display.max_rows', None) ## global options
CAT_NA_VALUE = "__MISSING__"

def main():
    # ---- infer schema ----
    df = pd.read_csv("data/bank_data_train.csv", nrows=1000)
    categorical_cols = df.select_dtypes(exclude="number").columns.to_list()
    numerical_cols   = df.select_dtypes(include='number').columns.to_list()
    numerical_cols   = [c for c in numerical_cols if c not in ("ID", "TARGET")]
    dtypes_dict      = df.dtypes.to_dict()

    encode_cols = {'orndinal':['CLNT_JOB_POSITION_TYPE', 'APP_EDUCATION', 'APP_POSITION_TYPE'],
                   'binary':['CLNT_JOB_POSITION']}
    encode_cols['nominal'] = [col for col in categorical_cols if col not in encode_cols['orndinal']]

    unique_categories = {}
    for key, cols in encode_cols.items():
        unique_categories[key] = {}
        for c in cols:
            unique_categories[key][c] = set()

    del df

    sum_x   = np.zeros(len(numerical_cols))
    sum_x2  = np.zeros(len(numerical_cols))
    samples = 0

    # ## ================= FIRST PASS =================
    chunks = pd.read_csv(
        "data/bank_data_train.csv",
        dtype=dtypes_dict,
        chunksize=100_000
    )

    for chunk in chunks:
        # ---- categorical stats ----
        for key, cols in unique_categories.items():

            for col in cols:
                unique_categories[key][col].add(CAT_NA_VALUE)
                unique_categories[key][col].update(
                    chunk[col].fillna(CAT_NA_VALUE).astype(str).unique()
                )
        # ---- numerical stats ----
        # for col in numerical_cols:
        chunk[numerical_cols] = chunk[numerical_cols].fillna(0.0)
        samples += len(chunk)
        sum_x   += chunk[numerical_cols].to_numpy().sum(axis=0)
        sum_x2  += chunk[numerical_cols].pow(2).sum().to_numpy()


    # ---- encoders ----
    categories = []
    for col in unique_categories['nominal']:
        if col != "CLNT_JOB_POSITION":
            categorical_cols.append(list(unique_categories['nominal'][col]))

    onehot_encoder = OneHotEncoder(categories=categories)
    binary_encoder = LabelBinarizer(categories=unique_categories['binary']['CLNT_JOB_POSITION'])
    # ordinal_encoder = OrdinalEncoder(categories)

    # # ---- numeric stats ----
    mean = sum_x / samples
    variance = (sum_x2 / samples) - (mean ** 2)
    std_dev = np.sqrt(variance)

    kept_num_cols = variance[variance > 1e-4].index.to_list()
    mean = mean[kept_num_cols]
    std_dev = std_dev[kept_num_cols]

    preproccessor = {
        'scaler':{
            'mean': mean,
            'std': std_dev
        },
        'encoders':{

            'norminal': {'cols':[], 'encoder':None}
        }
    }
    jb.dump(preproccessor, "preproccessor.")


if __name__ == "__main__":
    main()
