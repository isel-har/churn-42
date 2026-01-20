import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder

CAT_NA_VALUE = "__MISSING__"

def main():

    sum_x   = None
    sum_x2  = None
    samples = 0
    unique_categories = {}

    # ---- infer schema ----
    df = pd.read_csv("data/bank_data_train.csv", nrows=10000)

    numerical_cols = df.select_dtypes(include="number").columns.to_list()
    numerical_cols = [c for c in numerical_cols if c not in ("ID", "TARGET")]

    categorical_cols = df.select_dtypes(exclude="number").columns.to_list()
    dtypes_dict = df.dtypes.to_dict()

    for col in categorical_cols:
        unique_categories[col] = set()

    del df

    # ================= FIRST PASS =================
    chunks = pd.read_csv(
        "data/bank_data_train.csv",
        dtype=dtypes_dict,
        chunksize=100_000
    )

    for chunk in chunks:

        # ---- categorical stats ----
        for col in categorical_cols:
            unique_categories[col].add(CAT_NA_VALUE)
            unique_categories[col].update(
                chunk[col].fillna(CAT_NA_VALUE).astype(str).unique() ###AYYY!!!
            )

        # ---- numerical stats ----
        chunk[numerical_cols] = chunk[numerical_cols].fillna(0.0)

        if sum_x is None:
            sum_x  = pd.Series(0.0, index=numerical_cols)
            sum_x2 = pd.Series(0.0, index=numerical_cols)

        samples += len(chunk)
        sum_x   += chunk[numerical_cols].sum(axis=0)
        sum_x2  += (chunk[numerical_cols] ** 2).sum(axis=0)

    # ---- encoder ----
    categories_list = [sorted(unique_categories[col]) for col in categorical_cols]

    encoder = OrdinalEncoder(
        categories=categories_list,
        handle_unknown="use_encoded_value",
        unknown_value=-1,
    )

    # 👇 FIT ONCE (dummy frame)
    encoder.fit(
    pd.DataFrame(
        {col: [CAT_NA_VALUE] for col in categorical_cols}
        )
    )


    # ---- numeric stats ----
    mean = sum_x / samples
    variance = (sum_x2 / samples) - (mean ** 2)
    std_dev = np.sqrt(variance)

    kept_num_cols = variance[variance > 1e-4].index.to_list()
    mean = mean[kept_num_cols]
    std_dev = std_dev[kept_num_cols]

    cols = categorical_cols + kept_num_cols + ["TARGET"]

    # ================= SECOND PASS =================



    chunks = pd.read_csv(
        "data/bank_data_train.csv",
        usecols=cols,
        dtype=dtypes_dict,
        chunksize=100_000
    )

    for chunk in chunks:

        # ---- categorical ----
        chunk[categorical_cols] = (
            chunk[categorical_cols]
            .fillna(CAT_NA_VALUE)
            .astype(str)
        )
        chunk[categorical_cols] = encoder.transform(chunk[categorical_cols])

        # ---- numerical ----
        chunk[kept_num_cols] = chunk[kept_num_cols].fillna(0.0)
        chunk[kept_num_cols] = (chunk[kept_num_cols] - mean) / std_dev

        


if __name__ == "__main__":
    main()
