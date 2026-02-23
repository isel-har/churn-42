import sys

import pandas as pd
import numpy as np


from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

from preprocessors import MissingAwareColumnSelector

# pd.set_option('display.max_rows', None)

def build_preprocessor(X):

    selector = MissingAwareColumnSelector()
    selector.fit(X)

    num_cols = X.select_dtypes(include=np.number).columns
    cat_cols = X.select_dtypes(exclude=np.number).columns

    low_missing = selector.low_missing_
    mid_missing = selector.mid_missing_

    numeric_low = [c for c in low_missing if c in num_cols]
    numeric_mid = [c for c in mid_missing if c in num_cols]

    categorical_low = [c for c in low_missing if c in cat_cols]
    categorical_mid = [c for c in mid_missing if c in cat_cols]

    preprocessor = ColumnTransformer(

        transformers=[

            ("num_low",
             Pipeline([
                 ("imputer", SimpleImputer(strategy="median")),
                 ("scaler", StandardScaler())
             ]),
             numeric_low
            ),

            ("num_mid",
             Pipeline([
                 ("imputer", KNNImputer(n_neighbors=6)),
                 ("scaler", StandardScaler())
             ]),
             numeric_mid
            ),

            ("cat_low",
             Pipeline([
                 ("imputer", SimpleImputer(strategy="constant", fill_value="MISSING")),
                 ("encoder", OrdinalEncoder())
             ]),
             categorical_low
            ),

            ("cat_mid",
             Pipeline([
                 ("imputer", SimpleImputer(strategy="constant", fill_value="MISSING")),
                 ("encoder", OrdinalEncoder())
             ]),
             categorical_mid
            ),
        ],

        remainder="passthrough"
    )

    return preprocessor


def main():

    if len(sys.argv) != 2:
        return
    
    try:
        df = pd.read_csv(sys.argv[1], nrows=10000)


        X = df.drop(columns=["TARGET", "ID"])
        y = df["TARGET"]

        preprocessor = build_preprocessor(X)

        pipeline = Pipeline([
            ("selector", MissingAwareColumnSelector()),
            ("preprocessor", preprocessor),
        ])

        pipeline.fit(X, y)
        joblib.dump(pipeline, "pipline.pkl")


    except Exception as e:
        print(str(e))


if __name__ == "__main__":
    main()