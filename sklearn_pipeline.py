
import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder, StandardScaler#, FunctionTransformer


from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

from preprocessors import MissingAwareColumnSelector
import sys

# pd.set_option('display.max_rows', None)


def build_preprocessor(selector):

    numeric_low = selector.num_low
    numeric_mid = selector.num_mid
    numeric_nm  = [c for c in selector.kept_num if c not in numeric_low and c not in numeric_mid]

    cat_low = selector.cat_low
    cat_mid = selector.cat_mid
    cat_nm  = [c for c in selector.kept_cat if c not in cat_low and c not in cat_mid]

    preprocessor = ColumnTransformer(
        transformers=[

            # ===== NUMERIC LOW MISSING =====
            ("num_low",
             Pipeline([
                 ("imputer", SimpleImputer(strategy="mean")),
                 ("scaler", StandardScaler())
             ]),
             numeric_low
            ),
            # ===== NUMERIC MID MISSING =====
            ("num_mid",
             Pipeline([
                 ("imputer", KNNImputer(n_neighbors=5)),
                 ("scaler", StandardScaler())
             ]),
             numeric_mid
            ),

            # ===== NUMERIC NO MISSING =====
            ("num_not_miss",
             StandardScaler(),
             numeric_nm
            ),

            # ===== CATEGORICAL LOW =====
            ("cat_low",
             Pipeline([
                 ("imputer", SimpleImputer(strategy="constant", fill_value="MISSING")),
                 ("encoder", OrdinalEncoder(
                     handle_unknown="use_encoded_value",
                     unknown_value=-1
                 ))
             ]),
             cat_low
            ),

            # ===== CATEGORICAL MID =====
            ("cat_mid",
             Pipeline([
                 ("imputer", SimpleImputer(strategy="constant", fill_value="MISSING")),
                 ("encoder", OrdinalEncoder(
                     handle_unknown="use_encoded_value",
                     unknown_value=-1
                 ))
             ]),
             cat_mid
            ),

            # ===== CATEGORICAL NO MISSING =====
            ("cat_not_miss",
             OrdinalEncoder(
                 handle_unknown="use_encoded_value",
                 unknown_value=-1
             ),
             cat_nm
            ),

        ],
        remainder="drop"
    )

    return preprocessor


def main():

    if len(sys.argv) != 2:
        return
    
    try:
        df = pd.read_csv(sys.argv[1])

        selector = joblib.load("selector.pkl")

        column_transformer = build_preprocessor(selector)

        column_transformer.fit(df)
        joblib.dump(column_transformer, "column_transformer.pkl")

    except Exception as e:
        print("error:", str(e))


if __name__ == "__main__":
    main()