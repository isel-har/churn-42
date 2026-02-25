import sys
import pandas as pd
import numpy as np
import joblib

from sklearn.preprocessing import OrdinalEncoder, StandardScaler#, FunctionTransformer
from sklearn.impute import SimpleImputer



from column_transformer import ColumnTransformer
from pipeline import Pipeline

# from sklearn.compose import ColumnTransformer
# from sklearn.pipeline import Pipeline


def num_imputer_pipeline(selector, sample):

    num_imputers = []

    if selector.num_low:
        low_imputer = SimpleImputer(strategy='median')
        low_imputer.fit(sample[selector.num_low])
        num_imputers.append((selector.num_low, low_imputer))

    if selector.num_mid:
        mid_imputer = SimpleImputer(strategy='median')
        mid_imputer.fit(sample[selector.num_mid])
        num_imputers.append((selector.num_mid, mid_imputer))

    return ColumnTransformer(fit_transformers=num_imputers)


def cat_imputer_pipeline(selector, sample):

    cat_imputers = []

    if selector.cat_low:
        low_imputer = SimpleImputer(strategy='constant', fill_value='MISSING')
        low_imputer.fit(sample[selector.cat_low])
        cat_imputers.append((selector.cat_low, low_imputer))

    if selector.cat_mid:
        mid_imputer = SimpleImputer(strategy='constant', fill_value='MISSING')
        mid_imputer.fit(sample[selector.cat_mid])
        cat_imputers.append((selector.cat_mid, mid_imputer))

    return ColumnTransformer(fit_transformers=cat_imputers)


def encoder_pipeline(filepath, selector, cat_imputer, chunksize=50_000):

    unique_categories = {col: set() for col in selector.kept_cat}

    for chunk in pd.read_csv(filepath, usecols=selector.kept_cat, chunksize=chunksize):
        imputed_chunk = cat_imputer.transform(chunk)
        for col in selector.kept_cat:
            unique_categories[col].update(imputed_chunk[col].astype(str).unique())

    encoders = []
    for col in selector.kept_cat:
        categories_list = sorted(list(unique_categories[col]))
        encoder = OrdinalEncoder(
            categories=[categories_list],
            handle_unknown="use_encoded_value",
            unknown_value=-1
        )
        # Fit on dummy data to initialize
        encoder.fit(pd.DataFrame({col: categories_list})[[col]])
        encoders.append(([col], encoder))

    return ColumnTransformer(fit_transformers=encoders)


def scaler_pipeline(filepath, selector, num_imputer, chunksize=50_000):

    scaler = StandardScaler()
    for chunk in pd.read_csv(filepath, usecols=selector.kept_num, chunksize=chunksize):
        transformed = num_imputer.transform(chunk)
        scaler.partial_fit(transformed)
    return scaler


def build_final_preprocessor(num_imputer, scaler, cat_imputer, encoders, selector):

    # Pipelines
    num_pipeline = Pipeline([
        ('imputer', num_imputer),
        ('scaler', scaler)
    ])
    cat_pipeline = Pipeline([
        ('imputer', cat_imputer),
        ('encoder', encoders)
    ])
    # # Wrap in FunctionTransformer to skip fitting inner pipelines
    # num_wrapped = FunctionTransformer(func=num_pipeline.transform, validate=False)
    # cat_wrapped = FunctionTransformer(func=cat_pipeline.transform, validate=False)

    preprocessor = ColumnTransformer(
        fit_transformers=[
            (selector.kept_num, num_pipeline),
            (selector.kept_cat, cat_pipeline),
        ],
        # remainder='drop'
    )

    # -----------------------------
    # dummy_data = pd.DataFrame(
    #     {col: [0] for col in selector.kept_num} |
    #     {col: ["MISSING"] for col in selector.kept_cat}
    # )
    # preprocessor.fit(dummy_data)  # does not re-fit inner pipelines
    return preprocessor


def main():
    if len(sys.argv) != 2:
        print("Usage: python script.py data.csv")
        return

    try:
        selector = joblib.load("selector.pkl")
        sample = pd.read_csv(sys.argv[1], nrows=10_000)
        # sample = sample.drop(columns=['TARGET'])
        # Build imputers
        num_imputer = num_imputer_pipeline(selector, sample)
        cat_imputer = cat_imputer_pipeline(selector, sample)
        del sample

        # Build encoders
        encoders = encoder_pipeline(sys.argv[1], selector, cat_imputer)
        scaler   = scaler_pipeline(sys.argv[1], selector, num_imputer)
        preprocessor = build_final_preprocessor(num_imputer, scaler, cat_imputer, encoders, selector)


        joblib.dump(preprocessor, "preprocessor.pkl")
        print("Preprocessor saved successfully.")

    except Exception as e:
        print("Error:", str(e))


if __name__ == "__main__":
    main()