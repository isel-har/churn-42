# from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder#OneHotEncoder
import pandas as pd
import numpy as np
import lightgbm as lgb


from sklearn.feature_selection import chi2
from sklearn.feature_selection import mutual_info_classif
# from sklearn.experimental import enable_iterative_imputer
# from sklearn.impute import SimpleImputer
# from sklearn.impute import IterativeImputer

# from lightgbm import LGBMRegressor, LGBMClassifier
# import joblib as jb

# pd.set_option('display.max_columns', None) ## global options
pd.set_option('display.max_rows', None)

CAT_NA_VALUE     = "__MISSING__"
MISSING_TRESHOLD = 70


# def impute_with_lgb(df, target_col, numeric=True):
#     # Split data
#     train_data   = df[df[target_col].notna()]
#     predict_data = df[df[target_col].isna()]
    
#     if predict_data.empty:
#         return df[target_col]

#     X_train = train_data.drop(columns=[target_col])
#     y_train = train_data[target_col]
#     X_test  = predict_data.drop(columns=[target_col])

#     # Choose objective based on column type
#     params = {
#         'objective': 'regression' if numeric else 'multiclass',
#         'verbosity': -1,
#         'num_class': len(y_train.unique()) if not numeric else 1
#     }

#     # Train model
#     dtrain = lgb.Dataset(X_train, label=y_train)
#     model  = lgb.train(params, dtrain, num_boost_round=100)

#     # Predict and fill
#     predictions = model.predict(X_test)
#     df.loc[df[target_col].isna(), target_col] = predictions
#     return df[target_col]



def main():
    dtypes     = None
    nan_counts = None
    samples    = 0
    columns    = 0
    class_sum  = 0

    num_cols = []
    cat_cols = []

    chunks  = pd.read_csv(
        "data/bank_data_train.csv",
        chunksize=100_000
    )

    for chunk in chunks:
        
        if columns == 0:
            nan_counts = pd.Series(0.0, index=chunk.columns.to_list())
            columns    = len(chunk.columns)
            dtypes     = chunk.dtypes

            num_cols = chunk.select_dtypes(include='number').columns.to_list()
            cat_cols = chunk.select_dtypes(exclude='number').columns.to_list()



        samples    += len(chunk)
        class_sum  += len(chunk[chunk['TARGET'] == 1])
        nan_counts += chunk.isna().sum()

    print(f"dataset shape : {samples, columns}")
    print("columns type :\n", dtypes)
    print(f"class value 1 : {class_sum}")
    print(f"class value 0 : {samples - class_sum}")

    nan_counts = (nan_counts / samples) * 100
    print("Missing values percentage for each column:\n", nan_counts)
    # [✓] Feature distributions
    # [✓] Correlations

    # preprocessing
        # [✓] Handle missing values
        # [✓] Handle anomalies/outliers
        # [✓] Feature engineering/generation
        # [✓] Feature selection
        # [✓] Encode categorical variables
        # [✓] Scale/normalize features (essential for NN!)
        # [✓] Handle class imbalance (if needed)
    ## handling
    ## filter columns using missing threshold
    kept_cols = nan_counts[nan_counts < MISSING_TRESHOLD]
    kept_cols.drop(['ID', 'TARGET'], inplace=True)

    # kept_num_cols = kept_cols[ kept_cols.index.map(lambda x: x in num_cols) ]
    kept_cat_cols = kept_cols[ kept_cols.index.map(lambda x: x in cat_cols) ]

    # sum_impute = pd.Series(0.0, index=kept_num_cols.index)
    # total_impute = pd.Series(0.0, index=kept_num_cols.index)

    unique_categories = {}
    for cat in kept_cat_cols.index.to_list():
        unique_categories[cat] = set()


    chunks  = pd.read_csv(
        "data/bank_data_train.csv",
        chunksize=100_000,
        dtype=dtypes.to_dict(),
        usecols=kept_cols.index.to_list()
    )


    for chunk in chunks:

        # sum_impute   += chunk[kept_num_cols.index.to_list()].notna().sum()
        # total_impute += chunk[kept_num_cols.index.to_list()].notna().count()

        for col in kept_cat_cols.index.to_list():
            unique_categories[col].add(CAT_NA_VALUE)
            unique_categories[col].update(
                chunk[col].fillna(CAT_NA_VALUE).astype(str).unique()
            )

    # mean_impute = sum_impute / total_impute

    categories_for_e= [
        sorted(list(unique_categories[cat]))
        for cat in kept_cat_cols.index.to_list()
    ]

    # Change this line:
    encoder = OrdinalEncoder(
        categories=categories_for_e, 
        handle_unknown="use_encoded_value", # Fixed parameter
        unknown_value=-1,                   # Required when using 'use_encoded_value'
        encoded_missing_value=-1
    )
    df = pd.read_csv('data/bank_data_train.csv', usecols=kept_cat_cols.index.to_list(), nrows=100)
    encoder.fit(df)
    del df

    train_data = None
    chunks  = pd.read_csv(
        "data/bank_data_train.csv",
        chunksize=100_000,
        dtype=dtypes.to_dict(),
        usecols=kept_cols.index.to_list()
    )
    # use only one chunk to train lgm model!
    print("hana!")
    for chunk in chunks:
        clean_chunk = chunk[chunk[target_col].notna()]
        if not clean_chunk.empty:
            train_data = clean_chunk
            break
        # if not clean_chunk.empty:
        #     train_data = clean_chunk
        #     break


    




    # categorical_cols = df.select_dtypes(exclude="number").columns.to_list()
    # numerical_cols   = df.select_dtypes(include='number').columns.to_list()
    # numerical_cols   = [c for c in numerical_cols if c not in ("ID", "TARGET")]
    # dtypes_dict      = df.dtypes.to_dict()
    # nan_counts       = pd.Series(0.0, index=df.columns.to_list())

    # del df
    # samples = 0

    # chunks = pd.read_csv(
    #     "data/bank_data_train.csv",
    #     dtype=dtypes_dict,
    #     chunksize=100_000
    # )
    # ## first pass
    # for chunk in chunks:
    #     nan_counts += chunk.isna().sum()
    #     samples    += len(chunk)

    # nan_counts    = (nan_counts / samples) * 100
    # kep_cols       = nan_counts[nan_counts <= 40].index
    # # print("NaN percentage of each feature")
    # # print(nan_counts)


    # kept_num_cols = [n for n in numerical_cols if n in kep_cols]
    # kept_cat_cols = [n for n in categorical_cols if n in kep_cols]

    # chunks = pd.read_csv(
    #     "data/bank_data_train.csv",
    #     dtype=dtypes_dict,
    #     chunksize=100_000
    # )


    # unique_categories = {}
    # for cat in kept_cat_cols:
    #     unique_categories[cat] = set()


    # sum_x   = pd.Series(0.0, index=kept_num_cols)
    # sum_x2  = pd.Series(0.0, index=kept_num_cols)
    # for chunk in chunks:
    #     # ---- categorical stats ----

    #     for col in kept_cat_cols:
    #         unique_categories[col].add(CAT_NA_VALUE)
    #         unique_categories[col].update(
    #             chunk[col].fillna(CAT_NA_VALUE).astype(str).unique()
    #         )
    #     # ---- numerical stats ----
    #     chunk[kept_num_cols] = chunk[kept_num_cols].fillna(0.0)

    #     sum_x   += chunk[kept_num_cols].sum(axis=0)
    #     sum_x2  += chunk[kept_num_cols].pow(2).sum(axis=0)

    # # print("==============================sum of all kept numerical features==============================")
    # categories_for_ohe = [
    #     sorted(list(unique_categories[cat]))
    #     for cat in kept_cat_cols
    # ]
    # encoder = OneHotEncoder(
    #     categories=categories_for_ohe,
    #     handle_unknown="ignore",
    #     sparse_output=False
    # )


    # df = pd.read_csv('data/bank_data_train.csv', usecols=kept_cat_cols, nrows=100)
    # encoder.fit(df)
    # del df


    # mean     = sum_x / samples
    # variance = (sum_x2 / samples) - (mean ** 2)
    # std_dev  = np.sqrt(variance)

    # # kept_num_cols = variance[variance > 1e-4].index.to_list()

    # x_shape = len(encoder.get_feature_names_out(kept_cat_cols)) + len(kept_num_cols)

    # preprocessor = {
    #     "mean":mean.to_numpy(),
    #     "std" : std_dev,
    #     "encoder":encoder,
    #     "kept_num_cols":kept_num_cols,
    #     "kept_cat_cols":kept_cat_cols,
    #     'dtypes':dtypes_dict,
    #     'x_shape':x_shape
    # }

    # jb.dump(preprocessor, "preprocessor.joblib")
    # print("preprocessor params saved.")
if __name__ == "__main__":
    main()
