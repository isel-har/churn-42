# from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder#OneHotEncoder
import pandas as pd
import numpy as np
# import lightgbm as lgb

from scipy.stats import chi2_contingency
# from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import roc_auc_score
# from sklearn.experimental import enable_iterative_imputer
# from sklearn.impute import SimpleImputer
# from sklearn.impute import IterativeImputer

# from lightgbm import LGBMRegressor, LGBMClassifier
# import joblib as jb

# pd.set_option('display.max_columns', None) ## global options
pd.set_option('display.max_rows', None)

CAT_NA_VALUE     = "__MISSING__"
MISSING_TRESHOLD = 70

def cramers_v(chi2, n, k):
    return np.sqrt(chi2 / (n * (k - 1)))

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
    # print("columns type :\n", dtypes)
    print(f"class value 1 : {class_sum}")
    print(f"class value 0 : {samples - class_sum}")


    nan_counts = (nan_counts / samples) * 100
    kept_cols = nan_counts[nan_counts < MISSING_TRESHOLD]
    kept_cols.drop(['ID'], inplace=True)

    kept_num_cols = kept_cols[ kept_cols.index.map(lambda x: x in num_cols) ]
    kept_cat_cols = kept_cols[ kept_cols.index.map(lambda x: x in cat_cols) ]

    # unique_categories = {}
    # for cat in kept_cat_cols.index.to_list():
    #     unique_categories[cat] = set()

    # chunks  = pd.read_csv(
    #     "data/bank_data_train.csv",
    #     chunksize=100_000,
    #     dtype=dtypes.to_dict(),
    #     usecols=kept_cols.index.to_list()
    # )

    # for chunk in chunks:

    #     for col in kept_cat_cols.index.to_list():
    #         unique_categories[col].add(CAT_NA_VALUE)
    #         unique_categories[col].update(
    #             chunk[col].fillna(CAT_NA_VALUE).astype(str).unique()
    #         )

    # categories_for_e= [
    #     sorted(list(unique_categories[cat]))
    #     for cat in kept_cat_cols.index.to_list()
    # ]

    # encoder = OrdinalEncoder(
    #     categories=categories_for_e, 
    #     handle_unknown="use_encoded_value", # Fixed parameter
    #     unknown_value=-1,                   # Required when using 'use_encoded_value'
    #     encoded_missing_value=-1
    # )
    # df = pd.read_csv('data/bank_data_train.csv', usecols=kept_cat_cols.index.to_list(), nrows=10)
    # encoder.fit(df)
    # del df
    ## 50% or more missing values
    cols_gte_50 = kept_num_cols[kept_num_cols >= 50].index.tolist()
    missing_flag_cols = []

    for col in cols_gte_50:

    # --- For value AUC ---
        y_true_val = []
        y_score_val = []

        # --- For missing AUC ---
        y_true_miss = []
        y_score_miss = []

        for chunk in pd.read_csv(
            "data/bank_data_train.csv",
            chunksize=100_000,
            dtype=dtypes.to_dict(),
            usecols=[col, "TARGET"]
        ):
            notna_mask = chunk[col].notna()

            if notna_mask.any():
                y_true_val.append(chunk.loc[notna_mask, "TARGET"].values)
                y_score_val.append(chunk.loc[notna_mask, col].values)

            # ---------- MISSINGNESS AUC ----------
            miss_flag = chunk[col].isna().astype(int)

            # keep all rows (as long as TARGET exists)
            y_true_miss.append(chunk["TARGET"].values)
            y_score_miss.append(miss_flag.values)

        # ----- Compute AUCs safely -----
        auc_value = None
        if len(y_true_val) > 0:
            auc_value = roc_auc_score(
                np.concatenate(y_true_val),
                np.concatenate(y_score_val)
            )

        auc_missing = roc_auc_score(
            np.concatenate(y_true_miss),
            np.concatenate(y_score_miss)
        )

        print(
            f"{col:35s} | "
                f"AUC_value: {auc_value:.4f} | "
            f"AUC_missing: {auc_missing:.4f}, missing value: {nan_counts[col]}%"
        )
        if auc_missing < 0.55 and auc_value < 0.55:
            kept_cols.drop(col)
        if auc_value < 0.55 and auc_missing >= 0.55:
            missing_flag_cols.append(col)

    # cat_cols_gte_50 = kept_cat_cols[kept_cat_cols > 50].index.tolist()

    # for col in cat_cols_gte_50:
    #     # Dictionary to accumulate counts of (category, target) pairs
    #     contingency_counts = {}

    #     total_samples = 0

    #     for chunk in pd.read_csv(
    #         "data/bank_data_train.csv",
    #         chunksize=100_000,
    #         dtype=dtypes.to_dict(),
    #         usecols=[col, "TARGET"]
    #     ):
    #         # Drop rows with missing TARGET
    #         chunk = chunk.dropna(subset=["TARGET"])

    #         # Treat missing in feature as separate category
    #         chunk[col] = chunk[col].fillna("MISSING")

    #         # Count occurrences of (category, target)
    #         counts = chunk.groupby([col, "TARGET"]).size()

    #         # Accumulate counts
    #         for (category, target_val), count in counts.items():
    #             contingency_counts[(category, target_val)] = contingency_counts.get((category, target_val), 0) + count

    #         total_samples += len(chunk)

    #     # Build contingency table DataFrame
    #     categories = sorted(set([k[0] for k in contingency_counts.keys()]))
    #     targets = sorted(set([k[1] for k in contingency_counts.keys()]))

    #     table = pd.DataFrame(0, index=categories, columns=targets, dtype=int)

    #     for (category, target_val), count in contingency_counts.items():
    #         table.at[category, target_val] = count

    #     # Run Chi-square test
    #     chi2, p, dof, expected = chi2_contingency(table)

    #     # Calculate Cramér’s V
    #     k = min(table.shape)  # min of #rows or #cols
    #     V = cramers_v(chi2, total_samples, k)

    #     print(f"{col:35s} | p-value: {p:.6f} | Cramér’s V: {V:.4f} | Samples: {total_samples}")           







if __name__ == "__main__":
    main()
