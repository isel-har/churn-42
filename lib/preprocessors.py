import sklearn
from sklearn.preprocessing import TargetEncoder, FunctionTransformer, StandardScaler#, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from sklearn.feature_selection import VarianceThreshold
from lib.column_transformer import CorrelationFilter


import numpy as np

sklearn.set_config(transform_output="pandas")

# pd.set_option('display.max_rows', None)
# 0.8395581238803222 0.2 val 0.2 test without understampling

def num_imputers(selector):
    
    imputers = []

    if selector.num_low:
        imputers.append(("num_low", SimpleImputer(strategy='median'), selector.num_low))

    if selector.num_mid:#SimpleImputer(strategy='median', add_indicator=True)
        imputers.append(("num_mid", SimpleImputer(strategy='median', add_indicator=True), selector.num_mid))
        
    if selector.num_high:
        imputers.append(("num_high", SimpleImputer(strategy='median', add_indicator=True), selector.num_high))

    return ColumnTransformer(transformers=imputers, remainder='passthrough', verbose_feature_names_out=False)


def cat_imputers(selector):

    imputers = []
    if selector.cat_low:
        imputers.append(("cat_low", SimpleImputer(strategy='constant', fill_value='MISSING'), selector.cat_low))

    if selector.cat_mid:
        imputers.append(("cat_mid", SimpleImputer(strategy='constant', fill_value='MISSING'), selector.cat_mid))
        
    if selector.cat_high:
        imputers.append(
            (
                "cat_high",
                SimpleImputer(
                    strategy='most_frequent',
                    fill_value='MISSING',
                    add_indicator=True
                ),
                selector.cat_high
            )
        )
    return ColumnTransformer(transformers=imputers, remainder='passthrough', verbose_feature_names_out=False)



def num_pipeline(selector):

    return Pipeline(steps=[
        ('impute_group', num_imputers(selector)),
        ('remove_constants', VarianceThreshold(threshold=0.0001)),
        ('remove_duplicated', CorrelationFilter(threshold=0.95)),
        ('log_skewed', FunctionTransformer(np.log1p)),  # for skewed subset
        ('scaler', StandardScaler())
    ])


def cat_encoders():
    return ColumnTransformer(transformers=[
        ('pack_enc', TargetEncoder(smooth=1), ['PACK']),
        ('job_enc', TargetEncoder(smooth=100), ['CLNT_JOB_POSITION']),
    ], remainder='passthrough', verbose_feature_names_out=False)


def cat_pipeline(selector):
    return Pipeline([
        ('impute_group', cat_imputers(selector)),
        ('encoder', cat_encoders())
    ])