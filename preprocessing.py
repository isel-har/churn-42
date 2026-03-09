import sys
import joblib
import pandas as pd
import sklearn

from sklearn.impute import SimpleImputer#, KNNImputer, IterativeImputer
from sklearn.preprocessing import StandardScaler, TargetEncoder,OrdinalEncoder#, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.feature_selection import VarianceThreshold
from preprocessors import CorrelationFilter, MissingAwareColumnSelector, Transformer

sklearn.set_config(transform_output="pandas")

def num_imputers(selector):

    imputers = []
    if selector.num_low:
        imputers.append(("num_low", SimpleImputer(strategy='median'), selector.num_low))

    if selector.num_mid:#LGBMImputer()
        imputers.append(("num_mid", SimpleImputer(strategy='median', add_indicator=True), selector.num_mid))
        
    if selector.num_high:
        imputers.append(("num_high", SimpleImputer(strategy='median', add_indicator=True), selector.num_high))

    return ColumnTransformer(transformers=imputers, remainder='passthrough')


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
                    strategy='constant',
                    fill_value='MISSING',
                    add_indicator=True
                ),
                selector.cat_high
            )
        )
    return ColumnTransformer(transformers=imputers, remainder='passthrough')


def num_pipeline(selector):

    return Pipeline(steps=[
        ('impute_group', num_imputers(selector)),
        ('remove_constants', VarianceThreshold(threshold=0.0001)), # almost constant value
        ('remove_duplicated', CorrelationFilter(threshold=0.95)), # duplicated features
        ('scaler', StandardScaler())
    ])


# def cat_encoders():

#     return ColumnTransformer(transformers=[
#         ('ordinal_cols', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), ['remainder__PACK']),
#         ('target_cols', TargetEncoder(), ['cat_high__CLNT_JOB_POSITION'])]
#         ,
#         remainder='passthrough'
#     )

def cat_pipeline(selector):
    return Pipeline([
        ('impute_group', cat_imputers(selector)),
        ('encoder', TargetEncoder(smooth=20))
    ])


def main():
    if len(sys.argv) != 2:
        print("Usage: python script.py data.csv")
        return

    try:
        balanced_df = pd.read_csv(sys.argv[1])
        y = balanced_df['TARGET']

        balanced_df = balanced_df.drop(columns=['TARGET'])

        selector    = MissingAwareColumnSelector(missing_threshold=0.5, y_cols=[])

        balanced_df = selector.fit_transform(balanced_df)

        # OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        #['CLNT_JOB_POSITION', 'PACK']
        column_transformer = ColumnTransformer( 
            transformers=[
                ('num', num_pipeline(selector), selector.kept_num),
                ('cat', cat_pipeline(selector), selector.kept_cat),
            ],
            verbose_feature_names_out=False
        )
        
        column_transformer.fit(balanced_df, y=y)

        transformer = Transformer(steps=[
            selector,
            column_transformer
        ])
        
        joblib.dump(transformer, 'transformer.pkl')
    
        print('transformers.pkl saved successfully.')

    except Exception as e:

        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
