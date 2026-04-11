# import sys
import joblib
import pandas as pd
import sklearn

from sklearn.compose import ColumnTransformer
from lib.column_transformer import MissingAwareColumnSelector, Transformer
from lib.data_preparation import DataSplitter, DataBalancer
from lib.preprocessors import cat_pipeline, num_pipeline

sklearn.set_config(transform_output="pandas")

pd.set_option('display.max_rows', None)

splitter = DataSplitter(val_size=0.2, test_size=0.2)

train_df = splitter.split(path='data/bank_data_train.csv')
      
balancer = DataBalancer()

df_train_balanced = balancer.balance_training_set(df=train_df, ratio=0.1, save=True, balance=False)

print("New Distribution:")
print(df_train_balanced['TARGET'].value_counts(normalize=True))

del train_df, splitter, balancer

y = df_train_balanced['TARGET']

df_train_balanced = df_train_balanced.drop(columns=['TARGET'])

selector = MissingAwareColumnSelector(missing_threshold=0.5, y_cols=[])

df_train_balanced = selector.fit_transform(df_train_balanced)

column_transformer = ColumnTransformer( 
    transformers=[
        ('num', num_pipeline(selector), selector.kept_num),
        ('cat', cat_pipeline(selector), selector.kept_cat),
    ],
    verbose_feature_names_out=False
)

column_transformer.fit(df_train_balanced, y=y)
transformer = Transformer(steps=[
    selector,
    column_transformer
])

joblib.dump(transformer, 'transformer.pkl')
print("preprocessing pipeline saved")
