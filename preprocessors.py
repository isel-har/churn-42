from sklearn.experimental import enable_iterative_imputer  # Necessary
from sklearn.impute import IterativeImputer
from lightgbm import LGBMRegressor
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np


class CorrelationFilter(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.8):
        self.threshold = threshold
        
    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        corr = X.corr().abs()
        upper = corr.where(
            np.triu(np.ones(corr.shape), k=1).astype(bool)
        )
        self.to_drop_ = [col for col in upper.columns if any(upper[col] > self.threshold)]
        return self
    
    def transform(self, X):
        X = pd.DataFrame(X)
        return X.drop(columns=self.to_drop_, errors='ignore')

class MissingAwareColumnSelector(BaseEstimator, TransformerMixin):

    def __init__(self, missing_threshold=0.5, y_cols=[], to_drop=['ID']):
        self.missing_threshold = missing_threshold
        self.y_cols = y_cols
        self.to_drop = to_drop
        self.is_fit  = False


    def fit(self, X, y=None):

        if self.is_fit == True:
            return self

        X = X.drop(columns=self.to_drop)
        num_cols = X.select_dtypes(include='number').columns.to_list()
        cat_cols = X.select_dtypes(exclude='number').columns.to_list()

        self.missing_ratio_ = X.isnull().mean()

        self.columns_to_keep_ = self.missing_ratio_[
            self.missing_ratio_ < self.missing_threshold
        ].index.tolist()

        self.low_missing_ = self.missing_ratio_[
            (self.missing_ratio_ > 0) &
            (self.missing_ratio_ <= 0.1)
        ].index.tolist()

        self.mid_missing_ = self.missing_ratio_[
            (self.missing_ratio_ > 0.1) &
            (self.missing_ratio_ <= 0.3)
        ].index.tolist()

        self.high_missing_ = self.missing_ratio_[
            (self.missing_ratio_ > 0.3) &
            (self.missing_ratio_ <= self.missing_threshold)
        ].index.tolist()


        self.not_missing_ = self.missing_ratio_[
            self.missing_ratio_ == 0
        ].index.tolist()

        self.kept_num = [c for c in self.columns_to_keep_ if c in num_cols and c not in self.y_cols]

        self.num_low = [c for c in self.low_missing_ if c in self.kept_num]
        self.num_mid = [c for c in self.mid_missing_ if c in self.kept_num]
        self.num_high = [c for c in self.high_missing_ if c in self.kept_num]

        self.kept_cat = [c for c in self.columns_to_keep_ if c in cat_cols]
        self.cat_low = [c for c in self.low_missing_ if c in self.kept_cat]
        self.cat_mid = [c for c in self.mid_missing_ if c in self.kept_cat]
        self.cat_high = [c for c in self.high_missing_ if c in self.kept_cat]

        self.is_fit = True

        return self


    def transform(self, X):
        if self.is_fit == False:
            raise Exception("run fit before transform.")
        transformed = X.copy()
        return transformed[self.columns_to_keep_]



class LGBMImputer(TransformerMixin, BaseEstimator):

    def __init__(self, max_iter=20, n_estimators=300):
        self.max_iter = max_iter
        self.n_estimators = n_estimators
        
        self.lgbm_imputer = IterativeImputer(
            max_iter=max_iter,
            estimator=LGBMRegressor(
                n_estimators=n_estimators,
                learning_rate=0.05,
                max_depth=6,
                num_leaves=31,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=0.1,
                random_state=42,
                n_jobs=-1
            ),
        )
    def fit(self, X, y=None):
        self.lgbm_imputer.fit(X)
        return self

    def transform(self, X):
        return self.lgbm_imputer.transform(X)


class Transformer:

    def __init__(self, steps=[]):
        self.steps= steps

    def transform(self, X):
    
        transformed = None
        for i, step in enumerate(self.steps):

            if i == 0:
                transformed = step.transform(X)
            else:
                transformed = step.transform(transformed)

        return transformed