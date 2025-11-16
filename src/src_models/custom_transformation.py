# -*- coding: utf-8 -*-
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
import pandas as pd


class MyStandardScaler(BaseEstimator, TransformerMixin):
    def __init__(self, st_sc, columns):
        self.st_sc = st_sc
        self.columns = columns

    def fit(self, X, y=None):
        self.st_sc.fit(X[self.columns])
        return self

    def transform(self, X, y=None):
        df1 = X.copy()
        df2 = pd.DataFrame(self.st_sc.transform(X[self.columns]), columns=self.columns)
        for col in df2.columns:
            df1[col] = df2[col].to_list()
        return df1


class MyPCA(BaseEstimator, TransformerMixin):
    # initializer
    def __init__(self, pca, columns):
        # save the features list internally in the class
        self.pca = pca
        self.columns = columns

    def fit(self, X, y=None):
        self.pca.fit(X[self.columns])
        return self

    def transform(self, X, y=None):
        df1 = X.copy()
        df1.drop(columns=self.columns, inplace=True)
        cols_labels = []
        for i in range(0, self.pca.n_components_):
            cols_labels.append('VAR{}'.format(i))
        df2 = pd.DataFrame(self.pca.transform(X[self.columns]), columns=cols_labels)
        for col in df2.columns:
            df1[col] = df2[col].to_list()
        return df1
