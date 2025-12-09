import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import VarianceThreshold
from sklearn.neighbors import kneighbors_graph
from sklearn.decomposition import PCA


class Feature_Selector(BaseEstimator, TransformerMixin):

    def __init__(self,
                 method="variance",
                 threshold=0.0,
                 corr_threshold=0.95,
                 n_features=10,
                 **kwargs):
        
        self.method = method
        self.threshold = threshold
        self.corr_threshold = corr_threshold
        self.n_features = n_features
        self.kwargs = kwargs

    def fit(self, X, y=None):
        
        if self.method == "variance":
            self.selector_ = VarianceThreshold(threshold=self.threshold)
            self.selector_.fit(X)

        elif self.method == "correlation":
            corr = X.corr().abs()
            upper = corr.where(
                np.triu(np.ones(corr.shape), k=1).astype(bool)
            )

            self.to_drop_ = [
                column for column in upper.columns 
                if any(upper[column] > self.corr_threshold)
            ]

        elif self.method == "pca_loading":
            pca = PCA(n_components=1)
            pca.fit(X)
            loadings = np.abs(pca.components_[0])
            self.selected_idx_ = np.argsort(loadings)[-self.n_features:]
        
        return self

    def transform(self, X, y=None):

        if self.method == "variance":
            return self.selector_.transform(X)

        elif self.method == "correlation":
            return X.drop(columns=self.to_drop_, errors="ignore")
        
        elif self.method == "pca_loading":
            return X.iloc[:, self.selected_idx_]

