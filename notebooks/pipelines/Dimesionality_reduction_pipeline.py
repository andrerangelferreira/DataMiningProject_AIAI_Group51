import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA, KernelPCA, TruncatedSVD
import umap


class Dimensionality_Reductor(BaseEstimator, TransformerMixin):

    def __init__(self, 
                 method="pca",
                 n_components=2,
                 kernel="rbf",
                 n_neighbors=15,
                 min_dist=0.1,
                 metric='euclidean',
                 random_state=42,
                 **kwargs):
        
        self.method = method
        self.n_components = n_components
        self.kernel = kernel
        self.random_state = random_state
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.metric = metric
        self.kwargs = kwargs

    def fit(self, X, y=None):

        if self.method == "pca":
            self.model_ = PCA(
                n_components=self.n_components,
                random_state=self.random_state,
                **self.kwargs
            )

        elif self.method == "kernel_pca":
            self.model_ = KernelPCA(
                n_components=self.n_components,
                kernel=self.kernel,
                **self.kwargs
            )

        elif self.method == "svd":
            self.model_ = TruncatedSVD(
                n_components=self.n_components,
                random_state=self.random_state,
                **self.kwargs
            )

        elif self.method == "umap":
            self.model_ = umap.UMAP(
                n_components=self.n_components,
                n_neighbors=self.n_neighbors,
                min_dist=self.min_dist,
                metric=self.metric,
                random_state=self.random_state,
                **self.kwargs
            )

        self.model_.fit(X)
        return self

    def transform(self, X, y=None):
        return self.model_.transform(X)
