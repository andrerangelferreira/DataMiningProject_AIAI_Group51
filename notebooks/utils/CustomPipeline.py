import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, ClusterMixin, TransformerMixin, clone
from sklearn.utils.validation import check_is_fitted


class CustomPipeline(BaseEstimator, ClusterMixin):

    def __init__(
        self,
        imputer=None,
        outlier_remover=None,
        encoder=None,
        scaler=None,
        reducer=None,
        selector=None,
        model=None
    ):
        self.imputer = imputer
        self.outlier_remover = outlier_remover
        self.encoder = encoder
        self.scaler = scaler
        self.reducer = reducer
        self.selector = selector
        self.model = model

    def fit(self, X, y=None):
        X = X.copy()

        if hasattr(X, 'columns'):
            X = X.copy()
        else:
            X = pd.DataFrame(X)  # Convert to DataFrame

        # Clone and fit each transformer
        self.imputer_ = clone(self.imputer)
        X = self.imputer_.fit_transform(X)
        
        self.outlier_remover_ = clone(self.outlier_remover)
        X = self.outlier_remover_.fit_transform(X)
        
        self.encoder_ = clone(self.encoder)
        X = self.encoder_.fit_transform(X)
        
        self.scaler_ = clone(self.scaler)
        X = self.scaler_.fit_transform(X)
        
        self.reducer_ = clone(self.reducer)
        X = self.reducer_.fit_transform(X)
        
        self.selector_ = clone(self.selector)
        X_clean = self.selector_.fit_transform(X)

        self.model_ = clone(self.model)
        self.model_.fit(X_clean)
        
        self.labels_ = self.model_.labels_
        self.X_ = X_clean  # save transformed data
        
        return self
    
    def predict(self, X):
        check_is_fitted(self, 'model_')
    
        # Transform the input X through the pipeline
        X = X.copy()
        X = self.imputer_.fit_transform(X)
        X = self.outlier_remover_.fit_transform(X)
        X = self.encoder_.fit_transform(X)
        X = self.scaler_.fit_transform(X)
        X = self.reducer_.fit_transform(X)
        X_transformed = self.selector_.fit_transform(X)
        
        return self.model_.predict(X_transformed)

