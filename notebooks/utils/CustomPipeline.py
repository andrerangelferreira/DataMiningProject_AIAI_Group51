import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, ClusterMixin, TransformerMixin, clone
from sklearn.utils.validation import check_is_fitted


class CustomPipeline(BaseEstimator, ClusterMixin):

    def __init__(
        self,
        imputer=None,
        outlier_remover=None,
        scaler=None,
        model=None
    ):
        self.imputer = imputer
        self.outlier_remover = outlier_remover
        self.scaler = scaler
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
        
        self.scaler_ = clone(self.scaler)
        X_clean = self.scaler_.fit_transform(X)

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
        X_transformed = self.scaler_.fit_transform(X)
        
        return self.model_.predict(X_transformed)

