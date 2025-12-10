import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor


class OutliersDealer(BaseEstimator, TransformerMixin):

    def __init__(self, 
                 outlier_method = "z_score", # the default method
                 threshold=3, # Pick 2 or 3 as the threshold value of "z"
                 z_columns = None,
                 contamination_IF=0.05, 
                 estimators_IF = 50,
                 max_samples_IF = "auto",
                 random_state=42,
                 n_neighbors=20, 
                 contamination_LOF= 0.05,
                 metric_LOF = "euclidean",
                 model_columns = None, #columns selected for IF model or LOF model
                 **kwargs
                 ):
        
        self.outlier_method = outlier_method

        # Z-Score method parameters
        self.threshold = threshold
        self.z_columns = z_columns

        # Isolation Forest method parameters
        self.contamination_IF = contamination_IF
        self.random_state = random_state
        self.estimators_IF = estimators_IF
        self.max_samples_IF = max_samples_IF
        self.model_columns = model_columns

        # Local Outliers Factor method parameters
        self.n_neighbors = n_neighbors
        self.contamination_LOF = contamination_LOF
        self.metric_LOF = metric_LOF
        self.model_columns = model_columns


    def fit(self, X, y = None,  **kwargs):

        if self.outlier_method == "z_score":

            self.means_ = {}
            self.stds_ = {}

            if self.z_columns is None:
                self.z_columns = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

            for col in self.z_columns:
                self.means_[col] = X[col].mean()
                self.stds_[col] = X[col].std()

        elif self.outlier_method == "Isolation_Forest":

            if self.model_columns is None:
                self.model_columns = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

            self.model_ = IsolationForest(
            contamination=self.contamination_IF,
            n_estimators= self.estimators_IF,
            max_samples = self.max_samples_IF,
            random_state=self.random_state
            )
            self.model_.fit(X[self.model_columns])

            preds = self.model_.predict(X[self.model_columns])  # +1 = normal, -1 = outlier
            
            # Find outlier indices
            outlier_mask = (preds == -1)
            
            # Cap outliers to median values (or use quantiles)
            for col in self.model_columns:
                median_val = X[col].median()
                X.loc[outlier_mask, col] = median_val

            self.X_ = X

        elif self.outlier_method == "LOF":

            if self.model_columns is None:
                self.model_columns = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

            self.model_ = LocalOutlierFactor(
            n_neighbors=self.n_neighbors,
            metric= self.metric_LOF,
            contamination=self.contamination_LOF, 
            novelty= True
            )
            self.model_.fit(X[self.model_columns])

            preds = self.model_.predict(X[self.model_columns])  # +1 = normal, -1 = outlier
            
            # Find outlier indices
            outlier_mask = (preds == -1)
            
            # Cap outliers to median values (or use quantiles)
            for col in self.model_columns:
                median_val = X[col].median()
                X.loc[outlier_mask, col] = median_val
            
            self.X_ = X

        return self
    
    def transform(self, X, y = None, **kwargs):


        X = X.copy()
        y = y.copy() if y != None else None

        if self.outlier_method == "z_score":

            for col in self.z_columns:

                X[col] = np.clip(X[col],
                                self.means_[col] - self.threshold * self.stds_[col],
                                self.means_[col] + self.threshold * self.stds_[col]
                            )
            return X
        
        elif self.outlier_method in ["Isolation_Forest", "LOF"]:
                return self.X_
