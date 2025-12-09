import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler


class ScalingDealer(BaseEstimator, TransformerMixin):

    def __init__(self, 
                 scaler_name="robust", 
                 **kwargs
                 ):

        self.scaler_name = scaler_name

    def fit(self, X, **kwargs):

        scalers = {
            "robust": RobustScaler,
            "minmax": MinMaxScaler,
            "standard": StandardScaler
        }

        self.cols_to_scale_ = ["Latitude", "Longitude", "Income", "Customer Lifetime Value", "Days_in_prog", "TotalFlights",
                               "TotalDistance", "TotalPointsAccumulated", "TotalPointsRedeemed", "RecencyInMonths",
                               "Pct_Spend_Month_1", "Pct_Spend_Month_2", "Pct_Spend_Month_3", "Pct_Spend_Month_4", 
                               "Pct_Spend_Month_5", "Pct_Spend_Month_6", "Pct_Spend_Month_7", "Pct_Spend_Month_8", 
                               "Pct_Spend_Month_9", "Pct_Spend_Month_10", "Pct_Spend_Month_11", "Pct_Spend_Month_12", 
                               "Flights_per_day"]

        self.scaler_ = scalers[self.scaler_name]().fit(X[self.cols_to_scale_])
        return self

    def transform(self, X, **kwargs):
        
        X = X.copy()

        X_cols = X[self.cols_to_scale_]
        X_encoded = X[[col for col in X.columns if col not in self.cols_to_scale_]]

        X_scaled = self.scaler_.transform(X_cols)

        X_scaled = pd.DataFrame(X_scaled, columns= X_cols.columns, index= X_cols.index)
        
        return pd.concat([X_scaled, X_encoded], axis=1)
