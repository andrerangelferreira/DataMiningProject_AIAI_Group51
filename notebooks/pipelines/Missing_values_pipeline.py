import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler


class MissingValuesDealer(BaseEstimator, TransformerMixin):

    def __init__(
        self,
        imputation_method="simple",  # "simple", "knn", "iterative"
        simple_strategy_num="mean",      # for simple imputer with numerical
        strategy_cat="most_frequent", #  imputer for categorical
        fill_value=None,             # used if strategy="constant"
        knn_neighbors=5,           
        random_state=42,
        knn_scaling_method= "standard",
        iterative_max_iter = 10,
        **kwargs
    ):
        self.imputation_method = imputation_method

        # Simple Imputer params
        self.simple_strategy_num = simple_strategy_num  #numerical variables
        self.strategy_cat = strategy_cat  #categorical variables
        self.fill_value = fill_value

        # KNN Imputer params
        self.knn_neighbors = knn_neighbors
        self.knn_scaling_method = knn_scaling_method

        # Iterative imputer
        self.iterative_max_iter = iterative_max_iter
        self.random_state = random_state


    def fit(self, X_train, **kwargs):

        #----- FITTING WITH SIMPLE IMPUTER -----

        if self.imputation_method == "simple":

            #imputer for numerical
            self.imputer_num = SimpleImputer(
                strategy=self.simple_strategy_num,
                fill_value=self.fill_value
            )
            self.imputer_num.fit(X_train.select_dtypes(include=np.number))

        #----- FITTING WITH KNN -----

        elif self.imputation_method == "knn":
            self.metric_features = X_train.select_dtypes(include=np.number).columns
  
            #scaler for knn

            if self.knn_scaling_method == "standard":
                self.scaler = StandardScaler()
            elif self.knn_scaling_method == "minmax":
                self.scaler = MinMaxScaler()
            elif self.knn_scaling_method == "robust":
                self.scaler = RobustScaler()

            #fit scaler      
            self.scaler.fit(X_train[self.metric_features])
            #Transform the data set to discover knn imputer 
            scaled = self.scaler.transform(X_train[self.metric_features])


            #imputer for numerical
            self.imputer_num = KNNImputer(
                n_neighbors=self.knn_neighbors
            )
            self.imputer_num.fit(scaled)


        # ----- FITTING WITH ITERATIVE IMPUTER -----       

        elif self.imputation_method == "iterative":

            #imputer for numerical
            self.imputer_num = IterativeImputer(
                max_iter = self.iterative_max_iter,
                random_state=self.random_state
            )
            self.imputer_num.fit(X_train.select_dtypes(include=np.number))

        return self


    def transform(self, X, y = None, **kwargs):

        X = X.copy()

        # Simple / Iterative Imputation
        if self.imputation_method in ["simple", "iterative"]:
            
            # Split columns
            num_cols = X.select_dtypes(include=np.number).columns

            #impute
            X_num_imputed = self.imputer_num.transform(X.select_dtypes(include=np.number))

            # Convert back to DataFrames
            df_num = pd.DataFrame(X_num_imputed, columns=num_cols, index=X.index)

            return df_num
        
        #----- KNN IMPUTATION -----

        elif self.imputation_method == "knn":

            # Split columns
            num_cols = X.select_dtypes(include=np.number).columns

            #Scale numeric values
            scaled = self.scaler.transform(X[num_cols])
            #impute scaled values
            imputed_scaled = self.imputer_num.transform(scaled)

            #inverse scale
            X_num_imputed = self.scaler.inverse_transform(imputed_scaled) 

            # Convert back to DataFrames
            df_num = pd.DataFrame(X_num_imputed, columns=num_cols, index=X.index)

            return df_num