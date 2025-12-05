import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')


class MissForestImputer(BaseEstimator, TransformerMixin):
    """
    MissForest imputation using Random Forest.
    Iteratively imputes missing values using Random Forest predictions.
    """

    def __init__(self, max_iter=10, n_estimators=100, random_state=42):
        self.max_iter = max_iter
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.statistics_ = None
        
    def fit(self, X, y=None):
        """Fit the imputer."""
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        # Initialize with mean imputation
        self.statistics_ = np.nanmean(X, axis=0)
        return self
    
    def transform(self, X):
        """Transform X by imputing missing values using Random Forest."""
        if isinstance(X, pd.DataFrame):
            X_copy = X.values.copy()
        else:
            X_copy = X.copy()
        
        # Get missing value mask
        missing_mask = np.isnan(X_copy)
        
        # If no missing values, return as is
        if not missing_mask.any():
            return X_copy
        
        # Initialize with mean
        for col in range(X_copy.shape[1]):
            X_copy[missing_mask[:, col], col] = self.statistics_[col]
        
        # Iterative imputation
        for iteration in range(self.max_iter):
            X_previous = X_copy.copy()
            
            for col in range(X_copy.shape[1]):
                if not missing_mask[:, col].any():
                    continue
                
                # Features and target
                obs_mask = ~missing_mask[:, col]
                miss_mask = missing_mask[:, col]
                
                if obs_mask.sum() == 0:
                    continue
                
                # Other features
                other_cols = [i for i in range(X_copy.shape[1]) if i != col]
                X_train = X_copy[obs_mask][:, other_cols]
                y_train = X_copy[obs_mask, col]
                X_test = X_copy[miss_mask][:, other_cols]
                
                # Train Random Forest
                rf = RandomForestRegressor(
                    n_estimators=self.n_estimators,
                    random_state=self.random_state,
                    n_jobs=-1
                )
                rf.fit(X_train, y_train)
                
                # Predict missing values
                X_copy[miss_mask, col] = rf.predict(X_test)
            
            # Check convergence
            if np.allclose(X_copy, X_previous, rtol=1e-3):
                break
        
        return X_copy


class AutoencoderImputer(BaseEstimator, TransformerMixin):
    """
    Autoencoder-based imputation using neural networks.
    Learns a compressed representation and reconstructs missing values.
    """
    
    def __init__(self, encoding_dim=None, epochs=50, batch_size=32, random_state=42):
        self.encoding_dim = encoding_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.random_state = random_state
        self.model_ = None
        self.scaler_ = None
        
    def _build_autoencoder(self, input_dim):
        """Build autoencoder model."""
        try:
            from tensorflow import keras
            from tensorflow.keras import layers
            import tensorflow as tf
            tf.random.set_seed(self.random_state)
        except ImportError:
            raise ImportError("TensorFlow required for AutoencoderImputer. Install: pip install tensorflow")
        
        encoding_dim = self.encoding_dim if self.encoding_dim else max(2, input_dim // 2)
        
        # Encoder
        encoder_input = layers.Input(shape=(input_dim,))
        encoded = layers.Dense(encoding_dim * 2, activation='relu')(encoder_input)
        encoded = layers.Dropout(0.2)(encoded)
        encoded = layers.Dense(encoding_dim, activation='relu')(encoded)
        
        # Decoder
        decoded = layers.Dense(encoding_dim * 2, activation='relu')(encoded)
        decoded = layers.Dropout(0.2)(decoded)
        decoded = layers.Dense(input_dim, activation='linear')(decoded)
        
        # Autoencoder
        autoencoder = keras.Model(encoder_input, decoded)
        autoencoder.compile(optimizer='adam', loss='mse')
        
        return autoencoder
    
    def fit(self, X, y=None):
        """Fit the autoencoder."""
        try:
            from tensorflow import keras
        except ImportError:
            raise ImportError("TensorFlow required. Install: pip install tensorflow")
        
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        # Scale data
        from sklearn.preprocessing import StandardScaler
        self.scaler_ = StandardScaler()
        
        # Use only complete cases for training
        complete_mask = ~np.isnan(X).any(axis=1)
        X_complete = X[complete_mask]
        
        if len(X_complete) < 10:
            # Not enough complete cases, use mean imputation first
            X_complete = X.copy()
            col_means = np.nanmean(X_complete, axis=0)
            for col in range(X_complete.shape[1]):
                X_complete[np.isnan(X_complete[:, col]), col] = col_means[col]
        
        X_scaled = self.scaler_.fit_transform(X_complete)
        
        # Build and train autoencoder
        self.model_ = self._build_autoencoder(X.shape[1])
        self.model_.fit(
            X_scaled, X_scaled,
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=0,
            validation_split=0.1
        )
        
        return self
    
    def transform(self, X):
        """Transform X by imputing with autoencoder."""
        if isinstance(X, pd.DataFrame):
            X_copy = X.values.copy()
        else:
            X_copy = X.copy()
        
        missing_mask = np.isnan(X_copy)
        
        # Initial imputation with mean
        col_means = np.nanmean(X_copy, axis=0)
        for col in range(X_copy.shape[1]):
            X_copy[missing_mask[:, col], col] = col_means[col]
        
        # Scale and reconstruct
        X_scaled = self.scaler_.transform(X_copy)
        X_reconstructed = self.model_.predict(X_scaled, verbose=0)
        X_reconstructed = self.scaler_.inverse_transform(X_reconstructed)
        
        # Replace only missing values
        X_copy[missing_mask] = X_reconstructed[missing_mask]
        
        return X_copy


class SoftImputeImputer(BaseEstimator, TransformerMixin):
    """
    Matrix completion using Soft-Impute algorithm.
    Uses singular value thresholding for low-rank matrix approximation.
    """
    
    def __init__(self, max_rank=None, lambda_reg=0.1, max_iter=100, tol=1e-5):
        self.max_rank = max_rank
        self.lambda_reg = lambda_reg
        self.max_iter = max_iter
        self.tol = tol
        self.col_means_ = None
        
    def _soft_threshold(self, x, threshold):
        """Soft thresholding operator."""
        return np.sign(x) * np.maximum(np.abs(x) - threshold, 0)
    
    def fit(self, X, y=None):
        """Fit the imputer."""
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        self.col_means_ = np.nanmean(X, axis=0)
        return self
    
    def transform(self, X):
        """Transform using Soft-Impute."""
        if isinstance(X, pd.DataFrame):
            X_copy = X.values.copy()
        else:
            X_copy = X.copy()
        
        missing_mask = np.isnan(X_copy)
        
        # Initialize with column means
        for col in range(X_copy.shape[1]):
            X_copy[missing_mask[:, col], col] = self.col_means_[col]
        
        # Soft-Impute iterations
        max_rank = self.max_rank if self.max_rank else min(X_copy.shape) // 2
        
        for iteration in range(self.max_iter):
            X_old = X_copy.copy()
            
            # SVD decomposition
            U, s, Vt = np.linalg.svd(X_copy, full_matrices=False)
            
            # Soft threshold singular values
            s_threshold = self._soft_threshold(s, self.lambda_reg)
            
            # Keep only top-k singular values
            s_threshold = s_threshold[:max_rank]
            U = U[:, :max_rank]
            Vt = Vt[:max_rank, :]
            
            # Reconstruct
            X_copy = U @ np.diag(s_threshold) @ Vt
            
            # Restore observed values
            X_copy[~missing_mask] = X[~missing_mask]
            
            # Check convergence
            if np.linalg.norm(X_copy - X_old) < self.tol:
                break
        
        return X_copy


class FlexibleImputer(BaseEstimator, TransformerMixin):
    """
    Flexible imputer supporting multiple advanced imputation strategies.
    Compatible with sklearn's RandomizedSearchCV.
    """
    
    def __init__(self, strategy='mean', n_neighbors=5, max_iter=10, 
                 n_estimators=100, encoding_dim=None, epochs=50,
                 lambda_reg=0.1, max_rank=None, random_state=42):
        """
        Imputation strategy. Options:
            - 'mean': Mean imputation
            - 'median': Median imputation
            - 'mode': Most frequent value
            - 'constant': Fill with constant (0)
            - 'knn': K-Nearest Neighbors
            - 'iterative': Iterative imputation (MICE)
            - 'missforest': Random Forest imputation
            - 'autoencoder': Deep learning autoencoder
            - 'softimpute': Matrix completion
            - 'forward_fill': Forward fill (time series)
            - 'backward_fill': Backward fill (time series)
        """
        self.strategy = strategy
        self.n_neighbors = n_neighbors
        self.max_iter = max_iter
        self.n_estimators = n_estimators
        self.encoding_dim = encoding_dim
        self.epochs = epochs
        self.lambda_reg = lambda_reg
        self.max_rank = max_rank
        self.random_state = random_state
        self.imputer_ = None
        self.is_dataframe_ = False
        self.columns_ = None
        
    def fit(self, X, y=None):
        """Fit the imputer."""
        if isinstance(X, pd.DataFrame):
            self.is_dataframe_ = True
            self.columns_ = X.columns
            X_array = X.values
        else:
            X_array = X
        
        # Select and initialize imputer
        if self.strategy in ['mean', 'median', 'most_frequent', 'constant']:
            strategy_map = {
                'mean': 'mean',
                'median': 'median',
                'mode': 'most_frequent',
                'constant': 'constant'
            }
            self.imputer_ = SimpleImputer(
                strategy=strategy_map.get(self.strategy, self.strategy),
                fill_value=0 if self.strategy == 'constant' else None
            )
            
        elif self.strategy == 'knn':
            self.imputer_ = KNNImputer(n_neighbors=self.n_neighbors)
            
        elif self.strategy == 'iterative':
            self.imputer_ = IterativeImputer(
                max_iter=self.max_iter,
                random_state=self.random_state
            )
            
        elif self.strategy == 'missforest':
            self.imputer_ = MissForestImputer(
                max_iter=self.max_iter,
                n_estimators=self.n_estimators,
                random_state=self.random_state
            )
            
        elif self.strategy == 'autoencoder':
            self.imputer_ = AutoencoderImputer(
                encoding_dim=self.encoding_dim,
                epochs=self.epochs,
                random_state=self.random_state
            )
            
        elif self.strategy == 'softimpute':
            self.imputer_ = SoftImputeImputer(
                max_rank=self.max_rank,
                lambda_reg=self.lambda_reg,
                max_iter=self.max_iter
            )
            
        elif self.strategy in ['forward_fill', 'backward_fill']:
            self.imputer_ = None
            
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
        
        if self.imputer_ is not None:
            self.imputer_.fit(X_array)
        
        return self
    
    def transform(self, X):
        """Transform X by imputing missing values."""
        index = None
        if isinstance(X, pd.DataFrame):
            index = X.index
            X_array = X.values
        else:
            X_array = X
        
        # Apply imputation
        if self.strategy == 'forward_fill':
            if isinstance(X, pd.DataFrame):
                X_imputed = X.fillna(method='ffill').values
            else:
                X_imputed = pd.DataFrame(X).fillna(method='ffill').values
                
        elif self.strategy == 'backward_fill':
            if isinstance(X, pd.DataFrame):
                X_imputed = X.fillna(method='bfill').values
            else:
                X_imputed = pd.DataFrame(X).fillna(method='bfill').values
        else:
            X_imputed = self.imputer_.transform(X_array)
        
        # Convert back to DataFrame if needed
        if self.is_dataframe_ and self.columns_ is not None:
            return pd.DataFrame(X_imputed, columns=self.columns_, index=index)
        
        return X_imputed


def create_imputation_pipeline(model=None):
    """
    Create pipeline with flexible imputation.
    model : estimator, optional
        Sklearn model to add to pipeline
    """
    steps = [
        ('imputer', FlexibleImputer()),
        ('scaler', StandardScaler())
    ]
    
    if model is not None:
        steps.append(('model', model))
    
    return Pipeline(steps)


def get_imputation_param_grid(include_deep_learning=False):
    """
    Get parameter grid for random search.
    include_deep_learning : bool, default=False
        Whether to include autoencoder (requires TensorFlow)
    """
    strategies = ['mean', 'median', 'mode', 'knn', 'iterative', 'missforest', 'softimpute']
    
    if include_deep_learning:
        strategies.append('autoencoder')
    
    param_distributions = {
        'imputer__strategy': strategies,
        'imputer__n_neighbors': [3, 5, 7, 10],  # KNN
        'imputer__max_iter': [5, 10, 20],  # Iterative, MissForest, SoftImpute
        'imputer__n_estimators': [50, 100, 200],  # MissForest
        'imputer__lambda_reg': [0.01, 0.1, 1.0],  # SoftImpute
        'imputer__epochs': [30, 50, 100],  # Autoencoder
    }
    
    return param_distributions