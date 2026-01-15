"""Advanced anomaly detection models."""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import joblib

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    HAS_TENSORFLOW = True
except ImportError:
    HAS_TENSORFLOW = False


class RandomForestAnomalyDetector:
    """Random Forest based anomaly detector with hyperparameter tuning."""

    def __init__(self, random_state=42, n_splits=5):
        """
        Initialize Random Forest detector.

        Args:
            random_state: Random state for reproducibility
            n_splits: Number of cross-validation splits
        """
        self.random_state = random_state
        self.n_splits = n_splits
        self.scaler = StandardScaler()
        self.pca = None
        self.smote = SMOTE(random_state=random_state)
        self.model = None
        self.best_params = None
        self.is_fitted = False

    def fit(self, X, y, use_pca=True, pca_variance=0.95, use_smote=True, 
            param_grid=None, verbose=1):
        """
        Fit the Random Forest detector with optional PCA and SMOTE.

        Args:
            X: Feature matrix
            y: Labels
            use_pca: Whether to use PCA for dimensionality reduction
            pca_variance: Variance to retain in PCA
            use_smote: Whether to use SMOTE for balancing
            param_grid: Custom parameter grid for GridSearchCV
            verbose: Verbosity level

        Returns:
            Self
        """
        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # PCA
        if use_pca:
            self.pca = PCA(n_components=pca_variance, random_state=self.random_state)
            X_processed = self.pca.fit_transform(X_scaled)
            if verbose > 0:
                print(f"  PCA: {self.pca.n_components_} components "
                      f"({self.pca.explained_variance_ratio_.sum():.4f} variance)")
        else:
            X_processed = X_scaled
            self.pca = None

        # SMOTE
        if use_smote:
            X_resampled, y_resampled = self.smote.fit_resample(X_processed, y)
            if verbose > 0:
                print(f"  SMOTE: {X_processed.shape[0]} → {X_resampled.shape[0]} samples")
        else:
            X_resampled, y_resampled = X_processed, y

        # Default parameter grid
        if param_grid is None:
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2],
                'class_weight': ['balanced']
            }

        # GridSearchCV
        rf_model = RandomForestClassifier(random_state=self.random_state, n_jobs=-1)
        cv_strategy = StratifiedKFold(n_splits=self.n_splits, shuffle=True, 
                                      random_state=self.random_state)

        grid_search = GridSearchCV(
            rf_model, param_grid, cv=cv_strategy,
            scoring='f1', verbose=verbose, n_jobs=-1
        )
        grid_search.fit(X_resampled, y_resampled)

        self.model = grid_search.best_estimator_
        self.best_params = grid_search.best_params_
        self.best_cv_score = grid_search.best_score_
        self.is_fitted = True

        if verbose > 0:
            print(f"  Best params: {self.best_params}")
            print(f"  Best CV F1-Score: {self.best_cv_score:.4f}")

        return self

    def predict(self, X):
        """Predict anomalies."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        X_scaled = self.scaler.transform(X)
        if self.pca is not None:
            X_processed = self.pca.transform(X_scaled)
        else:
            X_processed = X_scaled

        return self.model.predict(X_processed)

    def predict_proba(self, X):
        """Predict anomaly probabilities."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        X_scaled = self.scaler.transform(X)
        if self.pca is not None:
            X_processed = self.pca.transform(X_scaled)
        else:
            X_processed = X_scaled

        return self.model.predict_proba(X_processed)

    def save(self, model_path):
        """Save model to disk."""
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted model")

        model_data = {
            "scaler": self.scaler,
            "pca": self.pca,
            "model": self.model,
            "best_params": self.best_params,
            "random_state": self.random_state,
        }
        joblib.dump(model_data, model_path)

    def load(self, model_path):
        """Load model from disk."""
        model_data = joblib.load(model_path)
        self.scaler = model_data["scaler"]
        self.pca = model_data.get("pca", None)
        self.model = model_data["model"]
        self.best_params = model_data.get("best_params", None)
        self.random_state = model_data["random_state"]
        self.is_fitted = True
        return self


class XGBoostAnomalyDetector:
    """XGBoost based anomaly detector with hyperparameter tuning."""

    def __init__(self, random_state=42, n_splits=5):
        """
        Initialize XGBoost detector.

        Args:
            random_state: Random state for reproducibility
            n_splits: Number of cross-validation splits
        """
        self.random_state = random_state
        self.n_splits = n_splits
        self.scaler = StandardScaler()
        self.pca = None
        self.smote = SMOTE(random_state=random_state)
        self.model = None
        self.best_params = None
        self.is_fitted = False

    def fit(self, X, y, use_pca=True, pca_variance=0.95, use_smote=True, 
            param_grid=None, verbose=1):
        """
        Fit the XGBoost detector with optional PCA and SMOTE.

        Args:
            X: Feature matrix
            y: Labels
            use_pca: Whether to use PCA for dimensionality reduction
            pca_variance: Variance to retain in PCA
            use_smote: Whether to use SMOTE for balancing
            param_grid: Custom parameter grid for GridSearchCV
            verbose: Verbosity level

        Returns:
            Self
        """
        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # PCA
        if use_pca:
            self.pca = PCA(n_components=pca_variance, random_state=self.random_state)
            X_processed = self.pca.fit_transform(X_scaled)
            if verbose > 0:
                print(f"  PCA: {self.pca.n_components_} components")
        else:
            X_processed = X_scaled
            self.pca = None

        # SMOTE
        if use_smote:
            X_resampled, y_resampled = self.smote.fit_resample(X_processed, y)
            if verbose > 0:
                print(f"  SMOTE: {X_processed.shape[0]} → {X_resampled.shape[0]} samples")
        else:
            X_resampled, y_resampled = X_processed, y

        # Calculate scale_pos_weight
        scale_pos_weight = len(y_resampled[y_resampled == 0]) / len(y_resampled[y_resampled == 1])

        # Default parameter grid
        if param_grid is None:
            param_grid = {
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1, 0.3],
                'n_estimators': [100, 200],
                'subsample': [0.8, 1.0],
                'colsample_bytree': [0.8, 1.0]
            }

        # GridSearchCV
        xgb_model = XGBClassifier(
            objective='binary:logistic',
            eval_metric='logloss',
            scale_pos_weight=scale_pos_weight,
            random_state=self.random_state
        )
        cv_strategy = StratifiedKFold(n_splits=self.n_splits, shuffle=True, 
                                      random_state=self.random_state)

        grid_search = GridSearchCV(
            xgb_model, param_grid, cv=cv_strategy,
            scoring='f1', verbose=verbose, n_jobs=-1
        )
        grid_search.fit(X_resampled, y_resampled)

        self.model = grid_search.best_estimator_
        self.best_params = grid_search.best_params_
        self.best_cv_score = grid_search.best_score_
        self.is_fitted = True

        if verbose > 0:
            print(f"  Best params: {self.best_params}")
            print(f"  Best CV F1-Score: {self.best_cv_score:.4f}")

        return self

    def predict(self, X):
        """Predict anomalies."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        X_scaled = self.scaler.transform(X)
        if self.pca is not None:
            X_processed = self.pca.transform(X_scaled)
        else:
            X_processed = X_scaled

        return self.model.predict(X_processed)

    def predict_proba(self, X):
        """Predict anomaly probabilities."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        X_scaled = self.scaler.transform(X)
        if self.pca is not None:
            X_processed = self.pca.transform(X_scaled)
        else:
            X_processed = X_scaled

        return self.model.predict_proba(X_processed)

    def save(self, model_path):
        """Save model to disk."""
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted model")

        model_data = {
            "scaler": self.scaler,
            "pca": self.pca,
            "model": self.model,
            "best_params": self.best_params,
            "random_state": self.random_state,
        }
        joblib.dump(model_data, model_path)

    def load(self, model_path):
        """Load model from disk."""
        model_data = joblib.load(model_path)
        self.scaler = model_data["scaler"]
        self.pca = model_data.get("pca", None)
        self.model = model_data["model"]
        self.best_params = model_data.get("best_params", None)
        self.random_state = model_data["random_state"]
        self.is_fitted = True
        return self


class AutoencoderAnomalyDetector:
    """Autoencoder based anomaly detector for unsupervised learning."""

    def __init__(self, encoding_dim=10, random_state=42):
        """
        Initialize Autoencoder detector.

        Args:
            encoding_dim: Dimension of the encoding layer
            random_state: Random state for reproducibility
        """
        if not HAS_TENSORFLOW:
            raise ImportError("TensorFlow is required for AutoencoderAnomalyDetector")
        
        self.encoding_dim = encoding_dim
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.pca = None
        self.autoencoder = None
        self.threshold = None
        self.is_fitted = False

    def _build_autoencoder(self, input_dim):
        """Build autoencoder model."""
        # Encoder
        input_layer = layers.Input(shape=(input_dim,))
        encoded = layers.Dense(64, activation='relu')(input_layer)
        encoded = layers.Dense(32, activation='relu')(encoded)
        encoded = layers.Dense(self.encoding_dim, activation='relu')(encoded)
        
        # Decoder
        decoded = layers.Dense(32, activation='relu')(encoded)
        decoded = layers.Dense(64, activation='relu')(decoded)
        decoded = layers.Dense(input_dim, activation='linear')(decoded)
        
        autoencoder = keras.Model(input_layer, decoded)
        autoencoder.compile(optimizer='adam', loss='mse')
        
        return autoencoder

    def fit(self, X, y=None, use_pca=True, pca_variance=0.95, epochs=50, 
            batch_size=32, validation_split=0.2, verbose=0):
        """
        Fit the Autoencoder on normal data only.

        Args:
            X: Feature matrix
            y: Labels (optional, used to filter normal samples)
            use_pca: Whether to use PCA
            pca_variance: Variance to retain in PCA
            epochs: Number of training epochs
            batch_size: Batch size
            validation_split: Validation split ratio
            verbose: Verbosity level

        Returns:
            Self
        """
        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # PCA
        if use_pca:
            self.pca = PCA(n_components=pca_variance, random_state=self.random_state)
            X_processed = self.pca.fit_transform(X_scaled)
        else:
            X_processed = X_scaled
            self.pca = None

        # Train only on normal data (label == 0)
        if y is not None:
            X_normal = X_processed[y == 0]
        else:
            X_normal = X_processed

        # Build and train autoencoder
        self.autoencoder = self._build_autoencoder(X_processed.shape[1])
        self.history = self.autoencoder.fit(
            X_normal, X_normal,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=verbose
        )

        # Calculate threshold (95th percentile of reconstruction error on normal data)
        recon = self.autoencoder.predict(X_normal, verbose=0)
        mse = np.mean(np.square(X_normal - recon), axis=1)
        self.threshold = np.percentile(mse, 95)
        
        self.is_fitted = True
        return self

    def predict(self, X):
        """Predict anomalies based on reconstruction error."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        X_scaled = self.scaler.transform(X)
        if self.pca is not None:
            X_processed = self.pca.transform(X_scaled)
        else:
            X_processed = X_scaled

        recon = self.autoencoder.predict(X_processed, verbose=0)
        mse = np.mean(np.square(X_processed - recon), axis=1)
        
        return (mse > self.threshold).astype(int)

    def predict_scores(self, X):
        """Get reconstruction error as anomaly scores."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        X_scaled = self.scaler.transform(X)
        if self.pca is not None:
            X_processed = self.pca.transform(X_scaled)
        else:
            X_processed = X_scaled

        recon = self.autoencoder.predict(X_processed, verbose=0)
        mse = np.mean(np.square(X_processed - recon), axis=1)
        
        return mse

    def save(self, model_path):
        """Save model to disk."""
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted model")

        # Save autoencoder separately
        autoencoder_path = model_path.replace('.pkl', '_autoencoder.h5')
        self.autoencoder.save(autoencoder_path)

        model_data = {
            "scaler": self.scaler,
            "pca": self.pca,
            "threshold": self.threshold,
            "encoding_dim": self.encoding_dim,
            "random_state": self.random_state,
            "autoencoder_path": autoencoder_path,
        }
        joblib.dump(model_data, model_path)

    def load(self, model_path):
        """Load model from disk."""
        model_data = joblib.load(model_path)
        self.scaler = model_data["scaler"]
        self.pca = model_data.get("pca", None)
        self.threshold = model_data["threshold"]
        self.encoding_dim = model_data["encoding_dim"]
        self.random_state = model_data["random_state"]
        
        autoencoder_path = model_data["autoencoder_path"]
        self.autoencoder = keras.models.load_model(autoencoder_path)
        
        self.is_fitted = True
        return self
