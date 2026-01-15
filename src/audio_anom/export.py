"""Model export and deployment utilities."""

import pickle
import json
from pathlib import Path
import numpy as np


class ModelExporter:
    """Export trained models with all necessary components for deployment."""
    
    def __init__(self):
        """Initialize exporter."""
        pass
    
    def export_model_package(
        self,
        model,
        scaler,
        pca,
        feature_cols,
        config,
        performance_metrics,
        output_path,
        autoencoder=None,
        threshold=None,
        model_name="best_model"
    ):
        """
        Export complete model package for deployment.
        
        Args:
            model: Trained model (sklearn/xgboost)
            scaler: Fitted StandardScaler
            pca: Fitted PCA transformer
            feature_cols: List of feature column names
            config: Configuration dict or class
            performance_metrics: Dict or list of performance metrics
            output_path: Path to save the model package
            autoencoder: Trained autoencoder (optional, for Keras models)
            threshold: Anomaly threshold (optional, for autoencoders)
            model_name: Name of the model
            
        Returns:
            Path to saved model package
        """
        # Convert config to dict if it's a class
        if hasattr(config, '__dict__'):
            config_dict = {
                k: v for k, v in config.__dict__.items()
                if not k.startswith('_') and not callable(v)
            }
        else:
            config_dict = config
        
        # Create model package
        model_package = {
            'model': model if not autoencoder else 'autoencoder',
            'scaler': scaler,
            'pca': pca,
            'feature_cols': feature_cols,
            'config': config_dict,
            'performance': performance_metrics,
            'model_name': model_name,
            'autoencoder': autoencoder,
            'threshold': threshold
        }
        
        # Save package
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'wb') as f:
            pickle.dump(model_package, f)
        
        print(f"\n✓ Model package saved to: {output_path}")
        print(f"  - Model: {model_name}")
        print(f"  - Features: {len(feature_cols)}")
        print(f"  - PCA components: {pca.n_components_ if pca else 'N/A'}")
        
        # Also save a JSON metadata file
        metadata_path = output_path.with_suffix('.json')
        metadata = {
            'model_name': model_name,
            'num_features': len(feature_cols),
            'pca_components': int(pca.n_components_) if pca else None,
            'performance': self._sanitize_metrics(performance_metrics),
            'config': self._sanitize_config(config_dict)
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✓ Metadata saved to: {metadata_path}")
        
        return output_path
    
    def load_model_package(self, package_path):
        """
        Load model package from file.
        
        Args:
            package_path: Path to model package
            
        Returns:
            Dict containing all model components
        """
        with open(package_path, 'rb') as f:
            package = pickle.load(f)
        
        print(f"✓ Model package loaded from: {package_path}")
        print(f"  - Model: {package.get('model_name', 'Unknown')}")
        
        return package
    
    def predict_with_package(self, package, audio_features):
        """
        Make predictions using a loaded model package.
        
        Args:
            package: Loaded model package dict
            audio_features: Feature dict from AudioFeatureExtractor
            
        Returns:
            Tuple of (prediction, probability/score)
        """
        # Build feature vector
        feature_vector = []
        for col in package['feature_cols']:
            feature_vector.append(audio_features.get(col, 0))
        
        X = np.array(feature_vector).reshape(1, -1)
        
        # Scale
        if package['scaler'] is not None:
            X = package['scaler'].transform(X)
        
        # PCA
        if package['pca'] is not None:
            X = package['pca'].transform(X)
        
        # Predict
        if package['autoencoder'] is not None:
            # Autoencoder prediction
            reconstruction = package['autoencoder'].predict(X, verbose=0)
            mse = np.mean(np.square(X - reconstruction))
            prediction = 1 if mse > package['threshold'] else 0
            score = mse
        else:
            # Sklearn/XGBoost prediction
            prediction = package['model'].predict(X)[0]
            if hasattr(package['model'], 'predict_proba'):
                score = package['model'].predict_proba(X)[0, 1]
            else:
                score = prediction
        
        return prediction, score
    
    def _sanitize_metrics(self, metrics):
        """Convert metrics to JSON-serializable format."""
        if isinstance(metrics, list):
            return [self._sanitize_dict(m) for m in metrics]
        elif isinstance(metrics, dict):
            return self._sanitize_dict(metrics)
        return metrics
    
    def _sanitize_dict(self, d):
        """Convert dict values to JSON-serializable types."""
        result = {}
        for k, v in d.items():
            if isinstance(v, (np.integer, np.floating)):
                result[k] = float(v)
            elif isinstance(v, np.ndarray):
                result[k] = v.tolist()
            elif isinstance(v, (list, tuple)):
                result[k] = [float(x) if isinstance(x, (np.integer, np.floating)) else x for x in v]
            else:
                result[k] = v
        return result
    
    def _sanitize_config(self, config):
        """Sanitize config dict for JSON serialization."""
        result = {}
        for k, v in config.items():
            if isinstance(v, (int, float, str, bool, type(None))):
                result[k] = v
            elif isinstance(v, (np.integer, np.floating)):
                result[k] = float(v)
            else:
                result[k] = str(v)
        return result
