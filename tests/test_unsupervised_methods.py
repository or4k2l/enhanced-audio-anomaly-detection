"""Test suite for unsupervised anomaly detection methods."""

import pytest
import numpy as np
import tempfile
from pathlib import Path

from audio_anom.unsupervised_anomaly import (
    LocalOutlierFactorAnomalyDetector,
    IsolationForestAnomalyDetector,
    EllipticEnvelopeAnomalyDetector,
    create_detector,
)
from audio_anom.preprocessing_unsupervised import UnsupervisedPreprocessor
from audio_anom.evaluation_unsupervised import (
    evaluate_anomaly_detector,
    ModelComparator,
)


@pytest.fixture
def synthetic_data():
    """Generate synthetic data for testing."""
    np.random.seed(42)
    
    n_features = 50
    n_train = 200
    n_test_normal = 100
    n_test_anomaly = 50
    
    # Normal distribution
    mean_normal = np.zeros(n_features)
    X_train_normal = np.random.randn(n_train, n_features) + mean_normal
    X_test_normal = np.random.randn(n_test_normal, n_features) + mean_normal
    
    # Anomaly distribution (shifted)
    mean_anomaly = mean_normal + 2.0
    X_test_anomaly = np.random.randn(n_test_anomaly, n_features) + mean_anomaly
    
    # Combine test data
    X_test = np.vstack([X_test_normal, X_test_anomaly])
    y_test = np.array([0] * n_test_normal + [1] * n_test_anomaly)
    
    # Shuffle
    indices = np.random.permutation(len(y_test))
    X_test = X_test[indices]
    y_test = y_test[indices]
    
    return X_train_normal, X_test, y_test


class TestLocalOutlierFactorDetector:
    """Test LOF anomaly detector."""
    
    def test_initialization(self):
        """Test LOF initialization."""
        model = LocalOutlierFactorAnomalyDetector(n_neighbors=20, contamination=0.1)
        assert model.n_neighbors == 20
        assert model.contamination == 0.1
        assert not model.is_fitted
    
    def test_fit(self, synthetic_data):
        """Test LOF fitting."""
        X_train, _, _ = synthetic_data
        model = LocalOutlierFactorAnomalyDetector()
        model.fit(X_train)
        assert model.is_fitted
        assert model.model is not None
    
    def test_predict(self, synthetic_data):
        """Test LOF prediction."""
        X_train, X_test, y_test = synthetic_data
        model = LocalOutlierFactorAnomalyDetector()
        model.fit(X_train)
        
        predictions = model.predict(X_test)
        assert predictions.shape == y_test.shape
        assert set(predictions) <= {0, 1}
    
    def test_anomaly_score(self, synthetic_data):
        """Test anomaly scoring."""
        X_train, X_test, _ = synthetic_data
        model = LocalOutlierFactorAnomalyDetector()
        model.fit(X_train)
        
        scores = model.anomaly_score(X_test)
        assert scores.shape[0] == X_test.shape[0]
        assert np.all(np.isfinite(scores))
    
    def test_predict_before_fit(self, synthetic_data):
        """Test prediction before fitting raises error."""
        _, X_test, _ = synthetic_data
        model = LocalOutlierFactorAnomalyDetector()
        
        with pytest.raises(ValueError, match="must be fitted"):
            model.predict(X_test)
    
    def test_save_load(self, synthetic_data, tmp_path):
        """Test model save and load."""
        X_train, X_test, _ = synthetic_data
        model = LocalOutlierFactorAnomalyDetector()
        model.fit(X_train)
        
        # Save
        model_path = tmp_path / "lof_model.pkl"
        model.save(str(model_path))
        assert model_path.exists()
        
        # Load
        model_loaded = LocalOutlierFactorAnomalyDetector()
        model_loaded.load(str(model_path))
        assert model_loaded.is_fitted
        
        # Compare predictions
        pred_original = model.predict(X_test)
        pred_loaded = model_loaded.predict(X_test)
        np.testing.assert_array_equal(pred_original, pred_loaded)


class TestIsolationForestDetector:
    """Test Isolation Forest anomaly detector."""
    
    def test_initialization(self):
        """Test Isolation Forest initialization."""
        model = IsolationForestAnomalyDetector(n_estimators=100, contamination=0.1)
        assert model.n_estimators == 100
        assert model.contamination == 0.1
        assert not model.is_fitted
    
    def test_fit(self, synthetic_data):
        """Test Isolation Forest fitting."""
        X_train, _, _ = synthetic_data
        model = IsolationForestAnomalyDetector()
        model.fit(X_train)
        assert model.is_fitted
    
    def test_predict(self, synthetic_data):
        """Test Isolation Forest prediction."""
        X_train, X_test, y_test = synthetic_data
        model = IsolationForestAnomalyDetector()
        model.fit(X_train)
        
        predictions = model.predict(X_test)
        assert predictions.shape == y_test.shape
        assert set(predictions) <= {0, 1}
    
    def test_save_load(self, synthetic_data, tmp_path):
        """Test model save and load."""
        X_train, X_test, _ = synthetic_data
        model = IsolationForestAnomalyDetector()
        model.fit(X_train)
        
        model_path = tmp_path / "iforest_model.pkl"
        model.save(str(model_path))
        
        model_loaded = IsolationForestAnomalyDetector()
        model_loaded.load(str(model_path))
        
        pred_original = model.predict(X_test)
        pred_loaded = model_loaded.predict(X_test)
        np.testing.assert_array_equal(pred_original, pred_loaded)


class TestEllipticEnvelopeDetector:
    """Test Elliptic Envelope anomaly detector."""
    
    def test_initialization(self):
        """Test Elliptic Envelope initialization."""
        model = EllipticEnvelopeAnomalyDetector(contamination=0.1)
        assert model.contamination == 0.1
        assert not model.is_fitted
    
    def test_fit(self, synthetic_data):
        """Test Elliptic Envelope fitting."""
        X_train, _, _ = synthetic_data
        model = EllipticEnvelopeAnomalyDetector()
        model.fit(X_train)
        assert model.is_fitted
    
    def test_predict(self, synthetic_data):
        """Test Elliptic Envelope prediction."""
        X_train, X_test, y_test = synthetic_data
        model = EllipticEnvelopeAnomalyDetector()
        model.fit(X_train)
        
        predictions = model.predict(X_test)
        assert predictions.shape == y_test.shape
        assert set(predictions) <= {0, 1}


class TestCreateDetector:
    """Test detector factory function."""
    
    def test_create_lof(self):
        """Test creating LOF detector."""
        model = create_detector('lof', contamination=0.1)
        assert isinstance(model, LocalOutlierFactorAnomalyDetector)
        assert model.contamination == 0.1
    
    def test_create_isolation_forest(self):
        """Test creating Isolation Forest detector."""
        model = create_detector('isolation_forest', contamination=0.1)
        assert isinstance(model, IsolationForestAnomalyDetector)
    
    def test_create_elliptic_envelope(self):
        """Test creating Elliptic Envelope detector."""
        model = create_detector('elliptic_envelope', contamination=0.1)
        assert isinstance(model, EllipticEnvelopeAnomalyDetector)
    
    def test_create_unknown_method(self):
        """Test creating unknown method raises error."""
        with pytest.raises(ValueError, match="Unknown method"):
            create_detector('unknown_method')


class TestUnsupervisedPreprocessor:
    """Test unsupervised preprocessor."""
    
    def test_initialization(self):
        """Test preprocessor initialization."""
        preprocessor = UnsupervisedPreprocessor(n_components=10, apply_pca=True)
        assert preprocessor.n_components == 10
        assert preprocessor.apply_pca
        assert not preprocessor.is_fitted
    
    def test_fit_transform(self, synthetic_data):
        """Test fit_transform."""
        X_train, _, _ = synthetic_data
        preprocessor = UnsupervisedPreprocessor(n_components=10)
        X_transformed = preprocessor.fit_transform(X_train)
        
        assert preprocessor.is_fitted
        assert X_transformed.shape[0] == X_train.shape[0]
        assert X_transformed.shape[1] == 10
    
    def test_transform(self, synthetic_data):
        """Test transform."""
        X_train, X_test, _ = synthetic_data
        preprocessor = UnsupervisedPreprocessor(n_components=10)
        preprocessor.fit(X_train)
        
        X_transformed = preprocessor.transform(X_test)
        assert X_transformed.shape[0] == X_test.shape[0]
        assert X_transformed.shape[1] == 10
    
    def test_without_pca(self, synthetic_data):
        """Test preprocessor without PCA."""
        X_train, _, _ = synthetic_data
        preprocessor = UnsupervisedPreprocessor(apply_pca=False)
        X_transformed = preprocessor.fit_transform(X_train)
        
        assert X_transformed.shape == X_train.shape
    
    def test_save_load(self, synthetic_data, tmp_path):
        """Test preprocessor save and load."""
        X_train, X_test, _ = synthetic_data
        preprocessor = UnsupervisedPreprocessor(n_components=10)
        preprocessor.fit(X_train)
        
        # Save
        preprocessor_path = tmp_path / "preprocessor.pkl"
        preprocessor.save(str(preprocessor_path))
        assert preprocessor_path.exists()
        
        # Load
        preprocessor_loaded = UnsupervisedPreprocessor.load(str(preprocessor_path))
        assert preprocessor_loaded.is_fitted
        
        # Compare transformations
        X_orig = preprocessor.transform(X_test)
        X_loaded = preprocessor_loaded.transform(X_test)
        np.testing.assert_array_almost_equal(X_orig, X_loaded)


class TestEvaluationFunctions:
    """Test evaluation functions."""
    
    def test_evaluate_anomaly_detector(self, synthetic_data):
        """Test evaluation function."""
        X_train, X_test, y_test = synthetic_data
        
        # Train model
        model = LocalOutlierFactorAnomalyDetector()
        model.fit(X_train)
        
        # Get predictions
        y_pred = model.predict(X_test)
        y_score = model.anomaly_score(X_test)
        
        # Evaluate
        metrics = evaluate_anomaly_detector(y_test, y_pred, y_score, "LOF")
        
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1_score' in metrics
        assert 'roc_auc' in metrics
        
        # Check metric ranges
        assert 0 <= metrics['accuracy'] <= 1
        assert 0 <= metrics['f1_score'] <= 1
        assert 0 <= metrics['roc_auc'] <= 1
    
    def test_model_comparator(self, synthetic_data):
        """Test model comparator."""
        X_train, X_test, y_test = synthetic_data
        
        # Train models
        lof_model = LocalOutlierFactorAnomalyDetector()
        lof_model.fit(X_train)
        
        iforest_model = IsolationForestAnomalyDetector()
        iforest_model.fit(X_train)
        
        # Compare
        comparator = ModelComparator()
        comparator.add_model('LOF', lof_model, X_test, y_test)
        comparator.add_model('IForest', iforest_model, X_test, y_test)
        
        comparison_df = comparator.get_comparison()
        assert len(comparison_df) == 2
        assert 'roc_auc' in comparison_df.columns
        
        best_model = comparator.get_best_model('roc_auc')
        assert best_model in ['LOF', 'IForest']


class TestEndToEndPipeline:
    """Test complete end-to-end pipeline."""
    
    def test_full_pipeline(self, synthetic_data):
        """Test full pipeline from data to evaluation."""
        X_train, X_test, y_test = synthetic_data
        
        # Step 1: Preprocess
        preprocessor = UnsupervisedPreprocessor(n_components=10)
        X_train_proc = preprocessor.fit_transform(X_train)
        X_test_proc = preprocessor.transform(X_test)
        
        # Step 2: Train
        model = LocalOutlierFactorAnomalyDetector(contamination=0.1)
        model.fit(X_train_proc)
        
        # Step 3: Predict
        y_pred = model.predict(X_test_proc)
        y_score = model.anomaly_score(X_test_proc)
        
        # Step 4: Evaluate
        metrics = evaluate_anomaly_detector(y_test, y_pred, y_score, "LOF")
        
        # Step 5: Verify performance (should be better than random)
        assert metrics['roc_auc'] > 0.5  # Better than random
        assert metrics['accuracy'] > 0.5
    
    def test_pipeline_with_save_load(self, synthetic_data, tmp_path):
        """Test pipeline with model saving and loading."""
        X_train, X_test, y_test = synthetic_data
        
        # Train
        preprocessor = UnsupervisedPreprocessor(n_components=10)
        X_train_proc = preprocessor.fit_transform(X_train)
        
        model = LocalOutlierFactorAnomalyDetector()
        model.fit(X_train_proc)
        
        # Save
        model_path = tmp_path / "model.pkl"
        preprocessor_path = tmp_path / "preprocessor.pkl"
        model.save(str(model_path))
        preprocessor.save(str(preprocessor_path))
        
        # Load
        model_loaded = LocalOutlierFactorAnomalyDetector()
        model_loaded.load(str(model_path))
        preprocessor_loaded = UnsupervisedPreprocessor.load(str(preprocessor_path))
        
        # Predict with loaded models
        X_test_proc = preprocessor_loaded.transform(X_test)
        y_pred = model_loaded.predict(X_test_proc)
        
        # Verify
        assert y_pred.shape == y_test.shape
        assert set(y_pred) <= {0, 1}
