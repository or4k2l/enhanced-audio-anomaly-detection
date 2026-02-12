import numpy as np
import sys
from pathlib import Path
import tempfile

# src-Verzeichnis für Importe verfügbar machen
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from audio_anom import (
    RandomForestAnomalyDetector,
    XGBoostAnomalyDetector,
)


class TestXGBoostAnomalyDetector:
    def test_initialization(self):
        detector = XGBoostAnomalyDetector(random_state=42)
        assert detector.random_state == 42
        assert not detector.is_fitted

    def test_fit(self):
        detector = XGBoostAnomalyDetector()
        X = np.random.randn(100, 20)
        y = np.random.randint(0, 2, 100)
        detector.fit(X, y)
        assert detector.is_fitted

    def test_predict(self):
        detector = XGBoostAnomalyDetector()
        X_train = np.random.randn(100, 20)
        y_train = np.random.randint(0, 2, 100)
        detector.fit(X_train, y_train)
        X_test = np.random.randn(10, 20)
        predictions = detector.predict(X_test)
        assert predictions.shape[0] == 10
        assert set(predictions).issubset({0, 1})

    def test_predict_unfitted(self):
        detector = XGBoostAnomalyDetector()
        X = np.random.randn(10, 20)
        try:
            detector.predict(X)
            assert False, "Sollte Exception werfen"
        except Exception:
            pass

    def test_predict_proba(self):
        detector = XGBoostAnomalyDetector()
        X_train = np.random.randn(100, 20)
        y_train = np.random.randint(0, 2, 100)
        detector.fit(X_train, y_train)
        X_test = np.random.randn(10, 20)
        proba = detector.predict_proba(X_test)
        assert proba.shape[0] == 10

    def test_save_load(self):
        detector = XGBoostAnomalyDetector(random_state=42)
        X_train = np.random.randn(100, 20)
        y_train = np.random.randint(0, 2, 100)
        detector.fit(X_train, y_train)
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            model_path = f.name
        try:
            detector.save(model_path)
            detector2 = XGBoostAnomalyDetector()
            detector2.load(model_path)
            assert detector2.is_fitted
            assert detector2.random_state == 42
            X_test = np.random.randn(10, 20)
            pred1 = detector.predict(X_test)
            pred2 = detector2.predict(X_test)
            np.testing.assert_array_equal(pred1, pred2)
        finally:
            Path(model_path).unlink()

    def test_save_unfitted_raises(self):
        detector = XGBoostAnomalyDetector()
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            model_path = f.name
        try:
            try:
                detector.save(model_path)
                assert False, "Sollte Exception werfen"
            except Exception:
                pass
        finally:
            Path(model_path).unlink(missing_ok=True)

    def test_evaluate(self):
        detector = XGBoostAnomalyDetector()
        X_train = np.random.randn(100, 20)
        y_train = np.random.randint(0, 2, 100)
        detector.fit(X_train, y_train)
        X_test = np.random.randn(50, 20)
        pred = detector.predict(X_test)
        assert pred.shape[0] == 50
        assert set(pred).issubset({0, 1})


class TestRandomForestAnomalyDetector:
    def test_initialization(self):
        detector = RandomForestAnomalyDetector(random_state=42)
        assert detector.random_state == 42
        assert not detector.is_fitted

    def test_fit(self):
        detector = RandomForestAnomalyDetector()
        X = np.random.randn(100, 20)
        y = np.random.randint(0, 2, 100)
        detector.fit(X, y)
        assert detector.is_fitted

    def test_predict(self):
        detector = RandomForestAnomalyDetector()
        X_train = np.random.randn(100, 20)
        y_train = np.random.randint(0, 2, 100)
        detector.fit(X_train, y_train)
        X_test = np.random.randn(10, 20)
        predictions = detector.predict(X_test)
        assert predictions.shape[0] == 10
        assert set(predictions).issubset({0, 1})

    def test_predict_unfitted(self):
        detector = RandomForestAnomalyDetector()
        X = np.random.randn(10, 20)
        try:
            detector.predict(X)
            assert False, "Sollte Exception werfen"
        except Exception:
            pass

    def test_predict_proba(self):
        detector = RandomForestAnomalyDetector()
        X_train = np.random.randn(100, 20)
        y_train = np.random.randint(0, 2, 100)
        detector.fit(X_train, y_train)
        X_test = np.random.randn(10, 20)
        proba = detector.predict_proba(X_test)
        assert proba.shape[0] == 10

    def test_save_load(self):
        detector = RandomForestAnomalyDetector(random_state=42)
        X_train = np.random.randn(100, 20)
        y_train = np.random.randint(0, 2, 100)
        detector.fit(X_train, y_train)
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            model_path = f.name
        try:
            detector.save(model_path)
            detector2 = RandomForestAnomalyDetector()
            detector2.load(model_path)
            assert detector2.is_fitted
            assert detector2.random_state == 42
            X_test = np.random.randn(10, 20)
            pred1 = detector.predict(X_test)
            pred2 = detector2.predict(X_test)
            np.testing.assert_array_equal(pred1, pred2)
        finally:
            Path(model_path).unlink()

    def test_save_unfitted_raises(self):
        detector = RandomForestAnomalyDetector()
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            model_path = f.name
        try:
            try:
                detector.save(model_path)
                assert False, "Sollte Exception werfen"
            except Exception:
                pass
        finally:
            Path(model_path).unlink(missing_ok=True)

    def test_evaluate(self):
        detector = RandomForestAnomalyDetector()
        X_train = np.random.randn(100, 20)
        y_train = np.random.randint(0, 2, 100)
        detector.fit(X_train, y_train)
        X_test = np.random.randn(50, 20)
        pred = detector.predict(X_test)
        assert pred.shape[0] == 50
        assert set(pred).issubset({0, 1})
