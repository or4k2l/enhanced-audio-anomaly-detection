import numpy as np
from audio_anom.evaluation import ModelEvaluator

class TestModelEvaluator:
    def test_evaluate_model_basic(self):
        evaluator = ModelEvaluator()
        y_true = np.array([0, 1, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 0, 1])
        y_prob = np.array([0.1, 0.9, 0.2, 0.3, 0.8])
        metrics = evaluator.evaluate_model(y_true, y_pred, y_prob, model_name="TestModel")
        assert "accuracy" in metrics
        assert metrics["accuracy"] >= 0
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics

    def test_evaluate_model_no_prob(self):
        evaluator = ModelEvaluator()
        y_true = np.array([0, 1, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 0, 1])
        metrics = evaluator.evaluate_model(y_true, y_pred, y_prob=None, model_name="TestModel")
        assert "accuracy" in metrics
        assert metrics["accuracy"] >= 0

    def test_compare_models(self):
        evaluator = ModelEvaluator()
        results = [
            {"accuracy": 0.8, "precision": 0.7, "recall": 0.6, "f1": 0.65, "Model": "A"},
            {"accuracy": 0.9, "precision": 0.8, "recall": 0.7, "f1": 0.75, "Model": "B"},
        ]
        df = evaluator.compare_models(results)
        assert "Model" in df.columns
        assert df.shape[0] == 2
