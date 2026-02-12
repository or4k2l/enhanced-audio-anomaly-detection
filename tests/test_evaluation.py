import numpy as np
from audio_anom.evaluation import ModelEvaluator

class TestModelEvaluator:
    def test_evaluate_model_basic(self):
        evaluator = ModelEvaluator()
        y_true = np.array([0, 1, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 0, 1])
        y_prob = np.array([0.1, 0.9, 0.2, 0.3, 0.8])
        metrics = evaluator.evaluate_model(y_true, y_pred, y_prob, model_name="TestModel")
        assert "Accuracy" in metrics
        assert metrics["Accuracy"] >= 0
        assert "Precision" in metrics
        assert "Recall" in metrics
        assert "F1-Score" in metrics

    def test_evaluate_model_no_prob(self):
        evaluator = ModelEvaluator()
        y_true = np.array([0, 1, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 0, 1])
        metrics = evaluator.evaluate_model(y_true, y_pred, y_prob=None, model_name="TestModel")
        assert "Accuracy" in metrics
        assert metrics["Accuracy"] >= 0

    def test_compare_models(self):
        evaluator = ModelEvaluator()
        results = [
            {"Accuracy": 0.8, "Precision": 0.7, "Recall": 0.6, "F1-Score": 0.65, "Model": "A"},
            {"Accuracy": 0.9, "Precision": 0.8, "Recall": 0.7, "F1-Score": 0.75, "Model": "B"},
        ]
        df = evaluator.compare_models(results)
        assert "Model" in df.columns
        assert df.shape[0] == 2
