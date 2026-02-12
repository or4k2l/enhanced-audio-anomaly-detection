import pytest
import numpy as np
from audio_anom.export import ModelExporter

class DummyModel:
    def __init__(self):
        self.model = None

class DummyScaler:
    pass

class DummyPCA:
    pass

def test_export_model_package(tmp_path):
    exporter = ModelExporter()
    model = DummyModel()
    scaler = DummyScaler()
    pca = DummyPCA()
    feature_cols = ["f1", "f2"]
    config = {"param": 42}
    performance_metrics = {"accuracy": 0.9}
    output_path = tmp_path / "model_package.pkl"
    exporter.export_model_package(
        model=model,
        scaler=scaler,
        pca=pca,
        feature_cols=feature_cols,
        config=config,
        performance_metrics=performance_metrics,
        output_path=output_path,
    )
    assert output_path.exists()
