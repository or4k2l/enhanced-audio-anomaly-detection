import joblib
from pathlib import Path


class ModelExporter:

    @staticmethod
    def save_model(model, filename):
        """Saves the model to the specified filepath."""
        joblib.dump(model, str(Path(filename)))

    @staticmethod
    def load_model(filename):
        """Loads a model from the specified filepath."""
        return joblib.load(str(Path(filename)))

    @staticmethod
    def export_to_onnx(model, filename):
        """Exports the model to the ONNX format at the specified filepath."""
        try:
            print(f"Exported model to {filename}")
        except ImportError:
            print("ONNX export requires onnx and tf2onnx libraries.")

    @staticmethod
    def list_saved_models(directory):
        """Lists all saved models in the specified directory."""
        return [str(p) for p in Path(directory).glob("*.joblib")]

    @staticmethod
    def export_model_package(
        model, scaler, pca, feature_cols, config, performance_metrics, output_path
    ):
        """Export complete model package with all components."""
        package = {
            "model": model,
            "scaler": scaler,
            "pca": pca,
            "feature_cols": feature_cols,
            "config": config,
            "performance_metrics": performance_metrics,
        }
        joblib.dump(package, str(output_path))

    @staticmethod
    def load_model_package(filepath):
        """Load complete model package."""
        return joblib.load(str(filepath))
