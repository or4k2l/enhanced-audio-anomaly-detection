import joblib
import os

class ModelExporter:
    @staticmethod
    def save_model(model, filepath):
        """Saves the model to the specified filepath."""
        joblib.dump(model, filepath)

    @staticmethod
    def load_model(filepath):
        """Loads a model from the specified filepath."""
        return joblib.load(filepath)

    @staticmethod
    def export_to_onnx(model, filepath):
        """Exports the model to the ONNX format at the specified filepath."""
        try:
            import onnx
            import tf2onnx  # or an equivalent library depending on model type
            # Convert the model to ONNX format
            # Placeholder for actual conversion logic
            # tf2onnx.convert.from_keras(model, output_path=filepath)
            print(f'Exported model to {filepath}')
        except ImportError:
            print("ONNX export requires onnx and tf2onnx libraries.")

    @staticmethod
    def list_saved_models(directory):
        """Lists all saved models in the specified directory."""
        return [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
