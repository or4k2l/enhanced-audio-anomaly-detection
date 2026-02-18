class ModelExporter:
    def __init__(self, model_name):
        self.model_name = model_name
        self.saved_models = []

    def save_model(self, model, filepath):
        """Save the given model to the specified filepath."""
        # Implementation for saving the model
        with open(filepath, 'wb') as f:
            pickle.dump(model, f)
        self.saved_models.append(filepath)

    def load_model(self, filepath):
        """Load the model from the specified filepath."""
        # Implementation for loading the model
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        return model

    def export_to_onnx(self, model, filepath):
        """Export the model to ONNX format."""
        # Implementation for exporting the model
        dummy_input = torch.randn(1, 3, 224, 224)  # Example input shape
        torch.onnx.export(model, dummy_input, filepath)

    def list_saved_models(self):
        """Return a list of saved models."""
        return self.saved_models
