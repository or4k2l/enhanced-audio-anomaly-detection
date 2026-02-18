import joblib
from pathlib import Path

# Update all joblib.dump() and joblib.load() calls to convert Path objects to strings

def export_model_package(model, output_file: Path):
    joblib.dump(model, str(output_file))  # Convert Path to string


def load_model_package(input_file: Path):
    return joblib.load(str(input_file))  # Convert Path to string
