import json

# Assume RandomForestAnomalyDetector, XGBoostAnomalyDetector, and AutoencoderAnomalyDetector are correctly imported

def load_model(model_file):
    with open(model_file, 'r') as f:
        model_info = json.load(f)
    model_type = model_info['model_type']
    
    if model_type == 'RandomForest':
        return RandomForestAnomalyDetector()
    elif model_type == 'XGBoost':
        return XGBoostAnomalyDetector()
    elif model_type == 'Autoencoder':
        return AutoencoderAnomalyDetector()
    else:
        raise ValueError(f"Unknown model type: {model_type}")

# Update line 36
# detector = AnomalyDetector() 
# is replaced with:
# Assuming model_file is defined earlier in the code
# Replace with correct model instantiation
model_file = 'path_to_model.json' # You should provide the correct path to your model file

detector = load_model(model_file)

# Line 58: Remove the call to the non-existent decision_function method

