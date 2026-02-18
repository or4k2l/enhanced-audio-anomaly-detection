# examples/demo.py

from audio_anom import (
    AudioFeatureExtractor,
    AudioDataProcessor,
    RandomForestAnomalyDetector,
    XGBoostAnomalyDetector,
    AutoencoderAnomalyDetector,
    ModelEvaluator
)
import numpy as np

def generate_synthetic_data(num_samples=1000, num_features=10):
    """Generate synthetic audio data for demo purposes."""
    return np.random.rand(num_samples, num_features)

def main():
    try:
        # Step 1: Generate synthetic data
        synthetic_data = generate_synthetic_data()
        print("Synthetic data generated.")

        # Step 2: Extract features from the audio data
        feature_extractor = AudioFeatureExtractor()
        features = feature_extractor.extract_features(synthetic_data)
        print("Features extracted.")

        # Step 3: Process the data
        data_processor = AudioDataProcessor()
        processed_data = data_processor.process(features)
        print("Data processed.")

        # Step 4: Train and evaluate different anomaly detectors
        rf_detector = RandomForestAnomalyDetector()
        xgb_detector = XGBoostAnomalyDetector()
        autoencoder_detector = AutoencoderAnomalyDetector()

        # Assuming we have a method to train these detectors
        rf_detector.train(processed_data)
        xgb_detector.train(processed_data)
        autoencoder_detector.train(processed_data)

        # Evaluate model performance
        evaluator = ModelEvaluator()
        rf_results = evaluator.evaluate(rf_detector, processed_data)
        xgb_results = evaluator.evaluate(xgb_detector, processed_data)
        ae_results = evaluator.evaluate(autoencoder_detector, processed_data)

        # Print results
        print("RF Results:", rf_results)
        print("XGBoost Results:", xgb_results)
        print("Autoencoder Results:", ae_results)

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()