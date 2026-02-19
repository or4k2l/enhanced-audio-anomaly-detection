"""Example: Embedding-Based Anomaly Detection for Audio.

This example demonstrates:
1. Feature extraction from audio using RobustFeatureExtractor
2. Training different anomaly detectors
3. Anomaly detection and scoring
4. Comparison of different methods
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from audio_anom import (
    RobustFeatureExtractor,
    MahalanobisDetector,
    KNNDetector,
    EmbeddingIsolationForestDetector,
    EnsembleDetector,
    create_embedding_detector,
)


def generate_synthetic_audio(n_samples=100, duration=1.0, sr=22050, anomaly_ratio=0.1):
    """Generate synthetic audio data for demonstration.
    
    Args:
        n_samples: Number of audio samples
        duration: Duration of each audio in seconds
        sr: Sample rate
        anomaly_ratio: Ratio of anomalies
        
    Returns:
        audio_list: List of audio signals
        labels: Binary labels (0=normal, 1=anomaly)
    """
    audio_list = []
    labels = []
    
    n_anomalies = int(n_samples * anomaly_ratio)
    n_normal = n_samples - n_anomalies
    
    # Generate normal audio (sine waves with low frequency)
    for _ in range(n_normal):
        t = np.linspace(0, duration, int(sr * duration))
        freq = np.random.uniform(100, 500)
        audio = np.sin(2 * np.pi * freq * t)
        audio += np.random.randn(len(audio)) * 0.1  # Add noise
        audio_list.append(audio)
        labels.append(0)
    
    # Generate anomalous audio (sine waves with high frequency + noise)
    for _ in range(n_anomalies):
        t = np.linspace(0, duration, int(sr * duration))
        freq = np.random.uniform(2000, 5000)
        audio = np.sin(2 * np.pi * freq * t)
        audio += np.random.randn(len(audio)) * 0.5  # More noise
        audio_list.append(audio)
        labels.append(1)
    
    # Shuffle
    indices = np.random.permutation(n_samples)
    audio_list = [audio_list[i] for i in indices]
    labels = np.array([labels[i] for i in indices])
    
    return audio_list, labels


def main():
    """Main example workflow."""
    print("=" * 70)
    print("Embedding-Based Anomaly Detection Example")
    print("=" * 70)
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # 1. Generate synthetic data
    print("\n1. Generating synthetic audio data...")
    audio_list, labels = generate_synthetic_audio(
        n_samples=200,
        duration=1.0,
        sr=22050,
        anomaly_ratio=0.1
    )
    
    # Split into train and test
    n_train = 150
    train_audio = audio_list[:n_train]
    train_labels = labels[:n_train]
    test_audio = audio_list[n_train:]
    test_labels = labels[n_train:]
    
    print(f"  Training samples: {n_train}")
    print(f"  Test samples: {len(test_audio)}")
    print(f"  Training anomalies: {np.sum(train_labels)}")
    print(f"  Test anomalies: {np.sum(test_labels)}")
    
    # 2. Extract features
    print("\n2. Extracting features...")
    extractor = RobustFeatureExtractor(sr=22050, n_mels=128, n_mfcc=13)
    
    train_features = extractor.extract_features_batch(train_audio)
    test_features = extractor.extract_features_batch(test_audio)
    
    print(f"  Feature dimension: {train_features.shape[1]}")
    
    # 3. Train different detectors
    print("\n3. Training anomaly detectors...")
    
    detectors = {
        'Mahalanobis': MahalanobisDetector(),
        'k-NN': KNNDetector(n_neighbors=5),
        'Isolation Forest': EmbeddingIsolationForestDetector(contamination=0.1),
        'Ensemble': EnsembleDetector(
            methods=['mahalanobis', 'knn', 'isolation_forest'],
            weights=[0.4, 0.35, 0.25]
        ),
    }
    
    # Use only normal samples for training (unsupervised)
    normal_indices = train_labels == 0
    train_normal_features = train_features[normal_indices]
    
    print(f"  Training on {len(train_normal_features)} normal samples...")
    
    results = {}
    for name, detector in detectors.items():
        print(f"    Training {name}...")
        detector.fit(train_normal_features)
        
        # Get scores and predictions on test set
        test_scores = detector.score(test_features)
        test_predictions = detector.predict(test_features)
        
        # Calculate metrics
        tp = np.sum((test_predictions == 1) & (test_labels == 1))
        fp = np.sum((test_predictions == 1) & (test_labels == 0))
        tn = np.sum((test_predictions == 0) & (test_labels == 0))
        fn = np.sum((test_predictions == 0) & (test_labels == 1))
        
        accuracy = (tp + tn) / len(test_labels)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        results[name] = {
            'scores': test_scores,
            'predictions': test_predictions,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'threshold': detector.threshold,
        }
    
    # 4. Display results
    print("\n4. Test Results:")
    print("=" * 70)
    print(f"{'Method':<20} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}")
    print("-" * 70)
    
    for name, result in results.items():
        print(
            f"{name:<20} "
            f"{result['accuracy']:<10.3f} "
            f"{result['precision']:<10.3f} "
            f"{result['recall']:<10.3f} "
            f"{result['f1']:<10.3f}"
        )
    
    print("=" * 70)
    
    # 5. Visualize score distributions
    print("\n5. Generating visualizations...")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()
    
    for idx, (name, result) in enumerate(results.items()):
        ax = axes[idx]
        
        scores = result['scores']
        normal_scores = scores[test_labels == 0]
        anomaly_scores = scores[test_labels == 1]
        
        ax.hist(normal_scores, bins=20, alpha=0.6, label='Normal', color='blue')
        ax.hist(anomaly_scores, bins=20, alpha=0.6, label='Anomaly', color='red')
        ax.axvline(result['threshold'], color='black', linestyle='--', 
                   label=f"Threshold={result['threshold']:.2f}")
        ax.set_xlabel('Anomaly Score')
        ax.set_ylabel('Frequency')
        ax.set_title(f'{name}\nF1-Score: {result["f1"]:.3f}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    output_path = Path(__file__).parent / "embedding_anomaly_results.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  Saved visualization to: {output_path}")
    
    # 6. Save best model
    print("\n6. Saving best model...")
    best_method = max(results.items(), key=lambda x: x[1]['f1'])
    best_name = best_method[0]
    
    model_path = Path(__file__).parent / "best_embedding_detector.pkl"
    detectors[best_name].save(str(model_path))
    print(f"  Saved {best_name} detector to: {model_path}")
    print(f"  Best F1-Score: {best_method[1]['f1']:.3f}")
    
    print("\n" + "=" * 70)
    print("Example completed successfully!")
    print("=" * 70)


if __name__ == '__main__':
    main()
