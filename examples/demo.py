print("DEBUG sys.path (nach Import):", sys.path)
print("DEBUG sys.path:", sys.path)
#!/usr/bin/env python3
"""
Enhanced demo script showcasing all features of the audio anomaly detection system.

This script demonstrates:
- Enhanced feature extraction
- Multiple model training (Random Forest, XGBoost, Autoencoder)
- Comprehensive evaluation and visualization
- Model comparison
"""

import sys
import os
print("DEBUG sys.path (ganz am Anfang):", sys.path)
print("DEBUG sys.path (vor insert):", sys.path)
import numpy as np

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src', 'audio_anom')))

from data import AudioDataProcessor
from features import AudioFeatureExtractor
from models import RandomForestAnomalyDetector, XGBoostAnomalyDetector
from evaluation import ModelEvaluator


def main():
    """Run enhanced demo."""
    print("DEBUG: main gestartet")
    print("=" * 80)
    print("ENHANCED AUDIO ANOMALY DETECTION - DEMO")
    print("=" * 80)
    
    # Configuration
    SAMPLE_RATE = 16000
    N_MELS = 128
    N_MFCC = 20
    DATA_DIR = "data/pump"
    
    print("\n[1/6] Initializing components...")
    feature_extractor = AudioFeatureExtractor(
        sr=SAMPLE_RATE,
        n_mels=N_MELS,
        n_mfcc=N_MFCC
    )
    
    data_processor = AudioDataProcessor(sr=SAMPLE_RATE)
    evaluator = ModelEvaluator()
    
    # Check if data directory exists
    if not os.path.exists(DATA_DIR):
        print(f"\n⚠️  Data directory '{DATA_DIR}' not found.")
        print("Please download the dataset following instructions in docs/DATASET.md")
        print("\nRunning with synthetic data for demonstration...")
        use_synthetic = True
    else:
        use_synthetic = False
    
    if use_synthetic:
        # Generate synthetic data for demo
        print("\n[2/6] Generating synthetic data...")
        np.random.seed(42)
        
        # Normal samples (Gaussian noise)
        X_normal = np.random.randn(100, 50)
        y_normal = np.zeros(100)
        
        # Anomaly samples (shifted distribution)
        X_anomaly = np.random.randn(20, 50) + 2
        y_anomaly = np.ones(20)
        
        # Combine and shuffle
        X = np.vstack([X_normal, X_anomaly])
        y = np.hstack([y_normal, y_anomaly])
        
        # Split
        from sklearn.model_selection import train_test_split
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
        )
        
        print(f"Train: {len(X_train)} samples")
        print(f"Val: {len(X_val)} samples")
        print(f"Test: {len(X_test)} samples")
    
    else:
        # Load real data
        print(f"\n[2/6] Loading dataset from {DATA_DIR}...")
        dataset = data_processor.load_dataset(DATA_DIR, "*.wav")
        
        if len(dataset) == 0:
            print("Error: No audio files found")
            return
        
        print(f"Loaded {len(dataset)} audio files")
        
        # Split dataset
        print("\n[3/6] Splitting dataset...")
        train_data, val_data, test_data = data_processor.split_dataset(
            dataset, train_ratio=0.7, val_ratio=0.15
        )
        
        # Extract features
        print("\n[4/6] Extracting enhanced features...")
        X_train, y_train = [], []
        for audio, label in train_data:
            features = feature_extractor.extract_features(audio, enhanced=True)
            if features is not None:
                feat_array = np.array([features[k] for k in sorted(features.keys())])
                X_train.append(feat_array)
                y_train.append(label)
        
        X_val, y_val = [], []
        for audio, label in val_data:
            features = feature_extractor.extract_features(audio, enhanced=True)
            if features is not None:
                feat_array = np.array([features[k] for k in sorted(features.keys())])
                X_val.append(feat_array)
                y_val.append(label)
        
        X_test, y_test = [], []
        for audio, label in test_data:
            features = feature_extractor.extract_features(audio, enhanced=True)
            if features is not None:
                feat_array = np.array([features[k] for k in sorted(features.keys())])
                X_test.append(feat_array)
                y_test.append(label)
        
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        X_val = np.array(X_val)
        y_val = np.array(y_val)
        X_test = np.array(X_test)
        y_test = np.array(y_test)
    
    print(f"\nFeature shape: {X_train.shape}")
    print(f"Label distribution - Train: Normal={np.sum(y_train==0)}, Anomaly={np.sum(y_train==1)}")
    
    # Train models
    step = 5 if use_synthetic else 4
    print(f"\n[{step}/6] Training models...")
    
    models = {}
    
    # Random Forest
    print("\n  Training Random Forest...")
    rf_detector = RandomForestAnomalyDetector(random_state=42, n_splits=3)
    rf_detector.fit(X_train, y_train, use_pca=True, use_smote=True, verbose=0)
    models['Random Forest'] = rf_detector
    print("  ✓ Random Forest trained")
    
    # XGBoost
    print("\n  Training XGBoost...")
    xgb_detector = XGBoostAnomalyDetector(random_state=42, n_splits=3)
    xgb_detector.fit(X_train, y_train, use_pca=True, use_smote=True, verbose=0)
    models['XGBoost'] = xgb_detector
    print("  ✓ XGBoost trained")
    
    # Evaluate models
    step += 1
    print(f"\n[{step}/6] Evaluating models...")
    
    results = []
    models_results = {}
    
    for name, model in models.items():
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        metrics = evaluator.evaluate_model(y_test, y_pred, y_prob, name)
        results.append(metrics)
        
        models_results[name] = {
            'y_pred': y_pred,
            'y_prob': y_prob,
            'model': model.model if hasattr(model, 'model') else model,
            'pca': model.pca if hasattr(model, 'pca') else None,
        }
    
    # Print comparison
    print("\n" + "=" * 80)
    print("MODEL COMPARISON")
    print("=" * 80)
    df_results = evaluator.compare_models(results)
    
    # Best model
    best_model_name = df_results.loc[df_results['F1-Score'].idxmax(), 'Model']
    best_f1 = df_results['F1-Score'].max()
    
    print(f"\n🏆 Best Model: {best_model_name}")
    print(f"   F1-Score: {best_f1:.4f}")
    
    # Create visualizations
    step += 1
    print(f"\n[{step}/6] Creating visualizations...")
    try:
        evaluator.create_comprehensive_report(models_results, y_test, save_dir=None)
        print("✓ Visualizations created successfully")
    except Exception as e:
        print(f"⚠️  Could not create visualizations: {e}")
    
    print("\n" + "=" * 80)
    print("DEMO COMPLETED!")
    print("=" * 80)
    
    if use_synthetic:
        print("\n💡 This demo used synthetic data.")
        print("   For real results, download the dataset following docs/DATASET.md")


if __name__ == "__main__":
    main()
