#!/usr/bin/env python3
"""
Enhanced demo script showcasing all features of the audio anomaly detection system.
"""

import sys
import os
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
    DATA_DIR = "/home/codespace/.cache/kagglehub/datasets/vuppalaadithyasairam/anomaly-detection-from-sound-data-fan/versions/1/dev_data_fan/train"
    
    print("\n[1/6] Initializing components...")
    feature_extractor = AudioFeatureExtractor(
        sr=SAMPLE_RATE,
        n_mels=N_MELS,
        n_mfcc=N_MFCC
    )
    
    data_processor = AudioDataProcessor(sr=SAMPLE_RATE)
    evaluator = ModelEvaluator()
    
    # Initialisierung der Arrays
    X_train, y_train, X_val, y_val, X_test, y_test = [], [], [], [], [], []
    use_synthetic = False
    dataset = []  # Initialisierung, damit die Variable immer existiert
    # Debug-Ausgabe: Dataset-Laden
    print("[DEBUG] Lade Dataset...")
    print(f"[DEBUG] DATA_DIR: {DATA_DIR}")
    print(f"[DEBUG] use_synthetic: {use_synthetic}")
    print(f"[DEBUG] dataset initial length: {len(dataset)}")
    if not use_synthetic:
        # Hier sollte das echte Dataset geladen werden
        dataset = data_processor.load_dataset(DATA_DIR)
        print(f"[DEBUG] Nach Laden: {len(dataset)} Dateien im Dataset.")
        if len(dataset) > 0:
            print(f"[DEBUG] Beispiel-Dateipfad: {dataset[0][0]}")
            print(f"[DEBUG] Beispiel-Label: {dataset[0][2]}")
        # Begrenzung direkt nach Laden
        MAX_FILES = 10
        if len(dataset) > MAX_FILES:
            print(f"[DEBUG] Begrenze Dataset auf {MAX_FILES} Dateien.")
            dataset = dataset[:MAX_FILES]
        print(f"[DEBUG] Nach Begrenzung: {len(dataset)} Dateien im Dataset.")
        # Split und Debug-Ausgabe
        train_data, val_data, test_data = data_processor.split_dataset(dataset, train_ratio=0.7, val_ratio=0.15)
        print(f"[DEBUG] Split-Ergebnis: train_data={len(train_data)}, val_data={len(val_data)}, test_data={len(test_data)}")
        if len(train_data) > 0:
            print(f"[DEBUG] Beispiel-Train-Dateipfad: {train_data[0][0]}")
            print(f"[DEBUG] Beispiel-Train-Label: {train_data[0][2]}")
    else:
        print("[DEBUG] Synthetic-Daten werden verwendet.")
    print("\n[3/6] Splitting dataset...")
    train_data, val_data, test_data = data_processor.split_dataset(
        dataset, train_ratio=0.7, val_ratio=0.15
    )
    # Debug-Ausgabe: Split-Ergebnis
    print(f"[DEBUG] Split-Ergebnis: train_data={len(train_data)}, val_data={len(val_data)}, test_data={len(test_data)}")
    print("\n[4/6] Extracting enhanced features...")
    # Debug-Ausgabe: Feature-Extraktion
    print("[DEBUG] Starte Feature-Extraktion für Trainingsdaten...")
    MAX_FILES = 10
    if not use_synthetic and len(dataset) > MAX_FILES:
        print(f"[DEBUG] Begrenze Dataset auf {MAX_FILES} Dateien.")
        dataset = dataset[:MAX_FILES]
    for i, (filepath, audio, label) in enumerate(train_data):
        features = feature_extractor.extract_features(audio, enhanced=True)
        if features is not None:
            print(f"[DEBUG] Features extrahiert für {filepath} (Index {i})")
            feat_array = np.array([features[k] for k in sorted(features.keys())])
            X_train.append(feat_array)
            y_train.append(1 if label == "anomaly" else 0 if label == "normal" else label)
        else:
            print(f"[DEBUG] Keine Features für {filepath} (Index {i})")
    for filepath, audio, label in val_data:
        features = feature_extractor.extract_features(audio, enhanced=True)
        if features is not None:
            feat_array = np.array([features[k] for k in sorted(features.keys())])
            X_val.append(feat_array)
            y_val.append(1 if label == "anomaly" else 0 if label == "normal" else label)
    for filepath, audio, label in test_data:
        features = feature_extractor.extract_features(audio, enhanced=True)
        if features is not None:
            feat_array = np.array([features[k] for k in sorted(features.keys())])
            X_test.append(feat_array)
            y_test.append(1 if label == "anomaly" else 0 if label == "normal" else label)
    # Umwandlung in numpy-Arrays
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_val = np.array(X_val)
    y_val = np.array(y_val)
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    # Fehlerbehandlung nur einmal
    if X_train.size == 0:
        print("Fehler: Keine Features extrahiert. Prüfe die Audiodatei und die Extraktionslogik.")
        return
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
