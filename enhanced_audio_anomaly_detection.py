# Enhanced Audio Anomaly Detection
# Filename: enhanced_audio_anomaly_detection.py
# Description: Performs feature extraction, training,
#              evaluation and export of anomaly detection models for pump sounds.

print("=" * 80)
print("AUDIO ANOMALY DETECTION - ENHANCED VERSION")
print("=" * 80)

# ============================================================================
# 0. SETUP & INSTALLATION
# ============================================================================
print("\n[1/10] Installing libraries...")
# Note: In a script you may want to avoid running pip install inline.
# Uncomment and run manually if needed:
# !pip install -q numpy pandas scipy scikit-learn matplotlib seaborn \
#     librosa soundfile xgboost tensorflow

# ============================================================================
# 1. IMPORTS
# ============================================================================
print("\n[2/10] Importing libraries...")

import os
import glob
import warnings
from pathlib import Path
from collections import defaultdict
import joblib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Audio
import librosa
import soundfile as sf
from scipy.fft import fft, fftfreq
from scipy import stats

# Machine Learning
from sklearn.model_selection import (
    StratifiedKFold, GridSearchCV, cross_val_score, 
    cross_validate, train_test_split
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_recall_fscore_support, roc_auc_score, roc_curve,
    precision_recall_curve
)
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

# Deep Learning
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

warnings.filterwarnings('ignore')
sns.set_style('whitegrid')

# ============================================================================
# 2. CONFIGURATION
# ============================================================================
print("\n[3/10] Setting up configuration...")

# Set random seeds for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# Audio parameters
SAMPLE_RATE = 16000
DURATION = 10  # seconds
N_MFCC = 40
N_FFT = 2048
HOP_LENGTH = 512

# Model parameters
TEST_SIZE = 0.2
N_SPLITS = 5

# Neural network parameters
NN_EPOCHS = 50
NN_BATCH_SIZE = 32
NN_VALIDATION_SPLIT = 0.2

print("Configuration complete.")

# ============================================================================
# 3. DATA LOADING
# ============================================================================
print("\n[4/10] Loading dataset...")

def load_audio_files(data_path, sample_rate=SAMPLE_RATE, duration=DURATION):
    """
    Load audio files from the specified directory.
    
    Args:
        data_path: Path to the audio files
        sample_rate: Target sample rate
        duration: Duration to load (in seconds)
        
    Returns:
        List of audio data and corresponding labels
    """
    audio_data = []
    labels = []
    
    if not os.path.exists(data_path):
        print(f"Warning: Data path {data_path} does not exist.")
        return audio_data, labels
    
    # Load normal and anomaly files
    for label_type in ['normal', 'anomaly']:
        file_pattern = os.path.join(data_path, label_type, '*.wav')
        files = glob.glob(file_pattern)
        
        for file_path in files:
            try:
                audio, sr = librosa.load(file_path, sr=sample_rate, duration=duration)
                audio_data.append(audio)
                labels.append(0 if label_type == 'normal' else 1)
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
    
    return audio_data, labels

# ============================================================================
# 4. FEATURE EXTRACTION
# ============================================================================
print("\n[5/10] Extracting features...")

def extract_features(audio, sample_rate=SAMPLE_RATE):
    """
    Extract comprehensive audio features for anomaly detection.
    
    Args:
        audio: Audio signal
        sample_rate: Sample rate of the audio
        
    Returns:
        Dictionary of extracted features
    """
    features = {}
    
    # MFCCs
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=N_MFCC)
    features['mfcc_mean'] = np.mean(mfccs, axis=1)
    features['mfcc_std'] = np.std(mfccs, axis=1)
    
    # Spectral features
    spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sample_rate)[0]
    features['spectral_centroid_mean'] = np.mean(spectral_centroids)
    features['spectral_centroid_std'] = np.std(spectral_centroids)
    
    spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sample_rate)[0]
    features['spectral_rolloff_mean'] = np.mean(spectral_rolloff)
    features['spectral_rolloff_std'] = np.std(spectral_rolloff)
    
    # Zero crossing rate
    zcr = librosa.feature.zero_crossing_rate(audio)[0]
    features['zcr_mean'] = np.mean(zcr)
    features['zcr_std'] = np.std(zcr)
    
    # Chroma features
    chroma = librosa.feature.chroma_stft(y=audio, sr=sample_rate)
    features['chroma_mean'] = np.mean(chroma, axis=1)
    
    # Flatten all features
    feature_vector = []
    for key, value in features.items():
        if isinstance(value, np.ndarray):
            feature_vector.extend(value)
        else:
            feature_vector.append(value)
    
    return np.array(feature_vector)

# ============================================================================
# 5. MODEL TRAINING
# ============================================================================
print("\n[6/10] Training models...")

def train_random_forest(X_train, y_train, X_test, y_test):
    """Train and evaluate Random Forest classifier."""
    print("\nTraining Random Forest...")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED)
    rf_model.fit(X_train, y_train)
    
    y_pred = rf_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Random Forest Accuracy: {accuracy:.4f}")
    
    return rf_model, accuracy

def train_xgboost(X_train, y_train, X_test, y_test):
    """Train and evaluate XGBoost classifier."""
    print("\nTraining XGBoost...")
    xgb_model = XGBClassifier(random_state=RANDOM_SEED, eval_metric='logloss')
    xgb_model.fit(X_train, y_train)
    
    y_pred = xgb_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"XGBoost Accuracy: {accuracy:.4f}")
    
    return xgb_model, accuracy

def build_neural_network(input_dim):
    """Build a neural network for anomaly detection."""
    model = keras.Sequential([
        layers.Dense(128, activation='relu', input_shape=(input_dim,)),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(32, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# ============================================================================
# 6. EVALUATION
# ============================================================================
print("\n[7/10] Evaluating models...")

def evaluate_model(model, X_test, y_test, model_name="Model"):
    """Comprehensive model evaluation."""
    y_pred = model.predict(X_test)
    
    if len(y_pred.shape) > 1:
        y_pred = (y_pred > 0.5).astype(int).flatten()
    
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
    
    print(f"\n{model_name} Evaluation:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

# ============================================================================
# 7. VISUALIZATION
# ============================================================================
print("\n[8/10] Creating visualizations...")

def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix"):
    """Plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.show()

# ============================================================================
# 8. MODEL EXPORT
# ============================================================================
print("\n[9/10] Exporting models...")

def export_model(model, filename, model_type='sklearn'):
    """Export trained model to disk."""
    output_dir = 'models'
    os.makedirs(output_dir, exist_ok=True)
    
    if model_type == 'sklearn':
        filepath = os.path.join(output_dir, f"{filename}.pkl")
        joblib.dump(model, filepath)
    elif model_type == 'tensorflow':
        filepath = os.path.join(output_dir, f"{filename}.h5")
        model.save(filepath)
    
    print(f"Model exported to {filepath}")
    return filepath

# ============================================================================
# 9. MAIN EXECUTION
# ============================================================================
print("\n[10/10] Main execution...")

def main():
    """Main execution function."""
    print("\nStarting Audio Anomaly Detection Pipeline...")
    
    # Example usage - requires actual data path
    data_path = 'data/pump_sounds'
    
    if not os.path.exists(data_path):
        print(f"\nNote: Data path '{data_path}' not found.")
        print("Please prepare your dataset according to the structure:")
        print("  data/pump_sounds/normal/*.wav")
        print("  data/pump_sounds/anomaly/*.wav")
        print("\nPipeline setup complete. Ready to process data when available.")
        return
    
    # Load data
    audio_data, labels = load_audio_files(data_path)
    
    if len(audio_data) == 0:
        print("No audio files found. Please check your data directory.")
        return
    
    print(f"Loaded {len(audio_data)} audio files")
    print(f"Normal samples: {labels.count(0)}")
    print(f"Anomaly samples: {labels.count(1)}")
    
    # Extract features
    print("\nExtracting features from audio files...")
    features = [extract_features(audio) for audio in audio_data]
    X = np.array(features)
    y = np.array(labels)
    
    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
    )
    
    # Train models
    rf_model, rf_acc = train_random_forest(X_train, y_train, X_test, y_test)
    xgb_model, xgb_acc = train_xgboost(X_train, y_train, X_test, y_test)
    
    # Neural network
    print("\nTraining Neural Network...")
    nn_model = build_neural_network(X_train.shape[1])
    nn_model.fit(X_train, y_train, epochs=NN_EPOCHS, batch_size=NN_BATCH_SIZE, 
                 validation_split=NN_VALIDATION_SPLIT, verbose=0)
    
    # Evaluate models
    rf_metrics = evaluate_model(rf_model, X_test, y_test, "Random Forest")
    xgb_metrics = evaluate_model(xgb_model, X_test, y_test, "XGBoost")
    nn_metrics = evaluate_model(nn_model, X_test, y_test, "Neural Network")
    
    # Export best model
    best_model = rf_model if rf_acc >= xgb_acc else xgb_model
    best_name = "random_forest" if rf_acc >= xgb_acc else "xgboost"
    export_model(best_model, f"best_model_{best_name}", 'sklearn')
    export_model(nn_model, "neural_network", 'tensorflow')
    
    print("\n" + "=" * 80)
    print("AUDIO ANOMALY DETECTION PIPELINE COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    main()
