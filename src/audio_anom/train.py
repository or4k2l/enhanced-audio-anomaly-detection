"""Training script for audio anomaly detection with multiple models."""

import argparse
import os
import numpy as np

from audio_anom import (
    AudioFeatureExtractor,
    AudioDataProcessor,
    RandomForestAnomalyDetector,
    XGBoostAnomalyDetector,
    AutoencoderAnomalyDetector,
    ModelEvaluator,
)


def train(args):
    """Train multiple anomaly detection models with comprehensive evaluation."""
    print("=" * 80)
    print("AUDIO ANOMALY DETECTION TRAINING")
    print("=" * 80)

    # Initialize components
    print("\n[1/8] Initializing components...")
    feature_extractor = AudioFeatureExtractor(
        sr=args.sample_rate,
        n_mels=args.n_mels,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        n_mfcc=args.n_mfcc,
    )

    data_processor = AudioDataProcessor(sr=args.sample_rate, duration=args.duration)
    evaluator = ModelEvaluator()

    # Load dataset
    print(f"\n[2/8] Loading dataset from {args.data_dir}...")
    if not os.path.exists(args.data_dir):
        print(f"Error: Data directory '{args.data_dir}' does not exist")
        print("Please prepare the dataset following docs/DATASET.md")
        return

    dataset = data_processor.load_dataset(args.data_dir, args.file_pattern)
    print(f"Loaded {len(dataset)} audio files")

    if len(dataset) == 0:
        print("Error: No audio files found in the specified directory")
        return

    # Split dataset
    print("\n[3/8] Splitting dataset...")
    train_data, val_data, test_data = data_processor.split_dataset(
        dataset, train_ratio=args.train_ratio, val_ratio=args.val_ratio
    )
    print(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)} samples")

    # Extract features
    print("\n[4/8] Extracting enhanced features...")
    
    # Extract features using the enhanced method
    X_train, y_train = [], []
    for audio, label in train_data:
        features = feature_extractor.extract_features(audio, enhanced=args.enhanced_features)
        if features is not None:
            # Convert dict to array (sorted keys for consistency)
            feat_array = np.array([features[k] for k in sorted(features.keys())])
            X_train.append(feat_array)
            y_train.append(label)
    
    X_val, y_val = [], []
    for audio, label in val_data:
        features = feature_extractor.extract_features(audio, enhanced=args.enhanced_features)
        if features is not None:
            feat_array = np.array([features[k] for k in sorted(features.keys())])
            X_val.append(feat_array)
            y_val.append(label)
    
    X_test, y_test = [], []
    for audio, label in test_data:
        features = feature_extractor.extract_features(audio, enhanced=args.enhanced_features)
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
    
    print(f"Feature shape: {X_train.shape}")
    print(f"Label distribution - Train: {np.bincount(y_train)}")

    # Train models
    print("\n[5/8] Training models...")
    models = {}
    results = []

    # Random Forest
    if 'rf' in args.models:
        print("\n[5.1] Training Random Forest...")
        rf_detector = RandomForestAnomalyDetector(
            random_state=args.random_state,
            n_splits=args.n_splits
        )
        rf_detector.fit(
            X_train, y_train,
            use_pca=args.use_pca,
            pca_variance=args.pca_variance,
            use_smote=args.use_smote,
            verbose=1
        )
        models['Random Forest'] = rf_detector

    # XGBoost
    if 'xgb' in args.models:
        print("\n[5.2] Training XGBoost...")
        xgb_detector = XGBoostAnomalyDetector(
            random_state=args.random_state,
            n_splits=args.n_splits
        )
        xgb_detector.fit(
            X_train, y_train,
            use_pca=args.use_pca,
            pca_variance=args.pca_variance,
            use_smote=args.use_smote,
            verbose=1
        )
        models['XGBoost'] = xgb_detector

    # Autoencoder
    if 'ae' in args.models:
        try:
            print("\n[5.3] Training Autoencoder...")
            ae_detector = AutoencoderAnomalyDetector(
                encoding_dim=args.encoding_dim,
                random_state=args.random_state
            )
            ae_detector.fit(
                X_train, y_train,
                use_pca=args.use_pca,
                pca_variance=args.pca_variance,
                epochs=args.epochs,
                batch_size=args.batch_size,
                verbose=0
            )
            print(f"  Autoencoder trained successfully")
            models['Autoencoder'] = ae_detector
        except ImportError as e:
            print(f"  Skipping Autoencoder: {e}")

    # Evaluate models
    print("\n[6/8] Evaluating models...")
    
    print("\n" + "=" * 80)
    print("VALIDATION SET RESULTS")
    print("=" * 80)
    
    for name, model in models.items():
        y_pred = model.predict(X_val)
        
        if hasattr(model, 'predict_proba'):
            y_prob = model.predict_proba(X_val)[:, 1]
        elif hasattr(model, 'predict_scores'):
            y_prob = model.predict_scores(X_val)
        else:
            y_prob = None
        
        evaluator.print_evaluation_report(y_val, y_pred, model_name=name)

    print("\n" + "=" * 80)
    print("TEST SET RESULTS")
    print("=" * 80)
    
    models_results = {}
    for name, model in models.items():
        y_pred = model.predict(X_test)
        
        if hasattr(model, 'predict_proba'):
            y_prob = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, 'predict_scores'):
            y_prob = model.predict_scores(X_test)
        else:
            y_prob = None
        
        metrics = evaluator.evaluate_model(y_test, y_pred, y_prob, name)
        results.append(metrics)
        
        models_results[name] = {
            'y_pred': y_pred,
            'y_prob': y_prob,
            'model': model.model if hasattr(model, 'model') else model,
            'pca': model.pca if hasattr(model, 'pca') else None,
        }
        
        evaluator.print_evaluation_report(y_test, y_pred, model_name=name)

    # Compare models
    print("\n[7/8] Model comparison...")
    df_results = evaluator.compare_models(results)

    # Create visualizations
    if args.visualize:
        print("\n[8/8] Creating visualizations...")
        try:
            evaluator.create_comprehensive_report(
                models_results,
                y_test,
                save_dir=args.output_dir if args.save_plots else None
            )
        except Exception as e:
            print(f"Warning: Could not create visualizations: {e}")

    # Save best model
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Find best model based on F1-Score
        best_idx = df_results['F1-Score'].idxmax()
        best_model_name = df_results.loc[best_idx, 'Model']
        best_model = models[best_model_name]
        
        model_path = os.path.join(args.output_dir, f"best_model_{best_model_name.replace(' ', '_').lower()}.pkl")
        best_model.save(model_path)
        print(f"\nBest model ({best_model_name}) saved to {model_path}")
        
        # Save all models if requested
        if args.save_all_models:
            for name, model in models.items():
                model_path = os.path.join(args.output_dir, f"model_{name.replace(' ', '_').lower()}.pkl")
                model.save(model_path)
            print(f"All models saved to {args.output_dir}")

    print("\n" + "=" * 80)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print(f"\nBest model: {df_results.loc[df_results['F1-Score'].idxmax(), 'Model']}")
    print(f"Test F1-Score: {df_results['F1-Score'].max():.4f}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Train enhanced audio anomaly detection models"
    )

    # Data arguments
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/pump",
        help="Directory containing audio files",
    )
    parser.add_argument(
        "--file-pattern", type=str, default="*.wav", help="Audio file pattern"
    )
    parser.add_argument(
        "--output-dir", type=str, default="models", help="Output directory for models"
    )

    # Audio processing arguments
    parser.add_argument(
        "--sample-rate", type=int, default=16000, help="Target sample rate"
    )
    parser.add_argument(
        "--duration", type=float, default=None, help="Target duration in seconds"
    )

    # Feature extraction arguments
    parser.add_argument("--n-mels", type=int, default=128, help="Number of mel bands")
    parser.add_argument("--n-fft", type=int, default=1024, help="FFT window size")
    parser.add_argument("--hop-length", type=int, default=512, help="Hop length")
    parser.add_argument("--n-mfcc", type=int, default=20, help="Number of MFCC coefficients")
    parser.add_argument(
        "--enhanced-features",
        action="store_true",
        default=True,
        help="Use enhanced feature extraction"
    )

    # Training arguments
    parser.add_argument(
        "--train-ratio", type=float, default=0.7, help="Training data ratio"
    )
    parser.add_argument(
        "--val-ratio", type=float, default=0.15, help="Validation data ratio"
    )
    parser.add_argument(
        "--random-state", type=int, default=42, help="Random state for reproducibility"
    )
    parser.add_argument(
        "--n-splits", type=int, default=5, help="Number of cross-validation splits"
    )

    # Model selection
    parser.add_argument(
        "--models",
        nargs="+",
        default=["rf", "xgb", "ae"],
        choices=["rf", "xgb", "ae"],
        help="Models to train (rf=Random Forest, xgb=XGBoost, ae=Autoencoder)"
    )

    # Model-specific arguments
    parser.add_argument(
        "--use-pca", action="store_true", default=True, help="Use PCA for dimensionality reduction"
    )
    parser.add_argument(
        "--pca-variance", type=float, default=0.95, help="PCA variance to retain"
    )
    parser.add_argument(
        "--use-smote", action="store_true", default=True, help="Use SMOTE for class balancing"
    )
    parser.add_argument(
        "--encoding-dim", type=int, default=10, help="Autoencoder encoding dimension"
    )
    parser.add_argument(
        "--epochs", type=int, default=50, help="Number of epochs for autoencoder"
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Batch size for autoencoder"
    )

    # Visualization arguments
    parser.add_argument(
        "--visualize", action="store_true", default=True, help="Create visualizations"
    )
    parser.add_argument(
        "--save-plots", action="store_true", help="Save plots to disk"
    )
    parser.add_argument(
        "--save-all-models", action="store_true", help="Save all trained models"
    )

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
