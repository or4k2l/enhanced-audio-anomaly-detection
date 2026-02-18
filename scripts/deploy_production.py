#!/usr/bin/env python
"""Production deployment script for anomaly detection inference.

This script provides a production-ready interface for real-time anomaly detection:
- Load trained models
- Process audio files
- Generate predictions with confidence scores
- Export results to CSV/JSON
- Alert generation

Usage:
    # Single file inference
    python scripts/deploy_production.py --model models/fan_lof_model.pkl --audio test.wav
    
    # Batch inference
    python scripts/deploy_production.py --model models/fan_lof_model.pkl --audio-dir /path/to/audio/ --output results.csv
    
    # With alerting
    python scripts/deploy_production.py --model models/fan_lof_model.pkl --audio test.wav --alert-threshold 0.8
"""

import argparse
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from audio_anom import (
    AudioFeatureExtractor,
    AudioDataProcessor,
    build_feature_vector,
    setup_logger,
)
from audio_anom.unsupervised_anomaly import BaseUnsupervisedAnomalyDetector, create_detector
from audio_anom.preprocessing_unsupervised import UnsupervisedPreprocessor

logger = setup_logger("deploy_production")


class AnomalyDetectionPipeline:
    """Production pipeline for anomaly detection inference.
    
    This class provides a complete pipeline from audio file to anomaly prediction.
    """
    
    def __init__(
        self,
        model_path: str,
        preprocessor_path: str,
        alert_threshold: float = 0.5,
    ):
        """Initialize the pipeline.
        
        Args:
            model_path: Path to trained model
            preprocessor_path: Path to fitted preprocessor
            alert_threshold: Threshold for anomaly score to trigger alert (0-1)
        """
        logger.info("Initializing production pipeline...")
        
        # Load model
        logger.info(f"Loading model from {model_path}")
        # Detect method from filename
        if 'lof' in str(model_path).lower():
            self.model = create_detector('lof')
        elif 'isolation' in str(model_path).lower() or 'iforest' in str(model_path).lower():
            self.model = create_detector('isolation_forest')
        elif 'elliptic' in str(model_path).lower() or 'envelope' in str(model_path).lower():
            self.model = create_detector('elliptic_envelope')
        else:
            raise ValueError("Could not determine model type from filename")
        
        self.model.load(model_path)
        
        # Load preprocessor
        logger.info(f"Loading preprocessor from {preprocessor_path}")
        self.preprocessor = UnsupervisedPreprocessor.load(preprocessor_path)
        
        # Initialize feature extractor and data processor
        self.feature_extractor = AudioFeatureExtractor()
        self.data_processor = AudioDataProcessor()
        
        self.alert_threshold = alert_threshold
        
        logger.info("Pipeline initialized successfully!")
        
    def process_audio_file(self, audio_path: str) -> Dict[str, any]:
        """Process a single audio file and return prediction.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            result: Dictionary with prediction results
        """
        try:
            # Load audio
            audio, sr = self.data_processor.load_audio(audio_path)
            
            # Extract features
            features = self.feature_extractor.extract_features(audio)
            if features is None:
                return {
                    'file': audio_path,
                    'error': 'Feature extraction failed (audio too short)',
                    'prediction': None,
                    'anomaly_score': None,
                    'confidence': None,
                    'alert': False,
                }
            
            # Build feature vector
            feature_vector = build_feature_vector(features).reshape(1, -1)
            
            # Preprocess
            feature_vector_proc = self.preprocessor.transform(feature_vector)
            
            # Predict
            prediction = self.model.predict(feature_vector_proc)[0]
            anomaly_score = self.model.anomaly_score(feature_vector_proc)[0]
            
            # Normalize score to 0-1 range (higher = more anomalous)
            # Using min-max normalization based on typical score ranges
            normalized_score = float(np.clip(anomaly_score / 10.0, 0, 1))
            
            # Confidence is the absolute distance from decision boundary
            confidence = abs(normalized_score - 0.5) * 2  # 0-1 scale
            
            # Check for alert
            alert = normalized_score >= self.alert_threshold
            
            result = {
                'file': str(Path(audio_path).name),
                'prediction': 'ANOMALY' if prediction == 1 else 'NORMAL',
                'anomaly_score': float(anomaly_score),
                'normalized_score': normalized_score,
                'confidence': float(confidence),
                'alert': alert,
                'timestamp': datetime.now().isoformat(),
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing {audio_path}: {e}")
            return {
                'file': audio_path,
                'error': str(e),
                'prediction': None,
                'anomaly_score': None,
                'confidence': None,
                'alert': False,
            }
    
    def process_batch(self, audio_paths: List[str]) -> List[Dict[str, any]]:
        """Process multiple audio files.
        
        Args:
            audio_paths: List of audio file paths
            
        Returns:
            results: List of prediction results
        """
        logger.info(f"Processing batch of {len(audio_paths)} files...")
        
        results = []
        for i, audio_path in enumerate(audio_paths, 1):
            logger.info(f"Processing {i}/{len(audio_paths)}: {audio_path}")
            result = self.process_audio_file(audio_path)
            results.append(result)
            
            # Log alerts
            if result.get('alert', False):
                logger.warning(f"⚠️  ALERT: Anomaly detected in {result['file']} "
                             f"(score: {result['normalized_score']:.3f})")
        
        logger.info(f"Batch processing complete!")
        return results


def save_results(results: List[Dict], output_path: str, format: str = 'csv'):
    """Save results to file.
    
    Args:
        results: List of prediction results
        output_path: Output file path
        format: Output format ('csv' or 'json')
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if format == 'csv':
        df = pd.DataFrame(results)
        df.to_csv(output_path, index=False)
        logger.info(f"Results saved to {output_path} (CSV)")
    elif format == 'json':
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {output_path} (JSON)")
    else:
        raise ValueError(f"Unknown format: {format}")


def print_result(result: Dict):
    """Pretty print a single result."""
    print("\n" + "="*80)
    print("ANOMALY DETECTION RESULT")
    print("="*80)
    print(f"File:             {result['file']}")
    print(f"Prediction:       {result['prediction']}")
    if result.get('normalized_score') is not None:
        print(f"Anomaly Score:    {result['normalized_score']:.3f}")
        print(f"Confidence:       {result['confidence']:.3f}")
        print(f"Alert:            {'⚠️  YES' if result['alert'] else '✓ NO'}")
    if 'error' in result:
        print(f"Error:            {result['error']}")
    print("="*80)


def main():
    """Main deployment pipeline."""
    parser = argparse.ArgumentParser(
        description="Production deployment for anomaly detection"
    )
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to trained model file'
    )
    parser.add_argument(
        '--preprocessor',
        type=str,
        default=None,
        help='Path to preprocessor file (auto-detected if not provided)'
    )
    parser.add_argument(
        '--audio',
        type=str,
        default=None,
        help='Single audio file to process'
    )
    parser.add_argument(
        '--audio-dir',
        type=str,
        default=None,
        help='Directory containing audio files to process'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='predictions.csv',
        help='Output file for results (default: predictions.csv)'
    )
    parser.add_argument(
        '--format',
        type=str,
        default='csv',
        choices=['csv', 'json'],
        help='Output format (default: csv)'
    )
    parser.add_argument(
        '--alert-threshold',
        type=float,
        default=0.5,
        help='Threshold for anomaly score to trigger alert (default: 0.5)'
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if args.audio is None and args.audio_dir is None:
        parser.error("Must provide either --audio or --audio-dir")
    
    # Auto-detect preprocessor path if not provided
    if args.preprocessor is None:
        model_path = Path(args.model)
        preprocessor_path = model_path.parent / model_path.name.replace('_model.pkl', '_preprocessor.pkl')
        if not preprocessor_path.exists():
            # Try alternative naming
            preprocessor_path = model_path.parent / 'preprocessor.pkl'
        if not preprocessor_path.exists():
            parser.error("Could not auto-detect preprocessor path. Please provide --preprocessor")
        args.preprocessor = str(preprocessor_path)
    
    logger.info(f"\n{'='*80}")
    logger.info("PRODUCTION ANOMALY DETECTION")
    logger.info(f"{'='*80}")
    logger.info(f"Model:            {args.model}")
    logger.info(f"Preprocessor:     {args.preprocessor}")
    logger.info(f"Alert Threshold:  {args.alert_threshold}")
    logger.info(f"{'='*80}\n")
    
    # Initialize pipeline
    pipeline = AnomalyDetectionPipeline(
        args.model,
        args.preprocessor,
        args.alert_threshold,
    )
    
    # Process audio
    if args.audio:
        # Single file
        logger.info(f"Processing single file: {args.audio}")
        result = pipeline.process_audio_file(args.audio)
        print_result(result)
        
        # Save result
        save_results([result], args.output, args.format)
        
    elif args.audio_dir:
        # Batch processing
        audio_dir = Path(args.audio_dir)
        audio_files = list(audio_dir.glob('*.wav')) + list(audio_dir.glob('*.mp3'))
        
        if not audio_files:
            logger.error(f"No audio files found in {audio_dir}")
            return
        
        logger.info(f"Found {len(audio_files)} audio files")
        results = pipeline.process_batch([str(f) for f in audio_files])
        
        # Save results
        save_results(results, args.output, args.format)
        
        # Print summary
        n_anomalies = sum(1 for r in results if r.get('prediction') == 'ANOMALY')
        n_alerts = sum(1 for r in results if r.get('alert', False))
        
        print("\n" + "="*80)
        print("BATCH PROCESSING SUMMARY")
        print("="*80)
        print(f"Total Files:      {len(results)}")
        print(f"Anomalies:        {n_anomalies}")
        print(f"Alerts:           {n_alerts}")
        print(f"Results saved to: {args.output}")
        print("="*80)
    
    logger.info("\nDeployment complete!")


if __name__ == "__main__":
    main()
