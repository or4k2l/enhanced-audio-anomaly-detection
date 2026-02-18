# Production Deployment Guide

## Overview

This guide covers deploying the unsupervised anomaly detection system to production environments.

## Deployment Options

### 1. Standalone Script (Simplest)

Use the provided `deploy_production.py` script for simple deployments.

**Single File Inference**:
```bash
python scripts/deploy_production.py \
    --model models/fan_lof_model.pkl \
    --preprocessor models/fan_preprocessor.pkl \
    --audio test_audio.wav
```

**Batch Processing**:
```bash
python scripts/deploy_production.py \
    --model models/fan_lof_model.pkl \
    --audio-dir /path/to/audio/files/ \
    --output results.csv \
    --alert-threshold 0.7
```

### 2. Python API Integration

Integrate into existing Python applications:

```python
from audio_anom import AudioFeatureExtractor, build_feature_vector
from audio_anom.unsupervised_anomaly import LocalOutlierFactorAnomalyDetector
from audio_anom.preprocessing_unsupervised import UnsupervisedPreprocessor

# Load models
model = LocalOutlierFactorAnomalyDetector()
model.load('models/fan_lof_model.pkl')
preprocessor = UnsupervisedPreprocessor.load('models/fan_preprocessor.pkl')

# Initialize feature extractor
feature_extractor = AudioFeatureExtractor()

# Process audio
def detect_anomaly(audio_path):
    # Extract features
    audio, sr = librosa.load(audio_path, sr=16000)
    features = feature_extractor.extract_features(audio)
    feature_vector = build_feature_vector(features).reshape(1, -1)
    
    # Preprocess
    feature_vector_proc = preprocessor.transform(feature_vector)
    
    # Predict
    prediction = model.predict(feature_vector_proc)[0]
    anomaly_score = model.anomaly_score(feature_vector_proc)[0]
    
    return {
        'is_anomaly': bool(prediction),
        'anomaly_score': float(anomaly_score),
    }
```

### 3. REST API Service

Deploy as a web service using Flask/FastAPI:

**app.py**:
```python
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)

# Load models at startup
model = LocalOutlierFactorAnomalyDetector()
model.load('models/fan_lof_model.pkl')
preprocessor = UnsupervisedPreprocessor.load('models/fan_preprocessor.pkl')
feature_extractor = AudioFeatureExtractor()

@app.route('/predict', methods=['POST'])
def predict():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file'}), 400
    
    file = request.files['audio']
    filename = secure_filename(file.filename)
    filepath = os.path.join('/tmp', filename)
    file.save(filepath)
    
    try:
        # Process audio
        audio, sr = librosa.load(filepath, sr=16000)
        features = feature_extractor.extract_features(audio)
        feature_vector = build_feature_vector(features).reshape(1, -1)
        feature_vector_proc = preprocessor.transform(feature_vector)
        
        # Predict
        prediction = model.predict(feature_vector_proc)[0]
        score = model.anomaly_score(feature_vector_proc)[0]
        
        return jsonify({
            'prediction': 'anomaly' if prediction == 1 else 'normal',
            'anomaly_score': float(score),
            'filename': filename,
        })
    finally:
        os.remove(filepath)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

**Run**:
```bash
python app.py
```

**Test**:
```bash
curl -X POST -F "audio=@test.wav" http://localhost:5000/predict
```

### 4. Docker Container

**Dockerfile**:
```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Expose port
EXPOSE 5000

# Run
CMD ["python", "app.py"]
```

**Build and run**:
```bash
docker build -t anomaly-detector .
docker run -p 5000:5000 -v ./models:/app/models anomaly-detector
```

## Performance Optimization

### 1. Model Loading

**Problem**: Loading models for each request is slow.

**Solution**: Load models once at startup (as shown in REST API example).

### 2. Batch Processing

**Problem**: Processing files one-by-one is inefficient.

**Solution**: Process in batches:

```python
def batch_predict(audio_paths, batch_size=32):
    results = []
    for i in range(0, len(audio_paths), batch_size):
        batch = audio_paths[i:i+batch_size]
        
        # Extract features for batch
        features_batch = []
        for path in batch:
            audio, sr = librosa.load(path, sr=16000)
            features = feature_extractor.extract_features(audio)
            features_batch.append(build_feature_vector(features))
        
        # Process batch
        X_batch = np.array(features_batch)
        X_batch_proc = preprocessor.transform(X_batch)
        predictions = model.predict(X_batch_proc)
        scores = model.anomaly_score(X_batch_proc)
        
        results.extend(zip(batch, predictions, scores))
    
    return results
```

### 3. Feature Caching

Cache extracted features to avoid re-computation:

```python
from functools import lru_cache

@lru_cache(maxsize=1000)
def get_features(audio_path):
    audio, sr = librosa.load(audio_path, sr=16000)
    features = feature_extractor.extract_features(audio)
    return build_feature_vector(features)
```

### 4. Model Quantization

Reduce model size and inference time (advanced):

```python
# For Isolation Forest models
from sklearn.tree import _tree

# Quantize tree thresholds to float32
for tree in model.model.estimators_:
    tree.tree_.threshold = tree.tree_.threshold.astype(np.float32)
```

## Monitoring & Maintenance

### 1. Logging

Log all predictions for monitoring:

```python
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def predict_with_logging(audio_path):
    result = detect_anomaly(audio_path)
    
    logger.info(
        f"Prediction: {result['is_anomaly']}, "
        f"Score: {result['anomaly_score']:.4f}, "
        f"File: {audio_path}"
    )
    
    return result
```

### 2. Performance Metrics

Track key metrics:

```python
from collections import defaultdict
from datetime import datetime

class MetricsTracker:
    def __init__(self):
        self.predictions = []
        self.start_time = datetime.now()
    
    def log_prediction(self, is_anomaly, score, latency_ms):
        self.predictions.append({
            'timestamp': datetime.now(),
            'is_anomaly': is_anomaly,
            'score': score,
            'latency_ms': latency_ms,
        })
    
    def get_stats(self):
        total = len(self.predictions)
        anomalies = sum(1 for p in self.predictions if p['is_anomaly'])
        avg_latency = np.mean([p['latency_ms'] for p in self.predictions])
        
        return {
            'total_predictions': total,
            'anomaly_rate': anomalies / total if total > 0 else 0,
            'avg_latency_ms': avg_latency,
            'uptime_hours': (datetime.now() - self.start_time).total_seconds() / 3600,
        }
```

### 3. Model Retraining

Retrain periodically with new normal data:

```bash
# Collect new normal data
# Retrain model
python scripts/train_unsupervised.py \
    --machine fan \
    --data-dir /path/to/new/data \
    --output models_v2/

# Test new model
python scripts/evaluate_dc2020.py \
    --machines fan \
    --output results_v2.csv

# If better, deploy new model
mv models_v2/fan_lof_model.pkl models/
```

### 4. Alerting

Set up alerts for anomalies:

```python
def send_alert(audio_path, anomaly_score):
    if anomaly_score > 0.8:
        # Send email/SMS/Slack notification
        send_notification(
            title="High Anomaly Detected",
            message=f"File: {audio_path}, Score: {anomaly_score:.3f}",
            severity="HIGH"
        )
```

## Best Practices

### 1. Model Selection

- **Small datasets (<10k)**: LOF (best accuracy)
- **Large datasets (>100k)**: Isolation Forest (faster)
- **Real-time (<100ms)**: Isolation Forest or Elliptic Envelope

### 2. Contamination Tuning

Start conservative (0.05-0.10) and adjust based on production data:

```python
# Monitor false positive rate
fpr = false_positives / total_predictions

if fpr > 0.15:  # Too many false alarms
    contamination = contamination * 0.8  # Decrease
elif fpr < 0.05:  # Might be missing anomalies
    contamination = contamination * 1.2  # Increase
```

### 3. Threshold Calibration

Use validation set to find optimal threshold:

```python
from audio_anom.evaluation_unsupervised import find_optimal_threshold

optimal_threshold, best_f1 = find_optimal_threshold(
    y_val, scores_val, metric='f1'
)
```

### 4. Error Handling

Robust error handling for production:

```python
def robust_predict(audio_path):
    try:
        return detect_anomaly(audio_path)
    except Exception as e:
        logger.error(f"Prediction failed for {audio_path}: {e}")
        return {
            'error': str(e),
            'is_anomaly': None,
            'anomaly_score': None,
        }
```

## Security Considerations

### 1. Input Validation

Validate audio files:

```python
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'flac'}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB

def validate_audio(file):
    # Check extension
    if not file.filename.endswith(tuple(ALLOWED_EXTENSIONS)):
        raise ValueError("Invalid file type")
    
    # Check size
    file.seek(0, os.SEEK_END)
    size = file.tell()
    file.seek(0)
    if size > MAX_FILE_SIZE:
        raise ValueError("File too large")
```

### 2. Rate Limiting

Prevent abuse:

```python
from flask_limiter import Limiter

limiter = Limiter(app, key_func=lambda: request.remote_addr)

@app.route('/predict', methods=['POST'])
@limiter.limit("10 per minute")
def predict():
    # ...
```

## Troubleshooting

### Issue: High Latency

**Check**:
1. Feature extraction time (should be <500ms)
2. Model inference time (should be <100ms)
3. Network latency (if using API)

**Solutions**:
- Use faster model (Isolation Forest)
- Reduce audio length
- Batch processing

### Issue: Many False Positives

**Solutions**:
1. Decrease contamination parameter
2. Retrain with more diverse normal data
3. Tune threshold on validation set
4. Try different method (LOF usually most accurate)

### Issue: Missing Anomalies

**Solutions**:
1. Increase contamination parameter
2. Check if anomalies are truly distinguishable
3. Improve feature extraction
4. Collect more training data

## Resources

- [Technical Guide](UNSUPERVISED.md)
- [Results Report](DC2020_RESULTS.md)
- [GitHub Repository](https://github.com/or4k2l/enhanced-audio-anomaly-detection)

---

**Status**: Production-Ready ✅
**Tested**: DCASE 2020 Task 2 (AUC 0.755)
**Support**: See GitHub Issues
