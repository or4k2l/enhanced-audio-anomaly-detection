# Unsupervised Anomaly Detection - Technical Guide

## Overview

This module implements production-ready unsupervised anomaly detection for audio anomaly detection. Unlike supervised methods that require labeled anomaly data, unsupervised methods learn from **normal data only** and detect deviations as anomalies.

## When to Use Unsupervised Methods

### ✅ Use When:
- **Anomalies are rare**: Not enough anomaly samples for supervised training
- **Unlabeled data**: No labels available or labeling is expensive
- **Novel anomalies**: Need to detect unknown/unseen anomaly types
- **Production scenarios**: Training data contains only normal operations

### ❌ Don't Use When:
- You have abundant labeled anomaly data → Use supervised methods
- Anomalies are very similar to normal samples → Harder to detect unsupervised
- Need to classify anomaly types → Use multi-class supervised methods

## Available Methods

### 1. Local Outlier Factor (LOF) ⭐ BEST

**Performance**: AUC 0.755+, F1 0.704

**How it works**: Measures local density deviation. Points with substantially lower density than neighbors are outliers.

**Best for**: General-purpose anomaly detection, works well on most datasets

**Parameters**:
- `n_neighbors` (default: 20): Number of neighbors to consider
  - Lower = more sensitive to local anomalies
  - Higher = more robust but may miss small clusters
- `contamination` (default: 0.1): Expected proportion of outliers
  - Range: 0.0 to 0.5
  - Set based on domain knowledge

**Example**:
```python
from audio_anom.unsupervised_anomaly import LocalOutlierFactorAnomalyDetector

model = LocalOutlierFactorAnomalyDetector(
    n_neighbors=20,
    contamination=0.1,
)
model.fit(X_train_normal)  # Only normal data!
predictions = model.predict(X_test)
anomaly_scores = model.anomaly_score(X_test)
```

### 2. Isolation Forest

**Performance**: AUC 0.687+, F1 0.637

**How it works**: Isolates anomalies using random trees. Anomalies are easier to isolate (require fewer splits).

**Best for**: High-dimensional data, fast training, scalable to large datasets

**Parameters**:
- `n_estimators` (default: 100): Number of trees
  - More trees = more stable predictions
  - Typical range: 50-200
- `contamination` (default: 0.1): Expected proportion of outliers
- `max_samples` (default: 'auto'): Samples per tree

**Example**:
```python
from audio_anom.unsupervised_anomaly import IsolationForestAnomalyDetector

model = IsolationForestAnomalyDetector(
    n_estimators=100,
    contamination=0.1,
)
model.fit(X_train_normal)
predictions = model.predict(X_test)
```

### 3. Elliptic Envelope

**Performance**: AUC 0.643+, F1 0.528

**How it works**: Fits an ellipse to central data points using robust covariance estimation.

**Best for**: Data following Gaussian distributions, interpretable results

**Parameters**:
- `contamination` (default: 0.1): Expected proportion of outliers
- `support_fraction` (default: None): Proportion of points in support

**Example**:
```python
from audio_anom.unsupervised_anomaly import EllipticEnvelopeAnomalyDetector

model = EllipticEnvelopeAnomalyDetector(contamination=0.1)
model.fit(X_train_normal)
predictions = model.predict(X_test)
```

## Preprocessing Pipeline

### UnsupervisedPreprocessor

Combines StandardScaler + PCA for optimal performance:

```python
from audio_anom.preprocessing_unsupervised import UnsupervisedPreprocessor

preprocessor = UnsupervisedPreprocessor(
    n_components=10,  # PCA components (88-91% variance)
    apply_pca=True,   # Enable PCA
)

# Fit on normal data only
X_train_proc = preprocessor.fit_transform(X_train_normal)

# Transform test data
X_test_proc = preprocessor.transform(X_test)
```

**Why PCA?**
- Reduces 284 features → 10 components
- Retains 88-91% of variance
- Removes noise and redundant features
- Faster training and inference

## Hyperparameter Tuning

### Contamination Parameter

Most important hyperparameter for all methods:

- **0.01-0.05**: Very clean training data, rare anomalies
- **0.10**: Default, balanced (recommended starting point)
- **0.15-0.30**: Noisy training data, more tolerance

**How to choose**:
1. Start with 0.1 (10% contamination)
2. If too many false positives → decrease contamination
3. If missing anomalies → increase contamination
4. Cross-validate on validation set if available

### LOF-Specific Parameters

**n_neighbors**: Controls sensitivity
- **Small (5-10)**: Detects local anomalies, more sensitive
- **Medium (20-30)**: Balanced (recommended)
- **Large (50-100)**: Detects global anomalies, more robust

### Isolation Forest-Specific Parameters

**n_estimators**: Number of trees
- **50-100**: Fast training, good for prototyping
- **100-200**: Better stability (recommended)
- **200+**: Diminishing returns, slower training

## Best Practices

### 1. Data Preparation
```python
# ✓ GOOD: Train on normal data only
X_train_normal = X[y == 0]  # Only normal samples
model.fit(X_train_normal)

# ✗ BAD: Don't include anomalies in training
X_train_mixed = X  # Contains both normal and anomaly
model.fit(X_train_mixed)  # Will learn wrong patterns!
```

### 2. Preprocessing
```python
# Always preprocess before training
preprocessor = UnsupervisedPreprocessor()
X_train_proc = preprocessor.fit_transform(X_train_normal)
X_test_proc = preprocessor.transform(X_test)

# Don't forget to save preprocessor!
preprocessor.save('preprocessor.pkl')
```

### 3. Evaluation
```python
from audio_anom.evaluation_unsupervised import evaluate_anomaly_detector

metrics = evaluate_anomaly_detector(
    y_true=y_test,
    y_pred=predictions,
    y_score=anomaly_scores,
    model_name="LOF"
)
```

### 4. Model Selection
```python
from audio_anom.evaluation_unsupervised import ModelComparator

comparator = ModelComparator()
comparator.add_model('LOF', lof_model, X_test, y_test)
comparator.add_model('IForest', iforest_model, X_test, y_test)
comparator.add_model('Envelope', envelope_model, X_test, y_test)

comparator.print_summary()
best_model = comparator.get_best_model('roc_auc')
```

## Common Issues & Solutions

### Issue: Low AUC Score (<0.6)

**Possible causes**:
1. Training data contains anomalies → Re-check data filtering
2. Contamination too high → Reduce contamination parameter
3. Features not discriminative → Try different feature extraction
4. Anomalies too similar to normal → Consider supervised methods

### Issue: Too Many False Positives

**Solutions**:
1. Decrease contamination parameter
2. Increase n_neighbors (LOF) or n_estimators (IForest)
3. Apply stricter preprocessing (e.g., more PCA components)
4. Use threshold tuning on validation set

### Issue: Missing Anomalies (Low Recall)

**Solutions**:
1. Increase contamination parameter
2. Decrease n_neighbors (LOF)
3. Try different method (LOF usually best)
4. Check if anomalies are truly anomalous in feature space

## Performance Benchmarks

Results on DCASE 2020 Task 2 (6 machine types, 1000+ test samples each):

| Method | Avg AUC | Avg F1 | Best Machine | Training Time |
|--------|---------|--------|--------------|---------------|
| LOF | 0.7554 | 0.7040 | fan (0.832) | Fast (~1s) |
| Isolation Forest | 0.6873 | 0.6374 | fan (0.758) | Very Fast (<1s) |
| Elliptic Envelope | 0.6426 | 0.5276 | pump (0.702) | Fast (~1s) |

**vs Baselines**:
- Random guessing: AUC = 0.500
- Your system: AUC = 0.755 (**+51% improvement**)
- DCASE 2020 baseline: AUC ≈ 0.70 (**your system beats it!**)

## References

1. Breunig et al. (2000). "LOF: Identifying Density-Based Local Outliers"
2. Liu et al. (2008). "Isolation Forest"
3. Rousseeuw & Driessen (1999). "A Fast Algorithm for the Minimum Covariance Determinant Estimator"
4. DCASE 2020 Challenge: https://dcase.community/challenge2020/task-unsupervised-detection-of-anomalous-sounds

## See Also

- [DC2020 Results Report](DC2020_RESULTS.md) - Detailed evaluation results
- [Production Guide](PRODUCTION_GUIDE.md) - Deployment instructions
- [API Documentation](../src/audio_anom/) - Module reference
