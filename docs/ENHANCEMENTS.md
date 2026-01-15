# Erweiterte Funktionen - Übersicht

## Was wurde hinzugefügt?

### 1. Erweiterte Feature-Extraktion (`features.py`)
- **Zeitdomäne**: RMS, Energie, Zero-Crossing Rate, statistische Momente (Mean, Std, Skewness, Kurtosis, Peak)
- **Frequenzdomäne (FFT)**: Dominante Frequenz, Spektraler Zentroid/Spread, Multi-Band-Energie (5 Bänder)
- **Spektrale Features (Librosa)**: Spectral Centroid, Rolloff, Flatness, Bandwidth, Contrast, Chroma
- **MFCCs**: Bis zu 20 Koeffizienten mit Mittelwert und Standardabweichung
- Insgesamt **77+ Features** pro Audio-Segment

### 2. Neue ML-Modelle (`models_advanced.py`)

#### RandomForestAnomalyDetector
- GridSearchCV mit automatischer Hyperparameter-Optimierung
- Parameter-Grid für n_estimators, max_depth, min_samples_split, etc.
- Optionale PCA-Dimensionsreduktion (95% Varianz)
- SMOTE für Klassenbalancierung
- 5-Fold Stratified Cross-Validation

#### XGBoostAnomalyDetector
- Gradient Boosting mit optimierten Parametern
- GridSearchCV für learning_rate, max_depth, n_estimators, etc.
- Automatische scale_pos_weight Berechnung
- PCA und SMOTE Integration
- Cross-Validation Support

#### AutoencoderAnomalyDetector
- Deep Learning Modell mit TensorFlow/Keras
- Encoder-Decoder Architektur (64-32-10-32-64)
- Training nur auf normalen Daten (unsupervised)
- Anomaliedetektion via Rekonstruktionsfehler
- Automatische Threshold-Berechnung (95. Perzentil)

### 3. Evaluierungs- und Visualisierungsmodul (`evaluation.py`)

#### ModelEvaluator-Klasse
- **Metriken**: Accuracy, Precision, Recall, F1-Score, AUC-ROC
- **Visualisierungen**:
  - Modellvergleich (Bar-Charts)
  - Confusion Matrices
  - ROC Curves für alle Modelle
  - Feature Importance (für Tree-basierte Modelle)
  - PCA Explained Variance
  - Comprehensive Reports (6 Subplots)

### 4. Erweitertes Trainingsskript (`train_enhanced.py`)
- Training mehrerer Modelle parallel (RF, XGBoost, Autoencoder)
- Automatische Modellauswahl per Command-Line
- PCA und SMOTE optional konfigurierbar
- Umfassende Evaluierung auf Validierungs- und Testset
- Automatisches Speichern des besten Modells
- Visualisierungsgenerierung

### 5. Demo-Skript (`demo_enhanced.py`)
- Funktioniert mit und ohne echte Daten
- Generiert synthetische Daten für Demonstration
- Zeigt komplette Pipeline von Feature-Extraktion bis Evaluation
- Erstellt Visualisierungen

## Verwendung

### Basis-Training (Legacy)
```bash
python src/audio_anom/train.py --data-dir data/pump
```

### Erweitertes Training
```bash
# Alle Modelle trainieren
python src/audio_anom/train_enhanced.py \
    --data-dir data/pump \
    --models rf xgb ae \
    --enhanced-features \
    --use-pca \
    --use-smote \
    --visualize

# Nur Random Forest
python src/audio_anom/train_enhanced.py \
    --data-dir data/pump \
    --models rf \
    --n-mfcc 20 \
    --pca-variance 0.95
```

### Demo ausführen
```bash
python examples/demo_enhanced.py  # Verwendet synthetische Daten wenn kein Dataset vorhanden
```

### Python API
```python
from audio_anom import (
    AudioFeatureExtractor,
    RandomForestAnomalyDetector,
    XGBoostAnomalyDetector,
    AutoencoderAnomalyDetector,
    ModelEvaluator
)

# Feature-Extraktion
extractor = AudioFeatureExtractor(sr=16000, n_mfcc=20)
features = extractor.extract_features(audio, enhanced=True)
# -> 77+ Features

# Training
rf_detector = RandomForestAnomalyDetector(random_state=42)
rf_detector.fit(X_train, y_train, use_pca=True, use_smote=True)

# Evaluierung
evaluator = ModelEvaluator()
metrics = evaluator.evaluate_model(y_test, y_pred, y_prob)
evaluator.create_comprehensive_report(models_results, y_test)
```

## Vorteile gegenüber dem ursprünglichen Code

### Modulare Struktur
- ✅ Klare Trennung in Module (features, models, evaluation)
- ✅ Wiederverwendbare Komponenten
- ✅ Einfache Erweiterbarkeit
- ❌ Original: Alles in einer 900+ Zeilen Datei

### Best Practices
- ✅ Objektorientiertes Design
- ✅ Type Hints und Docstrings
- ✅ Error Handling
- ✅ Konfigurierbare Parameter
- ✅ Unit Tests vorhanden

### Funktionalität
- ✅ Gleiche erweiterte Features wie Original
- ✅ Gleiche ML-Modelle (RF, XGBoost, Autoencoder)
- ✅ Gleiche Preprocessing-Techniken (PCA, SMOTE)
- ✅ Gleiche Evaluierungsmethoden
- ✅ **Plus**: Bessere Code-Organisation

## Vergleich: Original vs. Erweitert

| Feature | Original-Code | Ihr Projekt (neu) |
|---------|---------------|-------------------|
| **Struktur** | 1 große Datei (~900 Zeilen) | Modulare Dateien (5+ Module) |
| **Features** | 77+ Features | ✅ Gleich (77+ Features) |
| **Modelle** | RF, XGBoost, Autoencoder | ✅ Gleich |
| **GridSearchCV** | ✅ | ✅ |
| **PCA** | ✅ | ✅ |
| **SMOTE** | ✅ | ✅ |
| **Visualisierungen** | ✅ | ✅ Verbessert |
| **Code-Qualität** | Skript-Style | ✅ OOP, Tests, Docs |
| **Wartbarkeit** | Schwierig | ✅ Einfach |
| **Erweiterbarkeit** | Schwierig | ✅ Einfach |

## Zusammenfassung

Das Projekt wurde erfolgreich erweitert und enthält jetzt **alle Funktionen** des gezeigten Original-Codes, aber in einer **professionellen, modularen Struktur**:

1. ✅ Enhanced Feature Extraction (77+ Features)
2. ✅ Random Forest mit GridSearchCV
3. ✅ XGBoost mit GridSearchCV
4. ✅ Autoencoder (Deep Learning)
5. ✅ PCA Dimensionsreduktion
6. ✅ SMOTE Klassenbalancierung
7. ✅ Comprehensive Evaluation
8. ✅ Visualisierungen
9. ✅ Model Comparison
10. ✅ Feature Importance Analysis

**Bonus**: Bessere Code-Organisation, Tests, Dokumentation, und einfachere Verwendung!
