"""Integrationstest: Feature- und Modellverhalten mit echten Audiodaten (z.B. Sprache, Musik, Stille)."""

import numpy as np
import pytest
import sys
from pathlib import Path
import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from audio_anom.features import AudioFeatureExtractor
from audio_anom import RandomForestAnomalyDetector

# Beispielhafte Testdaten (könnten durch echte Dateien ersetzt werden)
TEST_AUDIO_DIR = Path(__file__).parent / "real_audio"


@pytest.mark.parametrize(
    "filename,expected_min,expected_max",
    [
        ("speech_sample.wav", -1.0, 1.0),
        ("music_sample.wav", -1.0, 1.0),
        ("silence.wav", -0.01, 0.01),
    ],
)
def test_feature_extraction_on_real_audio(filename, expected_min, expected_max):
    """
    Testet die Feature-Extraktion auf echten Audiodateien.
    """
    file_path = TEST_AUDIO_DIR / filename
    if not file_path.exists():
        pytest.skip(f"Testdatei {filename} nicht vorhanden.")
    audio, sr = sf.read(file_path)
    extractor = AudioFeatureExtractor(sr=sr)
    features = extractor.extract_features(audio)
    assert features is not None
    # Prüfe Wertebereich
    assert expected_min <= np.min(audio) <= expected_max
    assert expected_min <= np.max(audio) <= expected_max
    # Prüfe, dass Mel-Spektrogramm und MFCCs extrahiert wurden
    assert "mel_spec_mean" in features
    assert "mfcc_mean" in features


def test_model_on_real_audio():
    """
    Trainiert und testet ein Modell auf synthetisch erzeugten, aber realistischen Audiodaten.
    """
    sr = 16000
    extractor = AudioFeatureExtractor(sr=sr)
    # Erzeuge "normale" und "anomalie"-Samples
    normal = np.random.normal(0, 0.1, sr)
    anomaly = np.random.normal(0, 1.0, sr)
    X = []
    y = []
    for _ in range(10):
        X.append(extractor.extract_features(normal))
        y.append(0)
        X.append(extractor.extract_features(anomaly))
        y.append(1)
    X = np.array(
        [
            np.concatenate(
                [x["mel_spec_mean"], x["mel_spec_std"], x["mfcc_mean"], x["mfcc_std"]]
            )
            for x in X
        ]
    )
    y = np.array(y)
    model = RandomForestAnomalyDetector()
    model.fit(X, y)
    y_pred = model.predict(X)
    assert (y_pred == y).mean() > 0.8  # Modell sollte klar trennen können
