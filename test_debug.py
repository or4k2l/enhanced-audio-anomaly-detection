import sys
sys.path.insert(0, 'src')
from audio_anom.features import AudioFeatureExtractor
import numpy as np
extractor = AudioFeatureExtractor()
audio = np.random.randn(16000)
stats = extractor.extract_statistical_features(audio)
print('Keys in stats:', list(stats.keys()))
print('zcr in stats:', 'zcr' in stats)
features = extractor.extract_features(audio)
print('Features keys:', list(features.keys()) if features else 'None')
print('mel_spec_mean in features:', 'mel_spec_mean' in features if features else 'N/A')