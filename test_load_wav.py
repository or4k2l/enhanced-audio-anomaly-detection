import librosa
import sys

# Pfad zur ersten Audiodatei aus dem Datensatz
wav_path = "/home/codespace/.cache/kagglehub/datasets/vuppalaadithyasairam/anomaly-detection-from-sound-data-fan/versions/1/dev_data_fan/train/normal_id_02_00000410.wav"

print(f"Lade Datei: {wav_path}")
try:
    audio, sr = librosa.load(wav_path, sr=None)
    print(f"Audio geladen: Länge={len(audio)}, Sample Rate={sr}")
except Exception as e:
    print(f"Fehler beim Laden: {e}")
    sys.exit(1)

print("Datei erfolgreich geladen und analysiert.")
