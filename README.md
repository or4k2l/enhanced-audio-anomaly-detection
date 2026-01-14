# Enhanced Audio Anomaly Detection

This repository contains an enhanced pipeline for audio anomaly detection (pump sounds) including feature extraction, training, evaluation and model export. It is based on the MIMII Pump Sound dataset (download instructions included).

Quickstart
1. Create and activate a Python 3.10+ virtual environment
2. Install dependencies:
   pip install -r requirements.txt
3. Prepare the dataset following `docs/DATASET.md`.
4. Run training (example):
   bash scripts/train.sh

Structure
- src/audio_anom/: package with feature extraction, data processing, models and training scripts
- scripts/: convenience scripts
- examples/: example inference script
- tests/: unit tests
- docs/: dataset instructions

License: MIT
