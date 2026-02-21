# 🚀 Production Deployment Guide

## Prerequisites

- Python 3.8+
- System: `libsndfile1`, `ffmpeg` (for audio loading)
- GPU: Optional (AST runs fine on CPU for batch < 64)

---

## 1. Installation

```bash
# Clone the repository
git clone https://github.com/or4k2l/enhanced-audio-anomaly-detection.git
cd enhanced-audio-anomaly-detection

# Install (production)
pip install -e .

# Install (development)
pip install -e ".[dev]"
```

---

## 2. Training a Model

```bash
# Train GMM-16 on Pump
python scripts/train_hybrid.py \
    --train_dir data/pump/train \
    --test_dir data/pump/test \
    --machine pump \
    --method gmm \
    --n_components 16 \
    --output models/pump_hybrid_gmm16.pkl

# Train baseline
python scripts/train_baseline.py \
    --train_dir data/pump/train \
    --test_dir data/pump/test \
    --machine pump \
    --output models/pump_baseline.pkl
```

---

## 3. Inference

```bash
python scripts/inference.py \
    --model models/pump_hybrid_gmm16.pkl \
    --audio test_sample.wav
# → Anomaly score: 0.823 (likely anomaly)
```

---

## 4. Docker Containerization

```dockerfile
FROM python:3.10-slim

RUN apt-get update && apt-get install -y libsndfile1 ffmpeg && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN pip install -e .

CMD ["python", "scripts/inference.py", "--help"]
```

```bash
docker build -t audio-anomaly:latest .
docker run -v $(pwd)/models:/app/models \
           -v $(pwd)/data:/app/data \
           audio-anomaly:latest \
           python scripts/inference.py \
               --model /app/models/pump_hybrid_gmm16.pkl \
               --audio /app/data/test_sample.wav
```

---

## 5. FastAPI REST Service

```python
# app.py
from fastapi import FastAPI, UploadFile
import librosa
import numpy as np
import tempfile
import os

from models.ast_extractor import ASTEmbeddingExtractor
from models.classical_features import extract_classical_features
from models.ensemble import HybridAnomalyDetector

app = FastAPI(title="Audio Anomaly Detection API")

# Load model at startup
detector = HybridAnomalyDetector.load("models/pump_hybrid_gmm16.pkl")
ast = ASTEmbeddingExtractor()


@app.post("/score")
async def score_audio(file: UploadFile):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        wave, _ = librosa.load(tmp_path, sr=16000, duration=10.0)
        ast_feat = ast.extract_embedding(wave)
        cls_feat = extract_classical_features(wave)
        features = np.concatenate([ast_feat, cls_feat]).reshape(1, -1)
        score = float(detector.score_samples(features)[0])
        return {"score": score, "label": "anomaly" if score > 0.5 else "normal"}
    finally:
        os.unlink(tmp_path)
```

```bash
pip install fastapi uvicorn python-multipart
uvicorn app:app --host 0.0.0.0 --port 8000
```

---

## 6. Model Selection Guide

| Machine | Best Method | n_components | Expected AUC |
|---------|-------------|--------------|-------------|
| pump | GMM | 16 | 0.874 |
| slider | GMM | 8 | 0.870 |
| valve | GMM | 8 | 0.779 |
| ToyCar | GMM | 8 | 0.751 |
| ToyConveyor | GMM | 8 | 0.594 |
| fan | OCSVM | N/A | 0.651 |

---

## 7. Monitoring & Alerting

Track these metrics in production:
- Score distribution drift (KL divergence over sliding window)
- Anomaly rate (% of files scored above threshold)
- Feature extraction latency (p95 < 500ms)
- Model file checksum (integrity monitoring)

---

## 8. CI/CD Pipeline

```yaml
# .github/workflows/ci.yml (already configured)
# Runs on every PR:
# - flake8 linting
# - black formatting check
# - pytest (all 166+ tests)
```
