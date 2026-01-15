#!/bin/bash

# Training script for audio anomaly detection

echo "Starting training for audio anomaly detection..."

# Default parameters
DATA_DIR="${DATA_DIR:-data/pump}"
OUTPUT_DIR="${OUTPUT_DIR:-models}"
SAMPLE_RATE="${SAMPLE_RATE:-16000}"

# Run training
python src/audio_anom/train.py \
    --data-dir "$DATA_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --sample-rate "$SAMPLE_RATE" \
    --train-ratio 0.7 \
    --val-ratio 0.15 \
    --contamination 0.1 \
    --n-mels 128 \
    --n-fft 1024 \
    --hop-length 512 \
    --random-state 42

echo "Training complete!"
