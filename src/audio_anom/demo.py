#!/usr/bin/env python3
"""
Enhanced demo script showcasing all features of the audio anomaly detection system.

This script demonstrates:
- Enhanced feature extraction
- Multiple model training (Random Forest, XGBoost, Autoencoder)
- Comprehensive evaluation and visualization
- Model comparison
"""

import sys
import os
import numpy as np

from .features import AudioFeatureExtractor
from .data import AudioDataProcessor
from .models import RandomForestAnomalyDetector, XGBoostAnomalyDetector
from .evaluation import ModelEvaluator


def main():
    print("=" * 80)
    print("ENHANCED AUDIO ANOMALY DETECTION - DEMO")
    print("=" * 80)
    # ...existing code...

if __name__ == "__main__":
    main()
