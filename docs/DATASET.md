# Dataset Preparation Guide

## MIMII Pump Sound Dataset

This project uses the MIMII (Malfunctioning Industrial Machine Investigation and Inspection) dataset for pump sounds.

### Dataset Overview

The MIMII dataset contains sound recordings of industrial machines including pumps, valves, fans, and sliders. For this project, we focus on the **pump** category.

- **Normal sounds**: Recordings of properly functioning pumps
- **Anomaly sounds**: Recordings of pumps with various malfunctions

### Download Instructions

#### Option 1: Using KaggleHub (Recommended)

The dataset can be downloaded using the `kagglehub` package (already in requirements.txt):

```python
import kagglehub

# Download MIMII dataset
path = kagglehub.dataset_download("shrutikamble/mimii-dataset")
print(f"Dataset downloaded to: {path}")
```

#### Option 2: Manual Download

1. Visit the official MIMII dataset website or Kaggle
2. Download the pump sound dataset
3. Extract to `data/pump/` directory

### Directory Structure

After downloading and extracting, organize your data as follows:

```
data/
└── pump/
    ├── normal_id_00_00000000.wav
    ├── normal_id_00_00000001.wav
    ├── ...
    ├── anomaly_id_00_00000000.wav
    ├── anomaly_id_00_00000001.wav
    └── ...
```

### Audio File Naming Convention

The dataset follows a naming convention:
- Files containing "normal" in the filename are normal samples
- Files containing "anomaly" in the filename are anomalous samples

### Dataset Statistics

- **Sample Rate**: 16 kHz
- **Duration**: Typically 10 seconds per file
- **Format**: WAV (uncompressed)
- **Channels**: Mono

### Quick Setup Script

Create a script to download and prepare the dataset:

```python
import kagglehub
import os
import shutil
from pathlib import Path

# Download dataset
print("Downloading MIMII dataset...")
dataset_path = kagglehub.dataset_download("shrutikamble/mimii-dataset")

# Create target directory
target_dir = Path("data/pump")
target_dir.mkdir(parents=True, exist_ok=True)

# Copy pump files
pump_source = Path(dataset_path) / "pump"
if pump_source.exists():
    for wav_file in pump_source.glob("*.wav"):
        shutil.copy(wav_file, target_dir)
    print(f"Copied pump files to {target_dir}")
else:
    print("Please manually organize the dataset")

print("Dataset preparation complete!")
```

### Validation

After setting up, verify your dataset:

```bash
# Count files
ls data/pump/*.wav | wc -l

# Check for normal and anomaly samples
ls data/pump/*normal*.wav | wc -l
ls data/pump/*anomaly*.wav | wc -l
```

### Training with Custom Data

If you have your own pump sound recordings:

1. Organize WAV files in `data/pump/` directory
2. Name files with "normal" or "anomaly" in the filename
3. Ensure consistent sample rate (16 kHz recommended)
4. Run the training script as usual

### Troubleshooting

**Issue**: No files found during training
- Check that files are in the correct directory
- Verify file extensions match the pattern (default: `*.wav`)
- Ensure files have read permissions

**Issue**: Memory errors during training
- Reduce the dataset size
- Adjust the `--duration` parameter to trim audio length
- Process data in batches

**Issue**: Poor model performance
- Ensure balanced dataset (similar number of normal and anomaly samples)
- Adjust `--contamination` parameter to match actual anomaly ratio
- Try different feature extraction parameters

## References

- MIMII Dataset: https://zenodo.org/record/3384388
- Paper: "MIMII Dataset: Sound Dataset for Malfunctioning Industrial Machine Investigation and Inspection"
