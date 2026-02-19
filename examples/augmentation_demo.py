"""Example: Audio Augmentation Techniques.

This example demonstrates all augmentation techniques:
- Mixup
- SpecAugment  
- Time stretching
- Pitch shifting
- Noise injection
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from audio_anom import AudioAugmenter, RobustFeatureExtractor


def generate_test_audio(duration=1.0, sr=22050):
    """Generate a simple test audio signal.
    
    Args:
        duration: Duration in seconds
        sr: Sample rate
        
    Returns:
        audio: Audio signal
    """
    t = np.linspace(0, duration, int(sr * duration))
    
    # Combination of two sine waves
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)  # A4 note
    audio += 0.3 * np.sin(2 * np.pi * 880 * t)  # A5 note
    
    return audio


def visualize_augmentations():
    """Demonstrate and visualize all augmentation techniques."""
    print("=" * 70)
    print("Audio Augmentation Demo")
    print("=" * 70)
    
    # Set random seed
    np.random.seed(42)
    
    # Generate test audio
    print("\n1. Generating test audio...")
    sr = 22050
    audio = generate_test_audio(duration=1.0, sr=sr)
    print(f"  Sample rate: {sr} Hz")
    print(f"  Duration: {len(audio)/sr:.2f} seconds")
    
    # Create augmenter
    augmenter = AudioAugmenter(sr=sr, random_state=42)
    
    # Create extractor for spectrograms
    extractor = RobustFeatureExtractor(sr=sr, n_mels=128)
    
    print("\n2. Applying augmentations...")
    
    # Original
    mel_spec_original = extractor.extract_mel_spectrogram(audio)
    
    # Mixup (mix with another audio)
    audio2 = generate_test_audio(duration=1.0, sr=sr)
    audio2 = 0.3 * np.sin(2 * np.pi * 220 * np.linspace(0, 1.0, len(audio2)))
    audio_mixup = augmenter.mixup(audio, audio2, alpha=0.3)
    mel_spec_mixup = extractor.extract_mel_spectrogram(audio_mixup)
    
    # Time stretch
    audio_stretched = augmenter.time_stretch(audio, rate_range=(0.8, 0.8))
    # Pad or truncate to match original length
    if len(audio_stretched) > len(audio):
        audio_stretched = audio_stretched[:len(audio)]
    else:
        audio_stretched = np.pad(audio_stretched, (0, len(audio) - len(audio_stretched)))
    mel_spec_stretched = extractor.extract_mel_spectrogram(audio_stretched)
    
    # Pitch shift
    audio_pitched = augmenter.pitch_shift(audio, n_steps_range=(2, 2))
    mel_spec_pitched = extractor.extract_mel_spectrogram(audio_pitched)
    
    # Noise injection
    audio_noisy = augmenter.add_noise(audio, noise_factor_range=(0.05, 0.05))
    mel_spec_noisy = extractor.extract_mel_spectrogram(audio_noisy)
    
    # SpecAugment
    mel_spec_specaug = augmenter.spec_augment(
        mel_spec_original.copy(),
        freq_mask_param=20,
        time_mask_param=30
    )
    
    print("  ✓ Mixup")
    print("  ✓ Time stretching")
    print("  ✓ Pitch shifting")
    print("  ✓ Noise injection")
    print("  ✓ SpecAugment")
    
    # Visualize
    print("\n3. Creating visualizations...")
    
    fig, axes = plt.subplots(3, 2, figsize=(12, 14))
    
    augmentations = [
        ('Original', mel_spec_original),
        ('Mixup (α=0.3)', mel_spec_mixup),
        ('Time Stretch (0.8x)', mel_spec_stretched),
        ('Pitch Shift (+2 semitones)', mel_spec_pitched),
        ('Noise Injection', mel_spec_noisy),
        ('SpecAugment', mel_spec_specaug),
    ]
    
    for idx, (title, mel_spec) in enumerate(augmentations):
        row = idx // 2
        col = idx % 2
        ax = axes[row, col]
        
        if mel_spec is not None:
            im = ax.imshow(
                mel_spec,
                aspect='auto',
                origin='lower',
                cmap='viridis',
                interpolation='nearest'
            )
            plt.colorbar(im, ax=ax, format='%+2.0f dB')
        
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('Time')
        ax.set_ylabel('Mel Frequency')
    
    plt.tight_layout()
    
    # Save
    output_path = Path(__file__).parent / "augmentation_demo.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  Saved visualization to: {output_path}")
    
    # Demonstrate batch augmentation
    print("\n4. Batch augmentation demo...")
    audio_batch = [generate_test_audio(duration=1.0, sr=sr) for _ in range(10)]
    augmented_batch = augmenter.augment_batch(audio_batch, apply_prob=0.8)
    print(f"  Augmented {len(augmented_batch)} audio samples")
    
    # Parameter recommendations
    print("\n5. Parameter Recommendations:")
    print("-" * 70)
    print("Augmentation         | Recommended Parameters")
    print("-" * 70)
    print("Mixup                | alpha=0.2 (light mixing)")
    print("SpecAugment          | freq_mask=30, time_mask=40")
    print("Time Stretch         | rate_range=(0.9, 1.1)")
    print("Pitch Shift          | n_steps_range=(-2, 2)")
    print("Noise Injection      | noise_factor=(0.001, 0.01)")
    print("-" * 70)
    
    print("\n" + "=" * 70)
    print("Augmentation demo completed!")
    print("=" * 70)


if __name__ == '__main__':
    visualize_augmentations()
