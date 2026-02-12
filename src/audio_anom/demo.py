#!/usr/bin/env python3
"""Package-level demo entrypoint.

Run with:
    python -m audio_anom.demo [--data-dir /path/to/wavs]
"""

from .train import main as train_main


def main():
    """Forward to training CLI so `python -m audio_anom.demo` is useful."""
    train_main()


if __name__ == "__main__":
    main()
