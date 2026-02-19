"""Setup configuration for enhanced-audio-anomaly-detection package."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="audio-anom",
    version="3.0.0",
    author="Enhanced Audio Anomaly Detection Team",
    description="Enhanced pipeline for audio anomaly detection with embedding-based methods",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/or4k2l/enhanced-audio-anomaly-detection",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "black",
            "flake8",
            "pytest",
        ],
        "tensorflow": [
            "tensorflow",
        ],
    },
    entry_points={
        "console_scripts": [
            "audio-anom-train=audio_anom.train:main",
            "train-unsupervised=scripts.train_unsupervised:main",
            "evaluate-dc2020=scripts.evaluate_dc2020:main",
            "deploy-anomaly=scripts.deploy_production:main",
            "embedding-anomaly-example=examples.embedding_anomaly_example:main",
            "augmentation-demo=examples.augmentation_demo:visualize_augmentations",
        ],
    },
)
