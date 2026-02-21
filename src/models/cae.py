"""Convolutional Autoencoder for audio anomaly detection.

Implements a 4-layer encoder/decoder architecture with dropout regularization
for reconstruction-based anomaly detection on mel-spectrogram features.
"""

import logging
import numpy as np

logger = logging.getLogger(__name__)


class ConvolutionalAutoencoder:
    """4-layer Convolutional Autoencoder for anomaly detection.

    Uses reconstruction error as an anomaly score. Higher reconstruction error
    indicates a potential anomaly.

    Note:
        Requires PyTorch. Import is deferred to avoid hard dependency.

    Args:
        input_dim: Dimensionality of input features.
        latent_dim: Dimensionality of the bottleneck layer.
        dropout: Dropout rate for regularization.
        learning_rate: Learning rate for Adam optimizer.

    Example:
        >>> cae = ConvolutionalAutoencoder(input_dim=128)
        >>> cae.fit(X_train, epochs=30)
        >>> scores = cae.score_samples(X_test)
    """

    def __init__(
        self,
        input_dim: int = 128,
        latent_dim: int = 32,
        dropout: float = 0.3,
        learning_rate: float = 1e-3,
    ):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.dropout = dropout
        self.learning_rate = learning_rate
        self._model = None
        self._is_fitted = False

    def _build_model(self):
        """Build the encoder-decoder architecture."""
        try:
            import torch.nn as nn

            class _CAEModel(nn.Module):
                def __init__(self, input_dim, latent_dim, dropout):
                    super().__init__()
                    self.encoder = nn.Sequential(
                        nn.Linear(input_dim, 256),
                        nn.ReLU(),
                        nn.Dropout(dropout),
                        nn.Linear(256, 128),
                        nn.ReLU(),
                        nn.Dropout(dropout),
                        nn.Linear(128, 64),
                        nn.ReLU(),
                        nn.Linear(64, latent_dim),
                    )
                    self.decoder = nn.Sequential(
                        nn.Linear(latent_dim, 64),
                        nn.ReLU(),
                        nn.Linear(64, 128),
                        nn.ReLU(),
                        nn.Dropout(dropout),
                        nn.Linear(128, 256),
                        nn.ReLU(),
                        nn.Dropout(dropout),
                        nn.Linear(256, input_dim),
                    )

                def forward(self, x):
                    return self.decoder(self.encoder(x))

            return _CAEModel(self.input_dim, self.latent_dim, self.dropout)
        except ImportError:
            raise ImportError(
                "PyTorch is required for ConvolutionalAutoencoder. "
                "Install with: pip install torch"
            )

    def fit(
        self,
        X: np.ndarray,
        epochs: int = 30,
        batch_size: int = 32,
    ) -> "ConvolutionalAutoencoder":
        """Train the autoencoder on normal samples.

        Args:
            X: Training features of shape (n_samples, input_dim).
            epochs: Number of training epochs.
            batch_size: Mini-batch size.

        Returns:
            self
        """
        try:
            import torch
            import torch.nn as nn
            import torch.optim as optim
            from torch.utils.data import DataLoader, TensorDataset
        except ImportError:
            raise ImportError("PyTorch is required. Install with: pip install torch")

        self._model = self._build_model()
        optimizer = optim.Adam(self._model.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()

        X_tensor = torch.FloatTensor(X)
        dataset = TensorDataset(X_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self._model.train()
        for epoch in range(epochs):
            total_loss = 0.0
            for (batch,) in loader:
                optimizer.zero_grad()
                recon = self._model(batch)
                loss = criterion(recon, batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            if (epoch + 1) % 10 == 0:
                avg_loss = total_loss / len(loader)
                logger.info(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

        self._is_fitted = True
        return self

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Compute reconstruction error as anomaly score.

        Args:
            X: Features of shape (n_samples, input_dim).

        Returns:
            Anomaly scores of shape (n_samples,). Higher = more anomalous.

        Raises:
            ValueError: If the model has not been fitted.
        """
        if not self._is_fitted:
            raise ValueError("Model must be fitted before scoring. Call fit() first.")

        try:
            import torch
        except ImportError:
            raise ImportError("PyTorch is required. Install with: pip install torch")

        self._model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X)
            recon = self._model(X_tensor).numpy()

        return np.mean((X - recon) ** 2, axis=1)
