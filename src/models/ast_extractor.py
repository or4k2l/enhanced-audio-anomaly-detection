"""Audio Spectrogram Transformer (AST) embedding extractor.

Extracts 768-dimensional embeddings using MIT's pretrained AST model
(MIT/ast-finetuned-audioset-10-10-0.4593) via Hugging Face Transformers.
"""

import logging
import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "MIT/ast-finetuned-audioset-10-10-0.4593"
EMBEDDING_DIM = 768


class ASTEmbeddingExtractor:
    """Extract 768-dim embeddings using MIT's pretrained Audio Spectrogram Transformer.

    Uses the CLS token representation from the last hidden state as a
    fixed-size embedding for downstream anomaly detection.

    Args:
        model_name: Hugging Face model identifier.
        device: Compute device ('cpu' or 'cuda').
        sample_rate: Expected audio sample rate in Hz.

    Example:
        >>> extractor = ASTEmbeddingExtractor()
        >>> wave = np.random.randn(16000 * 10)
        >>> embedding = extractor.extract_embedding(wave)
        >>> assert embedding.shape == (768,)
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str = "cpu",
        sample_rate: int = 16000,
    ):
        self.model_name = model_name
        self.device = device
        self.sample_rate = sample_rate
        self._feature_extractor = None
        self._model = None

    def _load_model(self):
        """Lazily load the pretrained AST model from Hugging Face."""
        if self._model is not None:
            return

        try:
            from transformers import ASTFeatureExtractor, ASTModel
        except ImportError:
            raise ImportError(
                "Hugging Face Transformers is required for ASTEmbeddingExtractor. "
                "Install with: pip install transformers"
            )

        logger.info(f"Loading AST model: {self.model_name}")
        self._feature_extractor = ASTFeatureExtractor.from_pretrained(self.model_name)
        self._model = ASTModel.from_pretrained(self.model_name)
        self._model.eval()

        try:
            import torch

            self._model = self._model.to(self.device)
            self._torch = torch
        except ImportError:
            raise ImportError(
                "PyTorch is required for ASTEmbeddingExtractor. "
                "Install with: pip install torch"
            )

    def extract_embedding(self, waveform: np.ndarray) -> np.ndarray:
        """Extract a 768-dimensional CLS token embedding from a waveform.

        Args:
            waveform: Raw audio waveform as a 1D numpy array.

        Returns:
            768-dimensional embedding vector.
        """
        self._load_model()

        inputs = self._feature_extractor(
            waveform,
            sampling_rate=self.sample_rate,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with self._torch.no_grad():
            outputs = self._model(**inputs)

        cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()
        return cls_embedding

    def extract_batch(self, waveforms: list) -> np.ndarray:
        """Extract embeddings for a list of waveforms.

        Args:
            waveforms: List of raw audio waveforms.

        Returns:
            Array of shape (n_waveforms, 768).
        """
        embeddings = [self.extract_embedding(w) for w in waveforms]
        return np.stack(embeddings, axis=0)
