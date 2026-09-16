"""
Autoencoder wrapper for unsupervised anomaly detection.

Trained on normal (BENIGN) traffic only (Phase 1). Learns to compress and
reconstruct normal traffic patterns. At inference time, traffic that the
model struggles to reconstruct accurately (high reconstruction error) is
flagged as potentially anomalous.

Interface intentionally matches IsolationForestDetector (fit / anomaly_score /
save / load) so Phase 1/2/3 scripts work with either model unchanged.

NOTE ON SCORE DIRECTION: Isolation Forest's anomaly_score() returns LOWER
values for more anomalous rows. Reconstruction error works the opposite way
(HIGHER error = more anomalous). To keep the rest of the pipeline
model-agnostic, this wrapper's anomaly_score() returns the NEGATIVE
reconstruction error, so "lower score = more anomalous" holds for both
models, and the same threshold comparison (score < cutoff) works for either.
"""

from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

MODEL_PATH = Path("results/autoencoder.keras")


class AutoencoderDetector:
    def __init__(self, input_dim: int, encoding_dim: int = 8,
                 epochs: int = 20, batch_size: int = 256, random_state: int = 42):
        tf.random.set_seed(random_state)
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = self._build_model()

    def _build_model(self) -> keras.Model:
        """Symmetric encoder-decoder: gradually compress down to
        encoding_dim, then gradually expand back to input_dim."""
        inputs = keras.Input(shape=(self.input_dim,))

        # Encoder: compress
        x = layers.Dense(32, activation="relu")(inputs)
        x = layers.Dense(16, activation="relu")(x)
        bottleneck = layers.Dense(self.encoding_dim, activation="relu")(x)

        # Decoder: reconstruct
        x = layers.Dense(16, activation="relu")(bottleneck)
        x = layers.Dense(32, activation="relu")(x)
        outputs = layers.Dense(self.input_dim, activation="linear")(x)

        model = keras.Model(inputs, outputs)
        model.compile(optimizer="adam", loss="mse")
        return model

    def fit(self, X: np.ndarray):
        print(f"Training Autoencoder on {X.shape[0]:,} normal samples "
              f"({X.shape[1]} features -> {self.encoding_dim}-dim bottleneck)...")
        self.model.fit(
            X, X,  # autoencoder learns to reconstruct its own input
            epochs=self.epochs,
            batch_size=self.batch_size,
            shuffle=True,
            verbose=1,
        )
        return self

    def reconstruction_error(self, X: np.ndarray) -> np.ndarray:
        """Mean squared error between original and reconstructed input,
        per row. Higher = model struggled more to reconstruct = more
        anomalous."""
        X_reconstructed = self.model.predict(X, verbose=0)
        return np.mean(np.square(X - X_reconstructed), axis=1)

    def anomaly_score(self, X: np.ndarray) -> np.ndarray:
        """Returns NEGATIVE reconstruction error, so lower score = more
        anomalous -- matches IsolationForestDetector's convention."""
        return -self.reconstruction_error(X)

    def save(self, path: Path = MODEL_PATH):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(path)
        print(f"Saved Autoencoder model -> {path}")

    def load(self, path: Path = MODEL_PATH):
        self.model = keras.models.load_model(path)
        return self
