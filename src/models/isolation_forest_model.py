"""
Isolation Forest wrapper for unsupervised anomaly detection.

Trained on normal (BENIGN) traffic only (Phase 1). At inference time,
scores every row by how 'anomalous' it looks; rows scored as anomalies
are flagged as potential attacks (Phase 2 / Phase 3).
"""

from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import IsolationForest

MODEL_PATH = Path("results/isolation_forest.joblib")


class IsolationForestDetector:
    def __init__(self, contamination: float = "auto", n_estimators: int = 100,
                 random_state: int = 42):
        self.model = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=random_state,
            n_jobs=-1,
        )

    def fit(self, X: np.ndarray):
        print(f"Training Isolation Forest on {X.shape[0]:,} normal samples "
              f"({X.shape[1]} features)...")
        self.model.fit(X)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Returns 1 for normal, -1 for anomaly (sklearn convention)."""
        return self.model.predict(X)

    def anomaly_score(self, X: np.ndarray) -> np.ndarray:
        """Lower score = more anomalous. Useful for thresholding /
        confidence-based decisions (per project's 'confidence score'
        threshold design)."""
        return self.model.decision_function(X)

    def save(self, path: Path = MODEL_PATH):
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path)
        print(f"Saved Isolation Forest model -> {path}")

    def load(self, path: Path = MODEL_PATH):
        self.model = joblib.load(path)
        return self
