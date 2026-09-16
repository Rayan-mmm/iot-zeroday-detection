"""
One-Class SVM wrapper for unsupervised anomaly detection.

Uses SGDOneClassSVM (a linearly-scaling, stochastic gradient descent
formulation) instead of the classic OneClassSVM, which scales roughly
quadratically with training set size and becomes impractical on datasets
of this project's scale (100,000+ rows). SGDOneClassSVM solves the same
underlying problem -- drawing a boundary around normal data -- via a
different, scalable computational approach, and is the standard solution
used for large-scale one-class classification (built into scikit-learn
specifically for this use case).

Trained on normal (BENIGN) traffic only (Phase 1). Learns a boundary
around what normal traffic looks like; points falling outside that
boundary at inference time are flagged as potentially anomalous.

Interface matches IsolationForestDetector / AutoencoderDetector
(fit / anomaly_score / save / load) so Phase 1/2/3 scripts work with
any of the three models unchanged.
"""

from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import SGDOneClassSVM

MODEL_PATH = Path("results/one_class_svm.joblib")


class OneClassSVMDetector:
    def __init__(self, nu: float = 0.05, random_state: int = 42):
        # nu: upper bound on the fraction of training data allowed to fall
        # outside the boundary -- same role as Isolation Forest's
        # contamination parameter. 0.05 = tolerate ~5% margin during
        # training, a reasonable default that avoids an overly rigid boundary.
        self.model = SGDOneClassSVM(nu=nu, random_state=random_state)

    def fit(self, X: np.ndarray):
        print(f"Training One-Class SVM (SGD) on {X.shape[0]:,} normal samples "
              f"({X.shape[1]} features)...")
        self.model.fit(X)
        return self

    def anomaly_score(self, X: np.ndarray) -> np.ndarray:
        """Distance from the decision boundary. Positive = inside the
        boundary (normal-looking); negative = outside (anomalous).
        Lower score = more anomalous -- matches the convention used by
        IsolationForestDetector and AutoencoderDetector."""
        return self.model.decision_function(X)

    def save(self, path: Path = MODEL_PATH):
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path)
        print(f"Saved One-Class SVM model -> {path}")

    def load(self, path: Path = MODEL_PATH):
        self.model = joblib.load(path)
        return self
