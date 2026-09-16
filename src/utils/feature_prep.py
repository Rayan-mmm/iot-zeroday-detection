from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

LABEL_COL = "Label"
BENIGN_LABEL = "BENIGN"


def get_feature_columns(df):
    return [c for c in df.columns if c != LABEL_COL]


def fit_scaler(df, results_dir=Path("results"), save=True):
    feature_cols = get_feature_columns(df)
    scaler = StandardScaler()
    scaler.fit(df[feature_cols])
    if save:
        results_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(scaler, results_dir / "feature_scaler.joblib")
    return scaler


def load_scaler(results_dir=Path("results")):
    return joblib.load(results_dir / "feature_scaler.joblib")


def transform_features(df, scaler):
    feature_cols = get_feature_columns(df)
    return scaler.transform(df[feature_cols])