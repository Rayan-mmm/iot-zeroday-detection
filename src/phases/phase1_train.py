import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.models.isolation_forest_model import IsolationForestDetector
from src.models.autoencoder_model import AutoencoderDetector
from src.models.one_class_svm_model import OneClassSVMDetector
from src.models.dagmm_model import DAGMMDetector
from src.utils.feature_prep import fit_scaler, transform_features, LABEL_COL

FP_TOLERANCE_LEVELS = [0.01, 0.05, 0.10]

MODEL_FILENAMES = {
    "isolation_forest": "isolation_forest.joblib",
    "autoencoder": "autoencoder.keras",
    "one_class_svm": "one_class_svm.joblib",
    "dagmm": "dagmm",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--normal-data", required=True, type=Path)
    parser.add_argument("--model", default="isolation_forest",
                         choices=["isolation_forest", "autoencoder", "one_class_svm", "dagmm"])
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--results-dir", default=Path("results"), type=Path,
                         help="Where to save the trained model, scaler, and thresholds. "
                              "Use a separate dir (e.g. results_datasense) to keep datasets' "
                              "results independent.")
    args = parser.parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.normal_data)
    assert (df[LABEL_COL] == "BENIGN").all()
    print(f"Loaded {len(df):,} normal traffic rows.")

    df_train, df_val = train_test_split(df, test_size=args.val_split, random_state=42)
    print(f"Training on {len(df_train):,} rows, calibrating on {len(df_val):,} rows.")

    scaler = fit_scaler(df_train, results_dir=args.results_dir)
    X_train = transform_features(df_train, scaler)
    X_val = transform_features(df_val, scaler)

    model_path = args.results_dir / MODEL_FILENAMES[args.model]

    if args.model == "isolation_forest":
        detector = IsolationForestDetector()
    elif args.model == "autoencoder":
        detector = AutoencoderDetector(input_dim=X_train.shape[1], epochs=args.epochs)
    elif args.model == "one_class_svm":
        detector = OneClassSVMDetector()
    elif args.model == "dagmm":
        detector = DAGMMDetector(input_dim=X_train.shape[1], epochs=args.epochs)

    detector.fit(X_train)
    detector.save(model_path)

    val_scores = detector.anomaly_score(X_val)

    thresholds = {}
    for fp_rate in FP_TOLERANCE_LEVELS:
        cutoff = float(np.percentile(val_scores, fp_rate * 100))
        thresholds[str(fp_rate)] = cutoff
        print(f"  FP tolerance {fp_rate:.0%} -> cutoff = {cutoff:.5f}")

    thresholds_path = args.results_dir / "thresholds.json"
    all_thresholds = json.load(open(thresholds_path)) if thresholds_path.exists() else {}
    all_thresholds[args.model] = thresholds
    with open(thresholds_path, "w") as f:
        json.dump(all_thresholds, f, indent=2)
    print(f"Saved thresholds -> {thresholds_path}")


if __name__ == "__main__":
    main()