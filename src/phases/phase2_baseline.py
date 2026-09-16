import argparse
import json
from pathlib import Path
import pandas as pd
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score

from src.models.isolation_forest_model import IsolationForestDetector
from src.models.autoencoder_model import AutoencoderDetector
from src.models.one_class_svm_model import OneClassSVMDetector
from src.models.dagmm_model import DAGMMDetector
from src.utils.feature_prep import load_scaler, transform_features, LABEL_COL, BENIGN_LABEL

PRIMARY_FP_TOLERANCE = "0.1"
STEALTHY_ATTACK_KEYWORDS = [
    "RECON", "SQLINJECTION", "COMMANDINJECTION", "XSS", "BACKDOOR",
    "MITM", "DNS_SPOOFING", "VULNERABILITYSCAN", "DICTIONARYBRUTEFORCE",
    "BROWSERHIJACKING", "UPLOADING_ATTACK",
    # DataSense (secondary dataset) category-level labels
    "MALWARE", "WEB", "BRUTEFORCE",
]

MODEL_FILENAMES = {
    "isolation_forest": "isolation_forest.joblib",
    "autoencoder": "autoencoder.keras",
    "one_class_svm": "one_class_svm.joblib",
    "dagmm": "dagmm",
}


def evaluate_at_threshold(y_true, scores, cutoff, label):
    y_pred = (scores < cutoff).astype(int)
    f1 = f1_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    print(f"\n--- Threshold: {label} (cutoff = {cutoff:.5f}) ---")
    print(f"F1: {f1:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f}")
    print(classification_report(y_true, y_pred, target_names=["Normal", "Attack"]))
    return y_pred


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default=Path("data/processed"), type=Path,
                         help="Folder containing *_cleaned.csv files")
    parser.add_argument("--model", default="isolation_forest",
                         choices=["isolation_forest", "autoencoder", "one_class_svm", "dagmm"])
    parser.add_argument("--exclude-attack", default=None)
    parser.add_argument("--results-dir", default=Path("results"), type=Path)
    args = parser.parse_args()

    cleaned_files = sorted(args.input_dir.glob("*_cleaned.csv"))
    df = pd.concat([pd.read_csv(p) for p in cleaned_files], ignore_index=True)
    if args.exclude_attack:
        df = df[df[LABEL_COL] != args.exclude_attack].reset_index(drop=True)

    print(f"Total rows: {len(df):,}")
    scaler = load_scaler(results_dir=args.results_dir)
    X = transform_features(df, scaler)
    y_true = (df[LABEL_COL] != BENIGN_LABEL).astype(int)

    model_path = args.results_dir / MODEL_FILENAMES[args.model]

    if args.model == "isolation_forest":
        detector = IsolationForestDetector().load(model_path)
    elif args.model == "autoencoder":
        detector = AutoencoderDetector(input_dim=X.shape[1]).load(model_path)
    elif args.model == "one_class_svm":
        detector = OneClassSVMDetector().load(model_path)
    elif args.model == "dagmm":
        detector = DAGMMDetector(input_dim=X.shape[1]).load(model_path)

    scores = detector.anomaly_score(X)

    with open(args.results_dir / "thresholds.json") as f:
        thresholds = json.load(f)[args.model]

    best_pred = None
    for fp_rate_str, cutoff in thresholds.items():
        label = f"{float(fp_rate_str):.0%} FP tolerance"
        y_pred = evaluate_at_threshold(y_true, scores, cutoff, label)
        if fp_rate_str == PRIMARY_FP_TOLERANCE:
            best_pred = y_pred

    df["_predicted_attack"] = best_pred
    per_label = df.groupby(LABEL_COL)["_predicted_attack"].mean().sort_values(ascending=False)
    print("\n=== Detection rate by attack category (PRIMARY threshold) ===")
    print(per_label.to_string())

    is_stealthy = df[LABEL_COL].apply(lambda lbl: any(kw in lbl for kw in STEALTHY_ATTACK_KEYWORDS))
    attack_mask = df[LABEL_COL] != BENIGN_LABEL
    flood_mask = attack_mask & ~is_stealthy
    stealthy_mask = attack_mask & is_stealthy
    flood_recall = df.loc[flood_mask, "_predicted_attack"].mean()
    stealthy_recall = df.loc[stealthy_mask, "_predicted_attack"].mean()
    print(f"\nFlood mean detection: {flood_recall:.1%}")
    print(f"Stealthy mean detection: {stealthy_recall:.1%}")
    print(f"Gap: {flood_recall - stealthy_recall:.1%}")


if __name__ == "__main__":
    main()