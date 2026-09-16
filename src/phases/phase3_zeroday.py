import argparse
import json
import time
import tracemalloc
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
from sklearn.metrics import f1_score, precision_score, recall_score

from src.models.isolation_forest_model import IsolationForestDetector
from src.models.autoencoder_model import AutoencoderDetector
from src.models.one_class_svm_model import OneClassSVMDetector
from src.models.dagmm_model import DAGMMDetector
from src.utils.feature_prep import load_scaler, transform_features, LABEL_COL, BENIGN_LABEL

PRIMARY_FP_TOLERANCE = "0.1"
NORMAL_SAMPLE_SIZE = 20000

MODEL_FILENAMES = {
    "isolation_forest": "isolation_forest.joblib",
    "autoencoder": "autoencoder.keras",
    "one_class_svm": "one_class_svm.joblib",
    "dagmm": "dagmm",
}

RESOURCE_PROFILES = [
    {"label": "Unconstrained (gateway-class)", "cpu_cores": None, "batch_size": None},
    {"label": "Mid-tier device",               "cpu_cores": 2,    "batch_size": 1000},
    {"label": "Constrained sensor",             "cpu_cores": 1,    "batch_size": 100},
]


def set_cpu_affinity(n_cores):
    proc = psutil.Process()
    total_cores = psutil.cpu_count(logical=True)
    if n_cores is None:
        proc.cpu_affinity(list(range(total_cores)))
    else:
        proc.cpu_affinity(list(range(min(n_cores, total_cores))))


def run_inference(detector, X, cutoff, batch_size=None):
    tracemalloc.start()
    start = time.perf_counter()
    if batch_size is None:
        scores = detector.anomaly_score(X)
    else:
        scores = np.empty(len(X))
        for i in range(0, len(X), batch_size):
            chunk = X[i:i + batch_size]
            scores[i:i + batch_size] = detector.anomaly_score(chunk)
    y_pred = (scores < cutoff).astype(int)
    elapsed = time.perf_counter() - start
    _, peak_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return y_pred, elapsed, peak_memory


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default=Path("data/processed"), type=Path)
    parser.add_argument("--zero-day-attack", required=True)
    parser.add_argument("--model", default="isolation_forest",
                         choices=["isolation_forest", "autoencoder", "one_class_svm", "dagmm"])
    parser.add_argument("--results-dir", default=Path("results"), type=Path)
    args = parser.parse_args()

    cleaned_files = sorted(args.input_dir.glob("*_cleaned.csv"))
    df = pd.concat([pd.read_csv(p) for p in cleaned_files], ignore_index=True)

    zero_day_df = df[df[LABEL_COL] == args.zero_day_attack].copy()
    normal_df = df[df[LABEL_COL] == BENIGN_LABEL].sample(
        n=min(NORMAL_SAMPLE_SIZE, (df[LABEL_COL] == BENIGN_LABEL).sum()), random_state=42)
    eval_df = pd.concat([zero_day_df, normal_df], ignore_index=True)
    y_true = (eval_df[LABEL_COL] != BENIGN_LABEL).astype(int)

    print(f"Zero-day attack: {args.zero_day_attack} ({len(zero_day_df):,} rows)")
    scaler = load_scaler(results_dir=args.results_dir)
    X = transform_features(eval_df, scaler)

    with open(args.results_dir / "thresholds.json") as f:
        thresholds = json.load(f)[args.model]
    cutoff = thresholds[PRIMARY_FP_TOLERANCE]

    model_path = args.results_dir / MODEL_FILENAMES[args.model]

    if args.model == "isolation_forest":
        detector = IsolationForestDetector().load(model_path)
    elif args.model == "autoencoder":
        detector = AutoencoderDetector(input_dim=X.shape[1]).load(model_path)
    elif args.model == "one_class_svm":
        detector = OneClassSVMDetector().load(model_path)
    elif args.model == "dagmm":
        detector = DAGMMDetector(input_dim=X.shape[1]).load(model_path)

    results = []
    for profile in RESOURCE_PROFILES:
        print(f"\n=== {profile['label']} ===")
        set_cpu_affinity(profile["cpu_cores"])
        y_pred, elapsed, peak_memory = run_inference(detector, X, cutoff, profile["batch_size"])

        f1 = f1_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        zd_pred = y_pred[:len(zero_day_df)]
        zero_day_detection_rate = zd_pred.mean()
        normal_pred = y_pred[len(zero_day_df):]
        false_positive_rate = normal_pred.mean()

        print(f"Zero-day detection: {zero_day_detection_rate:.1%} | FPR: {false_positive_rate:.1%}")
        print(f"F1: {f1:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f}")
        print(f"Latency: {elapsed:.3f}s | Peak memory: {peak_memory/1024/1024:.2f} MB")

        results.append({
            "resource_profile": profile["label"],
            "cpu_cores": psutil.cpu_count(logical=True) if profile["cpu_cores"] is None else profile["cpu_cores"],
            "batch_size": profile["batch_size"] or "unbatched",
            "zero_day_detection_rate": zero_day_detection_rate,
            "false_positive_rate": false_positive_rate,
            "f1": f1, "precision": precision, "recall": recall,
            "latency_seconds": elapsed, "peak_memory_mb": peak_memory / 1024 / 1024,
        })

    set_cpu_affinity(None)
    results_df = pd.DataFrame(results)
    out_path = args.results_dir / f"phase3_{args.model}_{args.zero_day_attack}_results.csv"
    results_df.to_csv(out_path, index=False)
    print(f"\n{results_df.to_string(index=False)}")
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()