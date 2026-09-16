import argparse
from pathlib import Path

import numpy as np
import pandas as pd

LABEL_COL = "Label"
BENIGN_LABEL = "BENIGN"

# Percentile used to cap extreme / inf values instead of dropping rows.
# 99.9th percentile keeps flood-attack rows (which legitimately have very
# high Rate values) while neutralising the literal inf entries.
CAP_PERCENTILE = 99.9


def load_raw(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def report(df: pd.DataFrame, stage: str) -> None:
    print(f"\n--- {stage} ---")
    print(f"Rows: {len(df):,} | Columns: {df.shape[1]}")
    print(f"Label counts (top 5):\n{df[LABEL_COL].value_counts().head()}")


def drop_null_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Only a handful of rows have nulls (Std/Variance) — safe to drop."""
    before = len(df)
    df = df.dropna()
    dropped = before - len(df)
    if dropped:
        print(f"Dropped {dropped} rows containing null values.")
    return df


def cap_infinite_and_extreme_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replace inf with NaN, then cap every numeric column at the
    CAP_PERCENTILE so extreme legitimate values (e.g. flood attack Rate)
    are preserved but capped rather than discarded.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    for col in numeric_cols:
        n_inf = np.isinf(df[col]).sum()
        if n_inf:
            print(f"Column '{col}': {n_inf} inf values -> capping at "
                  f"{CAP_PERCENTILE}th percentile")
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)

        cap_value = df[col].quantile(CAP_PERCENTILE / 100)
        n_capped = (df[col] > cap_value).sum()
        if n_capped:
            df[col] = df[col].clip(upper=cap_value)

    # Any NaNs created by the inf replacement (if the max itself was inf)
    # get filled with the capped max rather than dropped.
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].max())
    return df


def drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates()
    dropped = before - len(df)
    if dropped:
        print(f"Dropped {dropped} exact duplicate rows "
              f"({dropped/before:.1%} of data).")
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    report(df, "Raw input")
    df = drop_null_rows(df)
    df = cap_infinite_and_extreme_values(df)
    df = drop_duplicates(df)
    report(df, "Cleaned output")
    return df


def split_normal_only(df: pd.DataFrame) -> pd.DataFrame:
    """For Phase 1: models train on normal (BENIGN) traffic only."""
    normal_df = df[df[LABEL_COL] == BENIGN_LABEL].copy()
    print(f"\nNormal-only subset: {len(normal_df):,} rows "
          f"({len(normal_df)/len(df):.1%} of cleaned data)")
    return normal_df


def process_one_file(input_path: Path, outdir: Path) -> pd.DataFrame:
    """Clean a single CSV, save its outputs, and return the normal-only subset
    (so the caller can merge it with other files' normal-only subsets)."""
    print(f"\n{'='*60}\nProcessing: {input_path.name}\n{'='*60}")
    df = load_raw(input_path)
    df_clean = clean(df)

    stem = input_path.stem
    full_out = outdir / f"{stem}_cleaned.csv"
    normal_out = outdir / f"{stem}_normal_only.csv"

    df_clean.to_csv(full_out, index=False)
    print(f"Saved cleaned data -> {full_out}")

    df_normal = split_normal_only(df_clean)
    df_normal.to_csv(normal_out, index=False)
    print(f"Saved normal-only data -> {normal_out}")

    return df_normal


def main():
    parser = argparse.ArgumentParser(description="Clean CIC-IoT2023 CSV files")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--input", type=Path,
                        help="Path to a single raw MERGED_CSV file, e.g. data/raw/Merged01.csv")
    group.add_argument("--input-dir", type=Path,
                        help="Path to a folder containing multiple raw MERGED_CSV files, "
                             "e.g. data/raw  (processes every .csv in the folder)")
    parser.add_argument("--outdir", default=Path("data/processed"), type=Path,
                         help="Output directory for cleaned files")
    parser.add_argument("--merge-normal", action="store_true",
                         help="When used with --input-dir, also save a single combined "
                              "normal_only file merging every processed file's BENIGN rows")
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    if args.input:
        process_one_file(args.input, args.outdir)
        return

    csv_files = sorted(args.input_dir.glob("*.csv"))
    if not csv_files:
        print(f"No CSV files found in {args.input_dir}")
        return

    print(f"Found {len(csv_files)} CSV file(s) in {args.input_dir}")
    normal_frames = []
    for csv_path in csv_files:
        normal_df = process_one_file(csv_path, args.outdir)
        normal_frames.append(normal_df)

    if args.merge_normal:
        combined = pd.concat(normal_frames, ignore_index=True)
        combined_out = args.outdir / "combined_normal_only.csv"
        combined.to_csv(combined_out, index=False)
        print(f"\n{'='*60}")
        print(f"Combined normal-only dataset: {len(combined):,} rows "
              f"from {len(csv_files)} file(s)")
        print(f"Saved -> {combined_out}")


if __name__ == "__main__":
    main()