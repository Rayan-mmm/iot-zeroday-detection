import argparse
from pathlib import Path

import numpy as np
import pandas as pd

LABEL_COL = "Label"
BENIGN_LABEL = "BENIGN"
CAP_PERCENTILE = 99.9

# Non-feature identifier/metadata columns -- not used as model input
METADATA_COLS = [
    "device_name", "device_mac", "label_full",
    "label1", "label2", "label3", "label4",
    "timestamp", "timestamp_start", "timestamp_end",
]

# List-like string columns -- dropped in favour of their numeric "_count"
# companion column, which already exists for each of these
LIST_LIKE_COLS = [
    "log_data-types",
    "network_ips_all", "network_ips_dst", "network_ips_src",
    "network_macs_all", "network_macs_dst", "network_macs_src",
    "network_ports_all", "network_ports_dst", "network_ports_src",
    "network_protocols_all", "network_protocols_dst", "network_protocols_src",
]


def load_and_combine(benign_path: Path, attack_path: Path) -> pd.DataFrame:
    print(f"Loading benign data from {benign_path}...")
    benign_df = pd.read_csv(benign_path)
    print(f"  {len(benign_df):,} benign rows")

    print(f"Loading attack data from {attack_path}...")
    attack_df = pd.read_csv(attack_path)
    print(f"  {len(attack_df):,} attack rows")

    combined = pd.concat([benign_df, attack_df], ignore_index=True)
    print(f"Combined: {len(combined):,} total rows")
    return combined


def build_label_column(df: pd.DataFrame) -> pd.DataFrame:
    """Use label2 (attack category) as the primary Label column, matching
    the role CIC-IoT2023's flat Label column played. Benign rows already
    have label2='benign'; standardise to 'BENIGN' for consistency with the
    primary dataset's convention."""
    df[LABEL_COL] = df["label2"].str.upper()
    df.loc[df[LABEL_COL] == "BENIGN", LABEL_COL] = BENIGN_LABEL
    return df


def drop_non_feature_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop identifier/metadata and list-like string columns, keeping only
    the new Label column and numeric feature columns."""
    cols_to_drop = [c for c in METADATA_COLS if c in df.columns]
    cols_to_drop += [c for c in LIST_LIKE_COLS if c in df.columns]
    print(f"Dropping {len(cols_to_drop)} non-feature columns "
          f"(metadata + list-like string columns).")
    return df.drop(columns=cols_to_drop)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    print(f"\n--- Raw combined ---")
    print(f"Rows: {len(df):,} | Columns: {df.shape[1]}")

    df = build_label_column(df)
    df = drop_non_feature_columns(df)

    before = len(df)
    df = df.dropna()
    dropped = before - len(df)
    if dropped:
        print(f"Dropped {dropped} rows containing null values.")

    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns]
    for col in numeric_cols:
        n_inf = np.isinf(df[col]).sum()
        if n_inf:
            print(f"Column '{col}': {n_inf} inf values -> capping at "
                  f"{CAP_PERCENTILE}th percentile")
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)
        cap_value = df[col].quantile(CAP_PERCENTILE / 100)
        df[col] = df[col].clip(upper=cap_value)
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].max())

    before = len(df)
    df = df.drop_duplicates()
    dropped = before - len(df)
    if dropped:
        print(f"Dropped {dropped} exact duplicate rows "
              f"({dropped/before:.1%} of data).")

    print(f"\n--- Cleaned output ---")
    print(f"Rows: {len(df):,} | Columns: {df.shape[1]}")
    print(f"Label counts:\n{df[LABEL_COL].value_counts()}")
    return df


def main():
    parser = argparse.ArgumentParser(description="Clean DataSense IIoT 2025 dataset")
    parser.add_argument("--benign", required=True, type=Path,
                         help="Path to benign_samples_Xsec.csv")
    parser.add_argument("--attack", required=True, type=Path,
                         help="Path to attack_samples_Xsec.csv")
    parser.add_argument("--outdir", default=Path("data/processed_datasense"), type=Path)
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    df = load_and_combine(args.benign, args.attack)
    df_clean = clean(df)

    full_out = args.outdir / "datasense_cleaned.csv"
    df_clean.to_csv(full_out, index=False)
    print(f"\nSaved cleaned data -> {full_out}")

    df_normal = df_clean[df_clean[LABEL_COL] == BENIGN_LABEL].copy()
    normal_out = args.outdir / "datasense_normal_only.csv"
    df_normal.to_csv(normal_out, index=False)
    print(f"Normal-only subset: {len(df_normal):,} rows "
          f"({len(df_normal)/len(df_clean):.1%} of cleaned data)")
    print(f"Saved normal-only data -> {normal_out}")


if __name__ == "__main__":
    main()
