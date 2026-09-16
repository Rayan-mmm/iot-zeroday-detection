# Unsupervised Anomaly Detection for Zero-Day IoT Cyberattacks Under Resource Constraints

## Research question
How effective are unsupervised machine learning methods at detecting zero-day IoT
cyberattacks under resource constraints?

## Models
- Isolation Forest
- Autoencoder
- One-Class SVM
- DAGMM

## Datasets
- Primary: CIC-IoT2023
- Secondary: DataSense IIoT 2025

## Experimental design
- Phase 1 — Train on normal traffic only
- Phase 2 — Test on known attacks (baseline)
- Phase 3 — Simulate zero-day by withholding one attack category, at varying
  resource levels (CPU/memory enforced via `psutil` and `tracemalloc`)

## Project structure
```
iot-zeroday-detection/
├── data/
│   ├── raw/            # original downloaded datasets (gitignored)
│   └── processed/      # cleaned/engineered data (gitignored)
├── src/
│   ├── utils/           # data loading, cleaning, resource monitoring
│   ├── models/          # model wrappers
│   └── phases/           # phase1_train.py, phase2_baseline.py, phase3_zeroday.py
├── notebooks/            # exploratory analysis
├── results/               # metrics, plots (gitignored contents)
├── logs/                    # run logs (gitignored contents)
├── requirements.txt
└── README.md
```

## Setup

```bash
# 1. Create and activate a virtual environment (Python 3.11 required — TensorFlow does not support 3.14)
py -3.11 -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS/Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. (VS Code) select the .venv interpreter
# Ctrl+Shift+P -> "Python: Select Interpreter" -> choose .venv
```

## Usage

### 1. Data Cleaning

Cleans raw traffic data: drops nulls, caps infinite values at the 99.9th percentile, removes duplicates, and splits off a normal-only subset for training.

```bash
# Clean all CIC-IoT2023 files in a folder and merge their normal-only rows
python -m src.utils.data_cleaning --input-dir data/raw --merge-normal

# Clean the DataSense dataset (different structure, separate script)
python -m src.utils.datasense_cleaning --benign data/raw_datasense/benign_samples_1sec.csv --attack data/raw_datasense/attack_samples_1sec.csv
```

### 2. Phase 1 — Train

Trains a chosen model exclusively on normal traffic, holds back 20% to calibrate detection thresholds (1%, 5%, 10% false-positive tolerance) without ever using attack labels.

```bash
python -m src.phases.phase1_train --normal-data data/processed/combined_normal_only.csv --model isolation_forest
# --model options: isolation_forest | autoencoder | one_class_svm | dagmm
# --results-dir results_datasense   (use when running on the DataSense dataset, to keep results separate)
```

### 3. Phase 2 — Baseline Evaluation

Evaluates the trained model against all known attack categories, at each calibrated threshold. `--exclude-attack` removes the category reserved for the zero-day simulation in Phase 3, so it isn't double-counted as both "known" and "unseen."

```bash
python -m src.phases.phase2_baseline --input-dir data/processed --model isolation_forest --exclude-attack MITM-ARPSPOOFING
```

### 4. Phase 3 — Zero-Day Simulation

Evaluates the trained model against only the withheld attack category, under three simulated resource profiles (unconstrained, mid-tier, constrained sensor), combining restricted CPU core affinity with batched inference to simulate limited memory.

```bash
python -m src.phases.phase3_zeroday --input-dir data/processed --zero-day-attack MITM-ARPSPOOFING --model isolation_forest
```

## Data

 raw dataset files in `data/raw/` (CIC-IoT2023) - `data/raw_datasense/` (DataSense):
- CIC-IoT2023: https://www.unb.ca/cic/datasets/iotdataset-2023.html
- DataSense IIoT 2025: https://www.unb.ca/cic/datasets/iiot-dataset-2025.html