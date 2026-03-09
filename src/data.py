from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
DEFAULT_CSV_CANDIDATES = [
    RAW_DATA_DIR / "paysim.csv",
    RAW_DATA_DIR / "PS_20174392719_1491204439457_log.csv",
]
DEFAULT_SAMPLE_CACHE_PATH = PROCESSED_DATA_DIR / "paysim_model_sample.csv"
DEFAULT_METADATA_CACHE_PATH = PROCESSED_DATA_DIR / "paysim_sampling_summary.json"
TARGET_COLUMN = "isFraud"

CSV_DTYPES = {
    "step": "int32",
    "type": "category",
    "amount": "float32",
    "nameOrig": "string",
    "oldbalanceOrg": "float32",
    "newbalanceOrig": "float32",
    "nameDest": "string",
    "oldbalanceDest": "float32",
    "newbalanceDest": "float32",
    "isFraud": "int8",
    "isFlaggedFraud": "int8",
}


def ensure_project_dirs() -> None:
    """Create the local folders used by the project."""
    for path in (RAW_DATA_DIR, PROCESSED_DATA_DIR):
        path.mkdir(parents=True, exist_ok=True)


def resolve_paysim_csv_path(csv_path: Path | None = None) -> Path:
    """Return the first available PaySim CSV path."""
    if csv_path is not None:
        if csv_path.exists():
            return csv_path
        raise FileNotFoundError(f"PaySim CSV not found at {csv_path}")

    for candidate in DEFAULT_CSV_CANDIDATES:
        if candidate.exists():
            return candidate

    candidate_list = ", ".join(str(path) for path in DEFAULT_CSV_CANDIDATES)
    raise FileNotFoundError(
        "PaySim CSV not found. Place the downloaded file at one of: "
        f"{candidate_list}"
    )


def load_paysim_sample(
    csv_path: Path | None = None,
    sample_cache_path: Path = DEFAULT_SAMPLE_CACHE_PATH,
    metadata_cache_path: Path = DEFAULT_METADATA_CACHE_PATH,
    non_fraud_frac: float = 0.02,
    chunk_size: int = 250_000,
    random_state: int = 42,
    force_rebuild: bool = False,
) -> tuple[pd.DataFrame, dict[str, float | int | str | bool]]:
    """
    Build a manageable training sample from the full PaySim CSV.

    Strategy:
    - keep every fraud case
    - keep a reproducible sample of non-fraud cases
    - cache the sampled frame and metadata for future runs
    """
    ensure_project_dirs()

    if (
        not force_rebuild
        and sample_cache_path.exists()
        and metadata_cache_path.exists()
    ):
        frame = pd.read_csv(sample_cache_path)
        with metadata_cache_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        metadata["loaded_from_cache"] = True
        return frame, metadata

    csv_path = resolve_paysim_csv_path(csv_path)
    rng = np.random.default_rng(random_state)

    sampled_chunks: list[pd.DataFrame] = []
    total_rows = 0
    fraud_rows = 0
    non_fraud_rows = 0

    for chunk in pd.read_csv(csv_path, chunksize=chunk_size, dtype=CSV_DTYPES):
        total_rows += len(chunk)

        fraud_chunk = chunk[chunk[TARGET_COLUMN] == 1]
        non_fraud_chunk = chunk[chunk[TARGET_COLUMN] == 0]

        fraud_rows += len(fraud_chunk)
        non_fraud_rows += len(non_fraud_chunk)

        if not fraud_chunk.empty:
            sampled_chunks.append(fraud_chunk)

        if not non_fraud_chunk.empty:
            sampled_non_fraud = non_fraud_chunk.sample(
                frac=non_fraud_frac,
                random_state=int(rng.integers(0, 1_000_000)),
            )
            if not sampled_non_fraud.empty:
                sampled_chunks.append(sampled_non_fraud)

    if not sampled_chunks:
        raise RuntimeError("No rows were loaded from the PaySim CSV.")

    frame = (
        pd.concat(sampled_chunks, ignore_index=True)
        .sample(frac=1.0, random_state=random_state)
        .reset_index(drop=True)
    )

    frame.to_csv(sample_cache_path, index=False)

    metadata: dict[str, float | int | str | bool] = {
        "source_csv": str(csv_path),
        "total_rows": int(total_rows),
        "fraud_rows": int(fraud_rows),
        "non_fraud_rows": int(non_fraud_rows),
        "sample_rows": int(len(frame)),
        "sample_fraud_rows": int(frame[TARGET_COLUMN].sum()),
        "sample_fraud_rate": float(frame[TARGET_COLUMN].mean()),
        "non_fraud_sampling_fraction": float(non_fraud_frac),
        "loaded_from_cache": False,
    }

    with metadata_cache_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    return frame, metadata


def train_valid_test_split(
    frame: pd.DataFrame,
    target_col: str = TARGET_COLUMN,
    test_size: float = 0.2,
    valid_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Create reproducible stratified train, validation, and test splits."""
    if target_col not in frame.columns:
        raise ValueError(f"Target column '{target_col}' was not found.")

    X = frame.drop(columns=target_col)
    y = frame[target_col]

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    valid_share_of_train = valid_size / (1 - test_size)
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train_full,
        y_train_full,
        test_size=valid_share_of_train,
        stratify=y_train_full,
        random_state=random_state,
    )

    return X_train, X_valid, X_test, y_train, y_valid, y_test
