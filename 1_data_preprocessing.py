"""
Preprocess the raw census data and save a clean dataset for training.
- Loads raw files from TakeHomeProject
- Normalizes the income label to a binary 0/1 target
- Trims categorical strings and preserves the weight column
- Saves cleaned data to preprocessed/data_clean.parquet and a meta file with column info
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import pandas as pd


# Raw input paths
RAW_COLS = Path("raw_data/census-bureau.columns")
RAW_DATA = Path("raw_data/census-bureau.data")
NA_TOKENS = [" ?"]

# Output paths
OUTPUT_DIR = Path("preprocessed")
OUTPUT_DATA = OUTPUT_DIR / "data_clean.parquet"
OUTPUT_META = OUTPUT_DIR / "meta.json"
# Manual drops based on domain intuition
MANUAL_DROP = {
    "detailed industry recode",
    "detailed occupation recode",
    "migration prev res in sunbelt",
    "migration code-change in msa",
    "migration code-change in reg",
    "migration code-move within reg",
    "fill inc questionnaire for veteran's admin",
    "reason for unemployment",
}


def load_raw(data_path: Path = RAW_DATA, cols_path: Path = RAW_COLS) -> pd.DataFrame:
    """Read the raw census data with column names and basic NA handling."""
    column_names = cols_path.read_text().strip().splitlines()
    return pd.read_csv(
        data_path,
        header=None,
        names=column_names,
        na_values=NA_TOKENS,
        skipinitialspace=True,
    )


def normalize_label(df: pd.DataFrame) -> pd.DataFrame:
    """Add a binary label column (label_bin) mapping income <=50k to 0 and >50k to 1."""
    df = df.copy()
    df["label_bin"] = df["label"].str.strip().str.replace(".", "", regex=False)
    df["label_bin"] = df["label_bin"].replace({"- 50000": 0, "50000+": 1})
    return df


def trim_strings(df: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace from string columns to reduce cardinality noise."""
    df = df.copy()
    obj_cols = df.select_dtypes(include=["object"]).columns
    for col in obj_cols:
        df[col] = df[col].str.strip()
    return df


def drop_low_signal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop manually specified low-signal columns.
    Leaves label/weight intact.
    """
    df = df.copy()
    protected = {"label", "label_bin", "weight"}
    manual_drop = {c for c in MANUAL_DROP if c in df.columns} - protected
    to_drop = sorted(manual_drop)

    if to_drop:
        print(f"Dropping {len(to_drop)} low-signal columns: {to_drop}")
        df = df.drop(columns=to_drop)
    else:
        print("No low-signal columns dropped.")

    return df


def save_meta(df: pd.DataFrame) -> None:
    """Persist basic column metadata to help downstream training scripts."""
    cat_cols = [c for c in df.select_dtypes(include=["object"]).columns if c != "label"]
    num_cols = df.columns.difference(cat_cols + ["label_bin", "weight", "label"]).tolist()
    meta: Dict[str, List[str] | str | int] = {
        "categorical_columns": cat_cols,
        "numeric_columns": num_cols,
        "label_column": "label_bin",
        "weight_column": "weight",
        "rows": int(len(df)),
    }
    OUTPUT_META.write_text(json.dumps(meta, indent=2))


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_raw()
    df = trim_strings(df)
    df = normalize_label(df)
    df = drop_low_signal_features(df)

    # Keep all original feature columns, plus label_bin and weight for training
    df.to_parquet(OUTPUT_DATA, index=False)
    save_meta(df)

    print(f"Saved cleaned data to {OUTPUT_DATA} with {len(df):,} rows.")
    print(f"Saved metadata to {OUTPUT_META}.")


if __name__ == "__main__":
    main()
