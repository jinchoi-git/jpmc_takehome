"""
Segmentation model for the census dataset.
- Loads preprocessed data from 1_data_preprocessing.py
- Builds a one-hot + scaled feature matrix
- Fits MiniBatchKMeans for efficiency
- Outputs cluster assignments and a summary (size, >50k rate)
- Supports --eval-only to load saved model and just produce outputs/plots
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, Tuple

import joblib
import matplotlib

matplotlib.use("Agg")  # safe for non-GUI environments

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Paths
DATA_PATH = Path("preprocessed/data_clean.parquet")
META_PATH = Path("preprocessed/meta.json")
OUTPUT_DIR = Path("artifacts/segmentation")
ASSIGNMENTS_PATH = OUTPUT_DIR / "segments.csv"
SUMMARY_PATH = OUTPUT_DIR / "segments_summary.csv"
MODEL_PATH = OUTPUT_DIR / "segments_model.joblib"

# Clustering config
N_CLUSTERS = 6
RANDOM_STATE = 42
BATCH_SIZE = 2048
MAX_ITER = 200


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Skip training and only load saved model for evaluation/plots.",
    )
    return parser.parse_args(list(argv))


def load_data() -> Tuple[pd.DataFrame, list[str], list[str]]:
    """Load cleaned data and metadata produced by 1_data_preprocessing.py."""
    df = pd.read_parquet(DATA_PATH)
    meta = json.loads(META_PATH.read_text())
    cat_cols = meta["categorical_columns"]
    num_cols = meta["numeric_columns"]
    return df, cat_cols, num_cols


def build_preprocess(cat_cols: list[str], num_cols: list[str]) -> ColumnTransformer:
    num_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    cat_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
        ]
    )


def fit_and_save(preprocess: ColumnTransformer, X: pd.DataFrame) -> Tuple[MiniBatchKMeans, pd.DataFrame]:
    kmeans = MiniBatchKMeans(
        n_clusters=N_CLUSTERS,
        batch_size=BATCH_SIZE,
        random_state=RANDOM_STATE,
        max_iter=MAX_ITER,
    )
    X_enc = preprocess.fit_transform(X)
    kmeans.fit(X_enc)
    joblib.dump({"preprocess": preprocess, "model": kmeans}, MODEL_PATH)
    return kmeans, X_enc


def load_model() -> Tuple[ColumnTransformer | None, MiniBatchKMeans | None]:
    if MODEL_PATH.exists():
        bundle = joblib.load(MODEL_PATH)
        return bundle["preprocess"], bundle["model"]
    return None, None


def summarize(df_out: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df_out.groupby("cluster")
        .agg(
            count=("cluster", "size"),
            weight_sum=("weight", "sum"),
            pos_rate=("label_bin", "mean"),
            pos_rate_weighted=(
                "label_bin",
                lambda s: (s * df_out.loc[s.index, "weight"]).sum()
                / df_out.loc[s.index, "weight"].sum(),
            ),
        )
        .reset_index()
        .sort_values("cluster")
    )
    return summary


def plot_cluster_sizes_weighted(summary: pd.DataFrame) -> None:
    ax = summary.set_index("cluster")["weight_sum"].plot.bar(color="steelblue", edgecolor="black")
    ax.set_title("Cluster Sizes (weighted)")
    ax.set_ylabel("Weighted count")
    for p in ax.patches:
        ax.annotate(
            f"{p.get_height():.0f}",
            (p.get_x() + p.get_width() / 2.0, p.get_height()),
            ha="center",
            va="bottom",
            fontsize=8,
        )
    plt.tight_layout()
    out = OUTPUT_DIR / "segments_cluster_sizes_weighted.png"
    plt.savefig(out)
    plt.close()
    print(f"Saved weighted cluster size plot to {out}")


def plot_pos_rate(summary: pd.DataFrame) -> None:
    ax = summary.set_index("cluster")[["pos_rate_weighted"]].plot.bar(
        color=["orange"], edgecolor="black", legend=False
    )
    ax.set_title("Weighted share of >50k by cluster")
    ax.set_ylabel("Weighted share")
    for p in ax.patches:
        ax.annotate(
            f"{p.get_height():.2f}",
            (p.get_x() + p.get_width() / 2.0, p.get_height()),
            ha="center",
            va="bottom",
            fontsize=8,
        )
    plt.tight_layout()
    out = OUTPUT_DIR / "segments_pos_rate.png"
    plt.savefig(out)
    plt.close()
    print(f"Saved weighted >50k rate plot to {out}")


def main():
    args = parse_args(sys.argv[1:])

    df, cat_cols, num_cols = load_data()
    X = df.drop(columns=["label", "label_bin"])

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.eval_only:
        preprocess, kmeans = load_model()
        if preprocess is None:
            print("No saved segmentation model found. Run without --eval-only to train first.")
            return
        X_enc = preprocess.transform(X)
    else:
        preprocess = build_preprocess(cat_cols, num_cols)
        kmeans, X_enc = fit_and_save(preprocess, X)

    clusters = kmeans.predict(X_enc)
    df_out = df.copy()
    df_out["cluster"] = clusters
    df_out.to_csv(ASSIGNMENTS_PATH, index=False)

    summary = summarize(df_out)
    summary.to_csv(SUMMARY_PATH, index=False)
    plot_cluster_sizes_weighted(summary)
    plot_pos_rate(summary)

    print(f"Saved cluster assignments to {ASSIGNMENTS_PATH}")
    print(f"Saved cluster summary to {SUMMARY_PATH}")
    print(summary[["cluster", "count", "pos_rate", "pos_rate_weighted"]])


if __name__ == "__main__":
    main()
