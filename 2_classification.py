"""
Train and evaluate multiple classifiers (logistic + random forest) for predicting
income >/$50k vs <=$50k using the preprocessed dataset. Supports an eval-only mode
that loads saved models to plot/evaluate without retraining.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, Tuple

import joblib
import matplotlib

matplotlib.use("Agg")  # safe for non-GUI environments
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# Paths
DATA_PATH = Path("preprocessed/data_clean.parquet")
META_PATH = Path("preprocessed/meta.json")
ARTIFACT_DIR = Path("artifacts/classification")
TEST_SIZE = 0.2
RANDOM_STATE = 42


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Skip training and only load saved models for evaluation/plots.",
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
    num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median"))])
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


def get_models() -> Dict[str, BaseEstimator]:
    """Define a small set of candidate models/hyperparameters to compare."""
    return {
        "log_reg_C0.5": LogisticRegression(max_iter=2000, n_jobs=-1, C=0.5),
        "log_reg_C1.0": LogisticRegression(max_iter=2000, n_jobs=-1, C=1.0),
        "log_reg_C2.0": LogisticRegression(max_iter=2000, n_jobs=-1, C=2.0),
        "rf_depth10_estim200": RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            n_jobs=-1,
            random_state=42,
        ),
        "rf_depthNone_estim300": RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            n_jobs=-1,
            random_state=42,
        ),
    }


def evaluate_and_plot(
    models: Dict[str, Pipeline],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    w_test: pd.Series,
) -> None:
    metrics = []

    for name, clf in models.items():
        print(f"\n=== Evaluating {name} ===")
        y_pred = clf.predict(X_test)
        y_proba = clf.predict_proba(X_test)[:, 1]

        print("Validation classification report (weighted):")
        print(classification_report(y_test, y_pred, sample_weight=w_test))
        auc = roc_auc_score(y_test, y_proba, sample_weight=w_test)
        print(f"Validation ROC AUC (weighted): {auc:.4f}")
        cm = confusion_matrix(y_test, y_pred, sample_weight=w_test)

        prec1 = cm[1, 1] / (cm[1, 1] + cm[0, 1] + 1e-9)
        rec1 = cm[1, 1] / (cm[1, 0] + cm[1, 1] + 1e-9)
        f1_1 = 2 * prec1 * rec1 / (prec1 + rec1 + 1e-9)

        metrics.append(
            {
                "model": name,
                "roc_auc": auc,
                "recall_1": rec1,
                "precision_1": prec1,
                "f1_1": f1_1,
            }
        )

        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
        disp.plot(values_format=".0f", cmap="Blues")
        plt.title(f"Confusion Matrix - {name}")
        plt.tight_layout()
        cm_path = ARTIFACT_DIR / f"{name}_confusion_matrix.png"
        plt.savefig(cm_path)
        plt.close()
        print(f"Saved confusion matrix plot to {cm_path}")

    # Comparison plots
    if metrics:
        mdf = pd.DataFrame(metrics).set_index("model")

        def plot_bar(series: pd.Series, title: str, fname: str) -> None:
            ax = series.plot.bar(color="steelblue", edgecolor="black")
            ax.set_title(title)
            ax.set_ylabel(series.name)
            # annotate values on top of bars
            for p in ax.patches:
                ax.annotate(
                    f"{p.get_height():.2f}",
                    (p.get_x() + p.get_width() / 2.0, p.get_height()),
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    rotation=0,
                )
            plt.tight_layout()
            out = ARTIFACT_DIR / fname
            plt.savefig(out)
            plt.close()
            print(f"Saved {title} to {out}")

        plot_bar(mdf["roc_auc"], "Validation ROC AUC by model", "compare_roc_auc.png")
        plot_bar(mdf["recall_1"], "Recall on >50k class by model", "compare_recall_positive.png")
        plot_bar(mdf["precision_1"], "Precision on >50k class by model", "compare_precision_positive.png")
        plot_bar(mdf["f1_1"], "F1 on >50k class by model", "compare_f1_positive.png")


def make_pipeline(preprocess_template: ColumnTransformer, model: BaseEstimator) -> Pipeline:
    steps = [("preprocess", clone(preprocess_template)), ("model", model)]
    return Pipeline(steps)


def train_models(
    preprocess_template: ColumnTransformer,
    model_defs: Dict[str, BaseEstimator],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    w_train: pd.Series,
) -> Dict[str, Pipeline]:
    trained = {}
    for name, model in model_defs.items():
        print(f"\n=== Training {name} ===")
        clf = make_pipeline(preprocess_template, model)
        clf.fit(X_train, y_train, model__sample_weight=w_train)
        out_path = ARTIFACT_DIR / f"{name}_classifier.joblib"
        joblib.dump(clf, out_path)
        print(f"Saved {name} model to {out_path}")
        trained[name] = clf
    return trained


def load_saved_models() -> Dict[str, Pipeline]:
    loaded = {}
    for name in get_models().keys():
        path = ARTIFACT_DIR / f"{name}_classifier.joblib"
        if path.exists():
            loaded[name] = joblib.load(path)
            print(f"Loaded saved model: {path}")
        else:
            print(f"Model not found, skipping: {path}")
    return loaded


def main():
    args = parse_args(sys.argv[1:])

    df, cat_cols, num_cols = load_data()

    y = df["label_bin"]
    w = df["weight"]
    X = df.drop(columns=["label", "label_bin"])

    X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
        X, y, w, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    if args.eval_only:
        trained_models = load_saved_models()
        if not trained_models:
            print("No saved models found. Run without --eval-only to train first.")
            return
    else:
        preprocess = build_preprocess(cat_cols, num_cols)
        trained_models = train_models(preprocess, get_models(), X_train, y_train, w_train)

    evaluate_and_plot(trained_models, X_test, y_test, w_test)


if __name__ == "__main__":
    main()
    
