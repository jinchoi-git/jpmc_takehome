# Census Income Classification & Segmentation

End-to-end pipeline for cleaning the census dataset, training/evaluating classification models, and clustering households.

## Environment
- Python 3.10+
- Key libraries: pandas, numpy, scikit-learn, joblib, matplotlib, pyarrow (or fastparquet)

Install dependencies:
```bash
pip install pandas numpy scikit-learn joblib matplotlib pyarrow
```

## Data
- Raw inputs: `raw_data/census-bureau.data`, `raw_data/census-bureau.columns`
- Preprocessing keeps the original `weight` column, normalizes income labels to `label_bin` (0 for <=50k, 1 for >50k), strips whitespace from all string columns, and drops a small set of low-signal migration/occupation fields.
- Outputs from preprocessing are written to `preprocessed/` for downstream scripts:
  - `data_clean.parquet`
  - `meta.json` (categorical/numeric columns + label/weight info)

## Workflow

### 1) Preprocess
Clean data, encode the label, drop low-signal columns, and persist parquet + metadata.
```bash
python 1_data_preprocessing.py
```

### 2) Classification (predict >$50k)
Trains multiple models (logistic variants + random forests) with imputing + one-hot encoding, using sample weights. Stratified train/test split; weighted metrics reported.
```bash
python 2_classification.py              # train + evaluate + save artifacts
python 2_classification.py --eval-only  # load saved models to re-run eval/plots
```
Artifacts in `artifacts/classification/`:
- `<model>_classifier.joblib`
- Confusion matrix per model
- Comparison bar charts for ROC AUC, precision/recall/F1 on the >50k class

### 3) Segmentation (cluster households)
MiniBatchKMeans with scaled numeric features and one-hot encoded categorical features (imputation included). Supports eval-only mode to reuse the saved model.
```bash
python 3_segmentation.py              # fit + output clusters/plots
python 3_segmentation.py --eval-only  # reuse saved model to regenerate outputs
```
Artifacts in `artifacts/segmentation/`:
- `segments_model.joblib`
- `segments.csv` (cluster assignments) and `segments_summary.csv` (sizes + weighted >50k rate)
- Weighted cluster size plot and weighted >50k rate plot

## Ignored/untracked (size/noise)
- Model binaries and heavy outputs: `artifacts/**/*.joblib`, `artifacts/segmentation/segments.csv`
- Large deliverables/archives: `*.zip`, `*.docx`
- Notebook checkpoints/envs: `.ipynb_checkpoints/`, `.venv/`, `venv/`, `.env`
- Temp Python byproducts: `__pycache__/`, `*.py[cod]`

## Notes
- Scripts expect the raw data paths above; adjust constants at the top of each script if your layout differs.
- Classification metrics and plots use sample weights; segmentation summaries and plots use weighted counts/positive rates.
- Segmentation uses `N_CLUSTERS=6` (modifiable at the top of `3_segmentation.py`).
