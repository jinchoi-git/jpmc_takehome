# Census Income Classification & Segmentation

## Environment
- Python 3.10+
- Key libraries: pandas, numpy, scikit-learn, joblib, matplotlib, pyarrow (or fastparquet)

Install dependencies (example):
```bash
pip install -r requirements.txt
# or
pip install pandas numpy scikit-learn joblib matplotlib pyarrow
```

## Data
- Raw files: `TakeHomeProject/census-bureau.data`, `TakeHomeProject/census-bureau.columns`
- Preprocessed outputs are written to `preprocessed/`
- Artifacts (models, plots) are written to `artifacts/classification/` and `artifacts/segmentation/`

## Workflow

### 1) Preprocess
Clean data, apply manual column drops, encode label, and save parquet/meta.
```bash
python 1_data_preprocessing.py
```
Outputs:
- `preprocessed/data_clean.parquet`
- `preprocessed/meta.json`

### 2) Classification
Trains/evaluates multiple models (logistic + random forest), saves models and plots.
```bash
python 2_classification.py            # train + eval
python 2_classification.py --eval-only  # eval/plots using saved models
```
Outputs in `artifacts/classification/`:
- `<model>_classifier.joblib`
- Confusion matrices per model
- Comparison plots: ROC AUC, precision/recall/F1 for >50k class

### 3) Segmentation
Clusters with MiniBatchKMeans, saves model, assignments, summaries, and weighted plots.
```bash
python 3_segmentation.py              # train + eval
python 3_segmentation.py --eval-only  # eval/plots using saved model
```
Outputs in `artifacts/segmentation/`:
- `segments_model.joblib`
- `segments.csv` (assignments) and `segments_summary.csv`
- Weighted cluster size plot and weighted >50k rate plot

## Notes
- Scripts expect the raw data paths as above; adjust paths in the scripts if your layout differs.
- Classification uses sample weights from the data; metrics are reported as weighted.
- Segmentation is set to `N_CLUSTERS=6` (adjust at the top of `3_segmentation.py` if desired).
