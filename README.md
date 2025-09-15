# CTR Prediction – End-to-End Pipeline

This repository presents an end-to-end pipeline for Click-Through Rate (CTR) prediction on large-scale, highly imbalanced data, covering EDA, feature engineering, model calibration, and evaluation.

## Data Access

Raw datasets (`train.parquet`, `test.parquet`) are excluded from this repository due to competition restrictions.  
See [`data/README.md`](data/README.md) for dataset schema, access policy, and structure.  
All scripts assume local data access via the `data/` folder or `DATA_DIR` environment variable.

## 1. EDA Summary

- **Class imbalance**: Positive rate (`clicked=1`) ≈ 1.9% → stratified splitting & weighted metrics used
- Data inspection revealed several near‑constant or highly correlated features → flagged for potential drop/clip
- Transformation experiments (clip, drop, flag) showed that **clipping outliers** improved PR‑AUC (+1.7%); others had minimal or negative effect
- Among features with zero gain in all folds, both permutation and SHAP confirmed **no features** qualified for removal
- Sequence-derived features were engineered (e.g., length, repetition, token position), but **yielded negligible lift** → excluded from final model
- Cleaned datasets saved as Parquet files for reproducibility

---

## 2. Modeling Summary

- **Train / Valid / Holdout split**: 3-fold CV used for all tuning and calibration.  
  → `valid` used only for early stopping and stage gating (never for model selection).  
  → `holdout` evaluated once at final stage.
- **Best model**: Tuned LightGBM + Isotonic Calibration
  | Model                  | AP     | LogLoss | WLL    | MeanPred |
  |------------------------|--------|---------|--------|----------|
  | Final (Calibrated)     | 0.0586 | 0.0916  | 2.0604 | 0.0135   |
  | Final (Uncalibrated)   | 0.0634 | 0.2168  | 1.0142 | 0.1480   |
  
- Isotonic calibration significantly reduced overconfidence, improving LogLoss and aligning predicted probabilities to the base rate (~1.9%).
- **Calibration trade-off**: Expected WLL(50:50) increase due to lower confidence on rare positive class.  
  → Chosen for more reliable probability estimates in high-imbalance context.
- **Protocol**: Fully leak-free. All tuning and selection used train CV only; holdout used once.


📎 Details: [`notebooks/EDA.ipynb`](notebooks/01_EDA.ipynb), [`notebooks/Modeling.ipynb`](notebooks/02_Modeling.ipynb)  
📄 Report: [`reports/Reports.md`](reports/Reports.md)

---

## 3. Reproducibility

```bash
# Install
git clone https://github.com/yourname/ctr-pipeline.git
cd ctr-pipeline
pip install -r requirements.txt

# Run EDA notebook
jupyter notebook notebooks/EDA.ipynb
```


---
## Related Work

This repository is accompanied by a thesis project on **Hybrid Bayesian Networks for Early COPD Screening**, which explores the combination of generative and discriminative models in a high-imbalance healthcare context.

➡️ See [`thesis/README.md`](thesis/README.md) for details.













