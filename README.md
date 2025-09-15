# CTR Prediction – End-to-End Pipeline

This repository contains an end-to-end CTR prediction pipeline. The project includes exploratory data analysis (EDA), feature pruning, and model evaluation for high-class imbalance data.

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


📎 Details: [`notebooks/EDA.ipynb`](notebooks/01_EDA.ipynb) | [`reports/EDA.md`](reports/EDA.md)

---

## 2. Modeling Summary

- **Train / Valid / Holdout split**: 3-fold CV used for all tuning and calibration.  
  → `valid` used only for early stopping and stage gating (never for model selection).  
  → `holdout` evaluated once at final stage.
- **Best model**: Tuned LightGBM + Isotonic Calibration  
  → Uncalibrated model had higher AP (6.34%) but heavily overconfident (MeanPred 14.8%).  
  → Calibrated model improved LogLoss (↓0.09) and predicted closer to true click rate (MeanPred 1.35% vs actual 1.91%).
- **Calibration trade-off**: Expected WLL(50:50) increase due to lower confidence on rare positive class.  
  → Chosen for more reliable probability estimates in high-imbalance context.
- **Protocol**: Fully leak-free. All tuning and selection used train CV only; holdout used once.

📎 Details: [`notebooks/Modeling.ipynb`](notebooks/02_Modeling.ipynb) | [`reports/Modeling.md`](reports/Modeling.md)
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











