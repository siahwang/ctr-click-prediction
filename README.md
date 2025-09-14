# CTR Prediction – End-to-End Pipeline

This repository contains an end-to-end CTR prediction pipeline. The project includes exploratory data analysis (EDA), feature pruning, and model evaluation for high-class imbalance data.

## Data Access

Raw datasets (`train.parquet`, `test.parquet`) are excluded from this repository due to competition restrictions.  
See [`data/README.md`](data/README.md) for dataset schema, access policy, and structure.  
All scripts assume local data access via the `data/` folder or `DATA_DIR` environment variable.

## 1. EDA Summary

## 1. EDA Summary

- **Class imbalance**: Positive rate (`clicked=1`) ≈ 1.9% → stratified splitting & weighted metrics used
- Data inspection revealed several near‑constant or highly correlated features → flagged for potential drop/clip
- Transformation experiments (clip, drop, flag) showed that **clipping outliers** improved PR‑AUC (+1.7%); others had minimal or negative effect
- Among features with zero gain in all folds, both permutation and SHAP confirmed **no features** qualified for removal
- Sequence-derived features were engineered (e.g., length, repetition, token position), but **yielded negligible lift** → excluded from final model
- Cleaned datasets saved as Parquet files for reproducibility


📎 Details: [`notebooks/EDA.ipynb`](notebooks/01_EDA.ipynb) | [`reports/EDA.md`](reports/EDA.md)

---

## 2. Modeling Summary *(To be added)*

- Model: XGBoost baseline vs. feature-augmented models
- Evaluation metrics: AP, ROC-AUC, WLL (50:50), Portfolio Score
- Early results suggest importance of `history_*` but minimal gain from `seq` features

📎 Details: *Coming soon in* [`notebooks/Modeling.ipynb`](ctr-click-prediction/notebooks/Modeling.ipynb)

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








