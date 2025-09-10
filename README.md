# CTR Prediction – End-to-End Pipeline

This repository contains an end-to-end CTR prediction pipeline. The project includes exploratory data analysis (EDA), feature pruning, and model evaluation for high-class imbalance data.

## 1. EDA Summary

- **Class imbalance**: Positive rate (`clicked=1`) ≈ 1.9% → stratified splitting & weighted metrics
- **Missingness**: Most <0.2%, but `feat_e_3` >10% → flagged
- **Feature pruning**: Removed low-signal, constant, or highly correlated features
- **Sequence logs**: Engineered compact token-level features; excluded after no lift on hold-out

📎 Details: [`notebooks/EDA.ipynb`](notebooks/EDA.ipynb) | [`reports/EDA.md`](reports/EDA.md)

---

## 2. Modeling Summary *(To be added)*

- Model: XGBoost baseline vs. feature-augmented models
- Evaluation metrics: AP, ROC-AUC, WLL (50:50), Portfolio Score
- Early results suggest importance of `history_*` but minimal gain from `seq` features

📎 Details: *Coming soon in* [`notebooks/Modeling.ipynb`](notebooks/Modeling.ipynb)

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
## File Structure
- [`notebooks/EDA.ipynb`](notebooks/EDA.ipynb)  : Full exploratory analysis
- notebooks/Modeling.ipynb: Training & evaluation pipeline (WIP)
- [`reports/EDA.md`](reports/EDA.md)  : Summary of key preprocessing decisions
- reports/modeling.md: To be added

---
## Related Work

This repository is accompanied by a thesis project on **Hybrid Bayesian Networks for Early COPD Screening**, which explores the combination of generative and discriminative models in a high-imbalance healthcare context.

➡️ See [`thesis/README.md`](thesis/README.md) for details.


