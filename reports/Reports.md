# 1. Exploratory Data Analysis (EDA)

### 1.1 Dataset Overview

- **Train**: 7.28M rows, **Pos rate** ≈ 1.91%
- **Valid**: 1.82M rows, **Pos rate** ≈ 1.91%
- **Holdout**: 1.61M rows, **Pos rate** ≈ 1.91%
- **Total features**: 119 columns (numerical, categorical, and sequence)
- **Note**: Most features (`feat_e_*`, `l_feat_*`, etc.) lack semantic meaning — treated purely based on distributional/statistical properties.

The target variable `clicked` is highly imbalanced (~1.9% positive class), requiring stratified splitting and weighted evaluation metrics.

---

### 1.2 Feature Audit & Preprocessing Decision

- **Near-constant features**: e.g., `feat_l_20`, `feat_l_2` → flagged (std ≈ 0.0 or one unique value)
- **Redundancy reduction**:
  - Candidates identified → tested via flag/drop/clip variants
  - **Clipping** outliers improved PR‑AUC (+1.7%) → retained
  - Flagging/dropping → no benefit
- **Zero gain check**:
  - 61 features with gain=0 across all CV folds were evaluated via SHAP & permutation importance.
  - None were flagged for removal, but all were retained for completeness

---

### 1.3 Categorical vs Continuous Heuristic

- No feature schema provided — types inferred empirically
- For `l_feat_*` variables:  
  - ≤25 unique values → treated as categorical  
  - >25 unique values → treated as continuous

---

### 1.4 Multicollinearity Check

- **Pearson Correlation (|r| ≥ 0.995)**:  
  - `history_b_3` to `history_b_30` showed extreme correlation  
  - → 26 redundant features removed

- **Cramér’s V ≈ 1.0** (categoricals):  
  - `l_feat_3`, `l_feat_16` were perfect duplicates  
  - → removed

---

### 1.5 Sequence Features (`seq`)

- Engineered: sequence length, repetition count, first/last token positions  
- → **No lift** in model performance → excluded from final model  
  *(details in `EDA.ipynb`)*

---

### 1.6 Reproducibility

- Data Source: `data/train.parquet` (local)
- Split: 80/20 stratified on `clicked`, `random_state=42`
- Artifacts: `train.parquet`, `valid.parquet`, `holdout.parquet` stored for modeling
- Environment: Python 3.11+, Libraries: pandas, numpy, scipy, sklearn, seaborn

---

# 2. Modeling Summary

This section summarizes the model development process, including baseline benchmarking, tuning protocol, and final evaluation.

---

## 2.1 Baseline Comparison (Train CV)

| Model             | AP     | LogLoss | WLL    |
|-------------------|--------|---------|--------|
| LightGBM(Default) | 0.0425 | 0.1108  | 1.3332 |
| XGBoost(Default)  | 0.0400 | 0.1044  | 2.1253 |
| LogReg(OHE)       | 0.0400 | 0.6558  | 0.6640 |

LightGBM was selected as the tuning base model for its combination of accuracy (highest AP), calibration stability, and scalability on large datasets (~7M rows).  
XGBoost showed competitive metrics but was excluded from further tuning due to training cost.

---

## 2.2 Tuning Strategy (Leak-Free, Train-Only CV)

**Objective**  
Maximize ranking performance (**AP**) under class imbalance, while ensuring well-calibrated probabilities via **LogLoss** and **WLL (50:50)**.

**Tuning Protocol**
- Tuning and model selection used **3-fold Stratified CV** on `train_df` only.
- All calibration used **out-of-fold predictions**.
- Neither `valid_df` nor `holdout_df` were used in any tuning step.
- Final model was trained on full `train_df`, calibrated with Isotonic Regression, then evaluated once on `holdout_df`.

**Stages**
| Stage | Purpose         | Trials | Summary |
|-------|------------------|--------|---------|
| 1     | Wide exploration | 120    | Tuned learning rate, depth, sampling, regularization, imbalance handling |
| 2     | Focused tuning   | 40     | Refined top trials with narrowed ranges |
| 3     | Final training   | 1      | Full-train with best config + Isotonic calibration |

---

## 2.3 Final Evaluation (Holdout)

| Model                   | AP     | LogLoss | WLL    | MeanPred |
|-------------------------|--------|---------|--------|----------|
| Final + Isotonic (OOF)  | 0.0586 | 0.0916  | 2.0604 | 0.0135   |
| Final (Uncalibrated)    | 0.0634 | 0.2168  | 1.0142 | 0.1480   |

**Why Calibrated Model Was Chosen**
- Uncalibrated model achieved slightly higher AP, but severely overestimated the positive class (MeanPred = 14.8% vs actual ≈ 1.9%)
- Isotonic calibration adjusted the predictions closer to the true base rate, improving LogLoss and stabilizing WLL
- Final model is more suitable for thresholding and downstream KPI-based decisions in real-world deployment

---

## 2.4 Reproducibility

- Seed-controlled (`SEED=42`) with stratified splitting
- CV folds, tuning trials, final models, and predictions are all saved to versioned `artifacts/`
- Final model is trained and evaluated using strictly leak-free protocol


---
