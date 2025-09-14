# EDA Summary for CTR Prediction

## 1. Dataset Overview

- **Train**: 7.28M rows, **Pos rate** ≈ 1.91%
- **Valid**: 1.82M rows, **Pos rate** ≈ 1.91%
- **Holdout**: 1.61M rows, **Pos rate** ≈ 1.91%
- **Total features**: 119 columns (including numerical, categorical, and sequence field)
- **Note**: Most features (`feat_e_*`, `l_feat_*`, etc.) have no semantic description — handled purely based on statistical characteristics.

The target variable `clicked` is highly imbalanced (~1.9% positive class), requiring stratified splitting and weighted evaluation metrics during modeling.


---
## 2. Feature Audit & Preprocessing Decision

- **Near-constant features**: e.g., `feat_l_20`, `feat_l_2` → flagged (std ≈ 0 or one unique value)
- **Redundancy reduction**:
  - Candidates identified → evaluated via flag/drop/clip variants
  - Clipping improved PR‑AUC (+1.7%); drop/flag did not
- **Zero gain check**: 61 features with gain=0 in 3-fold CV → all retained (SHAP + permutation ≈ 0)

---
## 3. Categorical vs Continuous Heuristic

- No feature schema was provided; types inferred from distributional properties.
- `l_feat_*` variables assigned type based on unique value count:  
  - ≤25 unique values → categorical  
  - >25 unique values → continuous


---

## 4. Multicollinearity Check

### Pearson Correlation (|r| ≥ 0.995)

- `history_b_3` to `history_b_30` → extreme pairwise correlations (26 features removed)

### Cramér’s V (V ≈ 1.0)

- Perfect duplication in `l_feat_3`, `l_feat_16` → removed

---

## 5. Sequence Features (`seq`)

- Extracted features: sequence length, repetition count, position encoding  
- No performance gain → excluded from final model  
  *(details in EDA.ipynb)*

---

## 6. Reproducibility

- Source: `/Users/hwangsia/Desktop/open/train.parquet`
- Split: 80/20 stratified on `clicked`, random_state=42
- Artifacts: `train.parquet`, `valid.parquet`, `holdout.parquet` saved for modeling
- Python 3.11+, Libraries: pandas, numpy, scipy, sklearn, seaborn




