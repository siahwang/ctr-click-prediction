# EDA Summary for CTR Prediction

## Objective

Summarize key findings and data preprocessing decisions from exploratory analysis to support CTR prediction modeling.  
Focus is on handling class imbalance, missing values, multicollinearity, and potential data leakage.

---

---

## 1. Class Imbalance

- Target variable (`clicked`) shows **~1.9% positive rate** in training data
- Stratified split applied (80/20) to maintain label distribution

---

## 2. Missing Value Summary

- 96 features contain missing values; most have a missing rate below 0.2%
- `feat_e_3` missing rate over **10%** → not dropped, but flagged for monitoring 
  *(visual summary and null pattern shown in EDA.ipynb)*

---

## 3. Feature Categorization & Pruning

### 3.1 Categorical vs Continuous Heuristic

- `l_feat_*` variables assigned type based on unique value count:  
  - ≤25 unique values → categorical  
  - \>\25 unique values → continuous

### 3.2 Dropped Features

| Reason                | Variables                       |
|----------------------|----------------------------------|
| Constant values       | `l_feat_20`, `l_feat_23`        |
| Poor CTR separation   | `l_feat_2`, `l_feat_13`, `l_feat_26` |
| Duplicates            | `l_feat_17` ≈ `l_feat_9`         |

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
  *(details in EDA.ipynb Section 6)*

---

## 6. Reproducibility

- Source: `/Users/hwangsia/Desktop/open/train.parquet`
- Split: 80/20 stratified on `clicked`, random_state=42
- Artifacts: `train_df.csv`, `valid_df.csv` saved for modeling
- Python 3.11+, Libraries: pandas, numpy, scipy, sklearn, seaborn


