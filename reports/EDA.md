# Exploratory Data Analysis (EDA)

## Objective
Understand the dataset structure and clean features for CTR (Click-Through Rate) prediction.

---

## Key Insights & Preprocessing Decisions

### 1. Severe Class Imbalance
- Positive class (clicked = 1) accounts for only **~1.9%** of the training data

---

### 2. Missing Value Summary

- 96 features contain missing values, but most have a missing rate below 0.2%
- Exception: `feat_e_3` has over 10% missing → considered adding a missingness indicator or dropping it

---

### 3. Feature Overview

#### 3.1 `l_feat_*` Variables

- All are integer-type, but actual semantics are undisclosed
- Therefore, variables were heuristically classified based on cardinality:
  - Unique values ≤ 25 → treated as **categorical**
  - Unique values > 25 → treated as **continuous**

**Dropped variables**
- Removed:  
  - Constant: `l_feat_20`, `l_feat_23`  
  - Low CTR separation: `l_feat_2`, `l_feat_13`, `l_feat_26`  
  - Duplicate: `l_feat_17` ≈ `l_feat_9`

#### 3.2 Summary of feat_* and history_* Groups

- `feat_e_3`: High missingness  
- `feat_c_*`: Several (e.g., `_1`, `_4`, `_5`, `_7`) have low variance → weak signal  
- `feat_a_*`: Mostly zeros → binarization considered  
- `history_b_*`: High inter-feature correlation → flagged for redundancy
---

## Feature Pruning: Multicollinearity (Correlation-based)

**Drop Criteria**
- **Pearson correlation**: |r| ≥ 0.995 AND p-value < 0.01
- **Cramér’s V**: V ≈ 1.0 AND p-value < 0.01 (bias-corrected)

**Dropped Features Summary**

| Criterion                  | Dropped Features                                    | Notes                      |
|---------------------------|-----------------------------------------------------|----------------------------|
| Cramér’s V ≈ 1.0 (p<0.01)  | `l_feat_3`, `l_feat_16`                              | Perfect duplicates         |
| Pearson \|r\| ≥ 0.995 (p<0.01) | `history_b_3` to `history_b_30` (total 26 features) | Strong multicollinearity   |

**Conclusion**
- Strong multicollinearity was concentrated in `history_b_*` features → removed accordingly
- Current feature set is streamlined and suitable for tree-based models

---
### Sequence (`seq`) Features

Explored CTR patterns in sequence length, repetition, and position; engineered compact features (e.g., length, first/last token one-hot), but excluded from final model due to minimal performance lift.
---

## Data Leakage Check

- **Categorical features**: No category mismatch between train and valid sets → ✅ No leakage
- **Continuous features**: KS-test p-values ≥ 0.01 for all features → ✅ No significant distribution shifts

**Conclusion**: No evidence of data leakage


---

## Reproducibility

- Data source: `/Users/hwangsia/Desktop/open/train.parquet`
- Train/validation split: 80/20 stratified on `clicked`, with `random_state=42`
- All figures and tables were generated using `EDA.ipynb`
- Key outputs (e.g., `train_df.csv`, `valid_df.csv`) saved under `Desktop` for downstream modeling

Environment:
- Python 3.11+
- Libraries: pandas, numpy, matplotlib, seaborn, scipy, scikit-learn


