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
- `l_feat_20`, `l_feat_23`: Only one category present → no predictive value
- `l_feat_2`, `l_feat_13`: Near-identical CTRs across categories → low information
- `l_feat_17`: Perfectly correlated with `l_feat_9` → removed due to duplication
- `l_feat_26`: Low cardinality + weak CTR signal → removed due to low information

#### 3.2 Summary of feat_* and history_* Groups

- `feat_e_3`: High missing rate
- `feat_c_1`, `feat_c_4`, `feat_c_5`, `feat_c_7`: Low variance → likely uninformative
- `feat_a_*`: Many features are 0 for over 75% of samples → binarization considered
- `history_b_*`: Many features are highly correlated with each other → redundancy detected

---

## Multicollinearity Removal (Correlation Test)

**Drop Criteria**
- **Pearson correlation**: |r| ≥ 0.995 AND p-value < 0.01
- **Cramér’s V**: V ≈ 1.0 AND p-value < 0.01

**Dropped Features Summary**

| Criterion             | Dropped Features                                    | Notes                     |
|-----------------------|-----------------------------------------------------|---------------------------|
| Cramér’s V ≈ 1.0      | `l_feat_3`, `l_feat_16`                              | Perfect duplicates        |
| Pearson `|r| ≥ 0.995`    | `history_b_3` to `history_b_30` (total 26 features) | Severe multicollinearity  |

**Conclusion**
- Strong multicollinearity was concentrated in `history_b_*` features → removed accordingly
- Current feature set is well-suited for tree-based models

---

## Data Leakage Check

- **Categorical features**: No category mismatch between train and valid sets → ✅ No leakage
- **Continuous features**: KS-test p-values ≥ 0.01 for all features → ✅ No significant distribution shifts

**Conclusion**: No evidence of data leakage

