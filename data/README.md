# Data Overview

This directory is for storing raw datasets used in training and evaluation.
Due to competition compliance, the data is not included in this repository.

## Access Policy

- All dataset files must be stored locally under a user-defined path and referenced via the `DATA_DIR` environment variable.
- Datasets must not be shared or uploaded publicly under any circumstances.

## Dataset Structure

### train.parquet

- 10,704,179 samples
- 119 columns (including target: `clicked`)

### test.parquet

- 1,527,298 samples
- 118 columns (no target; includes `ID`)

### Column Groups

| Group          | Description |
|----------------|-------------|
| `gender`, `age_group` | User demographic |
| `inventory_id`        | Ad inventory ID |
| `day_of_week`, `hour`, `seq` | Session info |
| `l_feat_*`            | Ad-level features (e.g. `l_feat_14` = ad set) |
| `feat_a_* ~ feat_e_*` | Feature groups A to E |
| `history_a_*`         | Historical popularity signals |
| `clicked`             | Target (train only) |
| `ID`                  | Identifier (test only) |

> Note: Exact feature semantics are intentionally undisclosed per competition guidelines.


