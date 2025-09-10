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

### Column Groups *(119 columns incl. target)*

| Group                         | Description                            | Notes                           |
|------------------------------|----------------------------------------|---------------------------------|
| `gender`, `age_group`        | User demographics                      | Categorical                     |
| `inventory_id`               | Ad inventory ID                        | Categorical                     |
| `day_of_week`, `hour`        | Time context (weekday index, hour)     | Ordinal / categorical           |
| `seq`                        | User server log **sequence**           | Exact meaning not disclosed     |
| `l_feat_*`                   | Ad-level attributes                    | Cardinality-based type handling |
| `feat_a_* ~ feat_e_*`        | Feature groups A to E                  | Mostly continuous / int-like    |
| `history_a_*`                | Historical popularity features         | Long-term engagement signal     |
| `history_b_*`                | Other historical features              | High inter-correlation observed |
| `clicked`                    | Target (train only)                    | Binary label                    |
| `ID`                         | Row identifier (test only)             | Not used for training           |

> Note: Exact feature semantics are intentionally undisclosed per competition guidelines.



