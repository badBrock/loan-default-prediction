# 📋 Project Guide — Loan Default Prediction

> **Quick reference:** what lives where, and the exact order to run things.
> Run notebooks **1 → 2 → 3 → 4** in order. Notebook 2 is the critical dependency for everything downstream.

---

## 🗂️ 1. File Map — What Is Where

### Raw Data (inputs)
| File | Location | Description |
|------|----------|-------------|
| `train_flag.csv` | `data/train/` | uid, NAME_CONTRACT_TYPE, **TARGET** (labels) |
| `accounts_data_train.json` | `data/train/` | Nested account history per applicant |
| `enquiry_data_train.json` | `data/train/` | Nested credit-enquiry history per applicant |
| `test_flag.csv` | `data/test/` | uid, NAME_CONTRACT_TYPE (no TARGET) |
| `accounts_data_test.json` | `data/test/` | Test account history |
| `enquiry_data_test.json` | `data/test/` | Test enquiry history |

### Notebooks (run in order)
| # | Notebook | Reads | Writes |
|---|----------|-------|--------|
| 1 | `01_eda.ipynb` | raw files | *(nothing — analysis only)* |
| 2 | `02_feature_engineering.ipynb` | raw files | `features_train.csv`, `features_test.csv` |
| 3 | `03_baseline_modelling.ipynb` | `features_*.csv` | *(nothing — prints CV scores)* |
| 4 | `04_tuning_and_submission.ipynb` | `features_*.csv` | `final_submission_Sakshi_Vedi.csv` |

### Generated Artifacts (outputs)
| File | Created by | Purpose |
|------|-----------|---------|
| `features_train.csv` | Notebook 2 | Engineered train features (1 row/applicant + TARGET) |
| `features_test.csv` | Notebook 2 | Engineered test features (1 row/applicant) |
| `final_submission_Sakshi_Vedi.csv` | Notebook 4 | Final predictions (uid, TARGET) |

---

## ▶️ 2. Execution Order

```
┌─────────────────────────────────────────────────────────────┐
│  RAW DATA                                                     │
│  data/train/*.json + *.csv                                   │
│  data/test/*.json  + *.csv                                   │
└───────────────┬─────────────────────────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────────────────────────┐
│  STEP 1  →  01_eda.ipynb                                      │
│  Purpose : Understand data, quality issues, key signals      │
│  Output  : none (read-only analysis)                         │
│  Runtime : ~1 min                                            │
└───────────────┬─────────────────────────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────────────────────────┐
│  STEP 2  →  02_feature_engineering.ipynb                     │
│  Purpose : Build applicant-level features                    │
│  Output  : features_train.csv, features_test.csv  ★REQUIRED★ │
│  Runtime : ~3-5 min                                          │
└───────────────┬─────────────────────────────────────────────┘
                │
        ┌───────┴───────┐
        ▼               ▼
┌───────────────┐ ┌─────────────────────────────────────────────┐
│  STEP 3       │ │  STEP 4  →  04_tuning_and_submission.ipynb   │
│  03_baseline  │ │  Purpose : Optuna tuning + final submission  │
│  _modelling   │ │  Output  : final_submission_Sakshi_Vedi.csv  │
│               │ │  Runtime : ~30-40 min (40 Optuna trials)     │
│  Optional /   │ │                                              │
│  exploratory  │ │  ★ THIS PRODUCES THE DELIVERABLE ★           │
│  Runtime ~5m  │ │                                              │
└───────────────┘ └─────────────────────────────────────────────┘
```

### ⚠️ Dependency Rules
- **Notebook 2 MUST run before 3 and 4** — they need `features_*.csv`.
- **Notebook 1 is independent** — run anytime (read-only).
- **Notebooks 3 & 4 both read the same CSVs** — either can run after 2.
- **Notebook 3 is optional** — it's exploratory; only Notebook 4 makes the submission.

### Minimum path to submission
```
02_feature_engineering.ipynb  →  04_tuning_and_submission.ipynb
```

---

## ⚙️ 3. Before You Run — Path Setup

Each notebook has a **PATHS** block near the top. Update these to match your machine:

```python
# In Notebooks 1 & 2
TRAIN_FLAG = r'E:\senior_ds_test\senior_ds_test\data\train\train_flag.csv'
TRAIN_ACC  = r'E:\senior_ds_test\senior_ds_test\data\train\accounts_data_train.json'
TRAIN_ENQ  = r'E:\senior_ds_test\senior_ds_test\data\train\enquiry_data_train.json'
TEST_FLAG  = r'E:\senior_ds_test\senior_ds_test\data\test\test_flag.csv'
TEST_ACC   = r'E:\senior_ds_test\senior_ds_test\data\test\accounts_data_test.json'
TEST_ENQ   = r'E:\senior_ds_test\senior_ds_test\data\test\enquiry_data_test.json'

# In Notebook 4
SUBMISSION_DIR = r'E:\senior_ds_test\senior_ds_test\final_submission'
```

> 💡 `features_train.csv` / `features_test.csv` are written to the **current working directory** (where Jupyter launched). Notebooks 3 & 4 read from the same place. **Keep all 4 notebooks in the same folder.**

---

## 🔑 4. Key Constants (must stay consistent)

| Constant | Value | Where used | Why |
|----------|-------|-----------|-----|
| `REF_DATE` | `2021-01-01` | Notebook 2 | Snapshot date (max across all dates); all recency features anchor to this |
| `KEEP_CREDIT_TYPES` | Consumer credit, Credit card, Car loan, Mortgage, Microloan | Notebook 2 | Rare credit types → grouped as "Other" |
| `KEEP_ENQ_TYPES` | Cash loans, Revolving loans | Notebook 2 | Rare enquiry types → grouped as "Other" |
| `scale_pos_weight` | ≈ 11.4 (auto: neg/pos) | Notebooks 3 & 4 | Handles ~8% class imbalance |
| CV | `StratifiedKFold(5, shuffle=True, random_state=42)` | Notebooks 3 & 4 | Reproducible evaluation |
| `random_state` | `42` | everywhere | Reproducibility |

---

## 📊 5. Expected Results (sanity checkpoints)

Use these to confirm each step ran correctly.

### After Notebook 1 (EDA)
```
Class balance       : ~8.06% default (TARGET=1)
has accounts        : ~0.857
has enquiries       : 1.0
Ref date (max)      : 2021-01-01
Payment string      : 100% length divisible by 3
Invalid closed<open : 13 rows
```

### After Notebook 2 (Features)
```
Train shape      : (261383, ~72)
Test shape       : (46127, ~71)
Feature count    : ~69
Any NaN?         : False (both train & test)
Assertions pass  : TARGET not in features, uids unique
```

### After Notebook 3 (Baseline)
```
XGBoost    CV AUC : ~0.677
LightGBM   CV AUC : ~0.678
CatBoost   CV AUC : ~0.678
Rank-Blend CV AUC : ~0.679
Model corr        : ~0.98 (highly correlated)
```

### After Notebook 4 (Tuning + Submission)
```
Optuna best AUC  : ~0.679
Best spw         : 1.0 (often beats 11.4)
Features kept    : ~48 (after dropping bottom 30%)
Final CV AUC     : ~0.680
Submission shape : (46127, 2)
Columns          : uid, TARGET
TARGET range     : [0, 1], no NaN
```

---

## 🧩 6. Feature Groups Reference

Features created in Notebook 2, by prefix:

| Prefix | Source | Count | Examples |
|--------|--------|------:|----------|
| `acc_` | Accounts | ~30 | acc_n_accounts, acc_open_ratio, acc_overdue_ratio, acc_cnt_Microloan |
| `pmt_` | Payment history | ~17 | pmt_max_dpd, pmt_n_90, pmt_ever_90plus, pmt_recent_dpd_max |
| `enq_` | Enquiries | ~18 | enq_n, enq_n_90d, enq_days_since_last, enq_span_days |
| *(flag)* | Contract type | 2 | is_cash_loan, has_accounts |

**Top predictive features** (from importance): `acc_days_since_open_mean`,
`enq_days_since_first`, `is_cash_loan`, `acc_n_open`, `enq_n_90d`,
`acc_cnt_Microloan_share`, `acc_open_ratio`.

---

## 🛠️ 7. Environment / Dependencies

```bash
pip install pandas numpy scikit-learn xgboost lightgbm catboost optuna scipy jupyter
```

| Library | Used in | Purpose |
|---------|---------|---------|
| pandas, numpy | all | Data handling |
| scikit-learn | 3, 4 | CV, metrics |
| xgboost | 3, 4 | Primary model |
| lightgbm, catboost | 3, 4 | Ensemble members |
| optuna | 4 | Hyperparameter search |
| scipy | 3, 4 | rankdata (rank-blend) |

---

## ✅ 8. Run Checklist

- [ ] Install dependencies (Section 7)
- [ ] Update PATHS in Notebooks 1, 2, and 4 (Section 3)
- [ ] Place all 4 notebooks in the same folder
- [ ] Run **01_eda** → verify checkpoints (Section 5)
- [ ] Run **02_feature_engineering** → confirm `features_*.csv` created
- [ ] *(optional)* Run **03_baseline_modelling** → check baseline AUC
- [ ] Run **04_tuning_and_submission** → confirm submission CSV created
- [ ] Verify submission: correct shape, no NaN, TARGET in [0,1]

---

## 🚀 9. Optional — Non-Interactive Full Run

To run all notebooks end-to-end from the command line (no manual clicking):

```bash
# Execute in order, saving outputs back into each notebook
jupyter nbconvert --to notebook --execute --inplace 01_eda.ipynb
jupyter nbconvert --to notebook --execute --inplace 02_feature_engineering.ipynb
jupyter nbconvert --to notebook --execute --inplace 03_baseline_modelling.ipynb
jupyter nbconvert --to notebook --execute --inplace 04_tuning_and_submission.ipynb
```

> ⏱️ Note: Notebook 4 runs 40 Optuna trials (~30-40 min). For a quick smoke test,
> temporarily lower `n_trials` in the Optuna cell.

One-liner (bash) to chain them and stop on first error:

```bash
for nb in 01_eda 02_feature_engineering 03_baseline_modelling 04_tuning_and_submission; do
  echo "Running $nb..."
  jupyter nbconvert --to notebook --execute --inplace "$nb.ipynb" || { echo "FAILED: $nb"; break; }
done
```

---

## 📝 10. Troubleshooting

| Problem | Cause | Fix |
|---------|-------|-----|
| `FileNotFoundError` on raw files | Wrong PATHS | Update paths in Section 3 |
| `FileNotFoundError: features_train.csv` | Ran NB 3/4 before NB 2 | Run Notebook 2 first |
| Features CSV not found by NB 3/4 | Different working dir | Keep notebooks + CSVs in same folder |
| CatBoost clone error | `scale_pos_weight` not float | Already handled: `spw = float(...)` |
| Optuna very slow | 40 trials × 5-fold CV | Reduce `n_trials` for quick tests |
| tqdm/ipywidgets warning | Missing widget extension | Harmless; or `pip install ipywidgets` |
| Windows path error (`\t`, `\n`) | Missing raw-string prefix | Use `r'...'` for all Windows paths |

---

## 📌 11. Deliverable

**Final output:** `final_submission_Sakshi_Vedi.csv`
**Location:** `SUBMISSION_DIR` (set in Notebook 4)
**Format:**
```csv
uid,TARGET
CMO22835242,0.041475
MRJ34316727,0.167568
...
```
**Metric:** ROC-AUC (~0.680 CV)

---

## 🧭 12. At-a-Glance Summary

| Question | Answer |
|----------|--------|
| Where's the data? | `data/train/` and `data/test/` |
| What do I run first? | `01_eda.ipynb` (or skip to `02` if you trust EDA) |
| What's the critical step? | `02_feature_engineering.ipynb` — everything needs its CSVs |
| Can I skip Notebook 3? | Yes — it's exploratory only |
| What produces the submission? | `04_tuning_and_submission.ipynb` |
| Fastest path to result? | `02` → `04` |
| Final metric? | ROC-AUC ≈ 0.680 |
| Final file? | `final_submission_Sakshi_Vedi.csv` |

---

*Last updated: run notebooks 1→2→3→4 in order. Notebook 2 is the critical dependency for everything downstream.*