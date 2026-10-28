# AutoML Playground

Web app: upload a CSV, profile it, clean it contextually, detect the learning task,
suggest a target, train and tune models, show a dashboard, then let the user play
with the best model in a gamified playground.

## Stack
Python 3.11, pandas, scikit-learn, XGBoost, LightGBM, Optuna, MLflow, Plotly,
Streamlit, pytest, ruff, Docker. Deploy target: GCP Cloud Run, models in a GCS bucket.
v1 is Streamlit only. There's no FastAPI layer; add one later only if something needs an API.

## Architecture
```
 browser
    |  upload CSV
    v
 Streamlit (frontend/)        CSV kept in st.session_state only, never written to disk
    |  read_csv_raw -> DataFrame
    v
 app/core: profile -> clean (stateless fixes only, logged, reversible) -> detect_task
    |  clean df + target + task
    v
 app/ml: train/test split -> one sklearn Pipeline per model
         (ColumnTransformer: impute/scale/encode, fit on train only + model)
         -> CV + Optuna (~120 s total budget, configurable)
         -> held-out test score -> Leaderboard
    |                          \
    |                           -> MLflow: dataset metadata, models, params, scores per run
    v
 best Pipeline -> one .joblib -> artifacts/ locally, GCS bucket in prod
    |
    v
 Streamlit dashboard + game (loads the same .joblib to predict)
```

## Structure
- `app/core/`      profiling.py, cleaning.py, preprocessing.py, task_detection.py
- `app/ml/`        models.py, tuning.py, evaluation.py
- `frontend/`      Streamlit app and pages (the only entry point)
- `tests/`         pytest, mirrors app/
- `sample_data/`   iris.csv, breast_cancer.csv, diabetes.csv, messy.csv

## Pipeline contract
1. `profile(df) -> DatasetProfile` : column types (numeric, categorical, boolean, datetime, time, text, id, constant), missing %, cardinality, outliers. Constant means 1 distinct value and under 5% missing; 1 value plus more blanks is categorical (present vs missing is the signal). A numeric column with exactly 2 values that aren't 0/1 (like sex 1/2) is categorical. Timezone-aware datetimes become UTC, then tz-naive
2. `clean(df, profile, disabled=None, reference_date=None) -> (clean_df, CleaningLog)` : every step logged with column, action, reason, and only if it changed something (plain parses like "5.1" -> 5.1 are silent). User can override any step by passing its step_id in `disabled`. Dropped id columns go to `CleaningLog.row_ids` (same index as clean_df), never to the features. Fixed rules only: placeholder codes (999, -1...), impossible values by column name (negative age, pct > 100, zero bmi...) and invalid dates (future vs `reference_date`, before 1900, placeholder dates) become missing. A rule that would hit more than 10% of a column is skipped and logged as a flag. Rows are never dropped for outliers
   `build_preprocessor(profile, target, df, *, family="linear"|"tree", options=PreprocessOptions) -> ColumnTransformer` (app/core/preprocessing.py) : unfitted. Phase 4 calls it with `profile(X_train)` and `df=X_train` after the split, once per model family. Everything here is fitted on train only: outlier cap/log/set-to-missing (cap and log only for linear, SVM, KNN; trees get raw values), median vs mean (or KNN / iterative) imputing, `was_missing` flags for columns over 5% missing, "Missing" category, date/time features (parts, sin/cos, days between dates). The target never appears in the output features
   Phase 4 must drop rows with a missing target before the split (clean() rules also run on the target column)
3. `detect_task(df, profile) -> TaskSuggestion` : ranked target candidates with reasons, task type (classification / regression / clustering)
4. `train_and_tune(df, target, task) -> Leaderboard` : CV on train, Optuna with a total time budget (default about 120 seconds, configurable), final score on held-out test. Each candidate is ONE sklearn Pipeline (ColumnTransformer + model), so preprocessing and model always travel together. Every training run is tracked in MLflow (dataset metadata, model names, params, scores)
5. Best Pipeline saved as a single .joblib to `artifacts/` (GCS bucket in prod)

## Rules
- Reinforcement learning is NOT supported for tabular CSVs. Say so clearly in the UI, never fake it.
- Split train/test BEFORE any fitting (scalers, encoders, imputers). Fit preprocessing inside sklearn Pipelines only. No leakage.
- Every module has tests. Run `pytest` before saying a task is done.
- Every cleaning decision is logged and reversible.
- `clean()` runs before the split, so it only does stateless, row-local fixes (types, trimming, case merging, dedupe, dropping id/constant columns). Anything learned from the data (imputation, scaling, encoding, outlier bounds) goes in the Pipeline's ColumnTransformer.
- Uploaded CSVs live in memory only (`st.session_state`). Never write user data to disk, logs or MLflow artifacts. Log dataset metadata (name, shape, hash), not the rows. The saved .joblib is the one accepted exception: fitted encoders hold category values.
- Load CSVs with `read_csv_raw` (all strings) so profiling sees the raw text. Plain `pd.read_csv` turns "0800" into 800.
- Type hints everywhere. Small, pure functions. No notebooks in `app/`.
- Run `ruff check . --fix` and `ruff format .` before committing.
- Commit after each working feature with a clear message. Push to `main` only when tests pass.
- Writing style for UI text and docs: plain, casual English. No em dashes.

## Phases (tick off as done)
- [x] 1. Profiling
- [x] 2. Contextual cleaning + cleaning log
- [ ] 3. Task detection + target suggestion
- [ ] 4. Models + Optuna tuning + evaluation + MLflow tracking
- [ ] 5. Dashboard (correlations, leaderboard, feature importance, best model card)
- [ ] 6. Game (Beat the Model, What-If sliders)
- [ ] 7. Docker, README, GitHub Actions (ruff, pytest, Docker build, deploy to Cloud Run), GCS bucket for saved pipelines, demo video
