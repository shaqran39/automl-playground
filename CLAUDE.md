# AutoML Playground

Web app: upload a CSV, profile it, clean it contextually, detect the learning task,
suggest a target, train and tune models, show a dashboard, then let the user play
with the best model in a gamified playground.

## Stack
Python 3.11, FastAPI, pandas, scikit-learn, XGBoost, LightGBM, Optuna, Plotly,
Streamlit (frontend), pytest, ruff, Docker. Deploy target: GCP Cloud Run.

## Structure
- `app/core/`      profiling.py, cleaning.py, task_detection.py
- `app/ml/`        models.py, tuning.py, evaluation.py
- `app/api/`       FastAPI routes (main.py)
- `frontend/`      Streamlit app and pages
- `tests/`         pytest, mirrors app/
- `sample_data/`   iris.csv, breast_cancer.csv, diabetes.csv, messy.csv

## Pipeline contract
1. `profile(df) -> DatasetProfile` : column types (numeric, categorical, boolean, datetime, text, id, constant), missing %, cardinality, outliers. Constant means 1 distinct value and under 5% missing; 1 value plus more blanks is categorical (present vs missing is the signal)
2. `clean(df, profile) -> (clean_df, CleaningLog)` : every step logged with column, action, reason. User can override any step
3. `detect_task(df, profile) -> TaskSuggestion` : ranked target candidates with reasons, task type (classification / regression / clustering)
4. `train_and_tune(df, target, task) -> Leaderboard` : CV on train, Optuna with a time budget, final score on held-out test
5. Best pipeline (preprocessing + model) saved with joblib to `artifacts/`

## Rules
- Reinforcement learning is NOT supported for tabular CSVs. Say so clearly in the UI, never fake it.
- Split train/test BEFORE any fitting (scalers, encoders, imputers). Fit preprocessing inside sklearn Pipelines only. No leakage.
- Every module has tests. Run `pytest` before saying a task is done.
- Every cleaning decision is logged and reversible.
- Type hints everywhere. Small, pure functions. No notebooks in `app/`.
- Run `ruff check . --fix` and `ruff format .` before committing.
- Commit after each working feature with a clear message. Push to `main` only when tests pass.
- Writing style for UI text and docs: plain, casual English. No em dashes.

## Phases (tick off as done)
- [x] 1. Profiling
- [ ] 2. Contextual cleaning + cleaning log
- [ ] 3. Task detection + target suggestion
- [ ] 4. Models + Optuna tuning + evaluation
- [ ] 5. Dashboard (correlations, leaderboard, feature importance, best model card)
- [ ] 6. Game (Beat the Model, What-If sliders)
- [ ] 7. Docker, README, deploy, demo video
