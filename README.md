# AutoML Playground

Upload a CSV and the app does the rest: profiles your data, cleans it with a decision log you can override, works out whether it's a classification, regression or clustering problem, suggests a target, trains and tunes a set of models, and shows the results on a dashboard. Then you can play with the winning model in a game.

> Status: early build. See the phase checklist in `CLAUDE.md`.

## Features (planned)
- Smart profiling: catches disguised types like `$1,200` and dates stored as text
- Contextual cleaning with a reversible cleaning log
- Task detection and ranked target suggestions with reasons
- Model leaderboard with Optuna hyperparameter tuning
- Dashboard: correlations, feature importance, best model card
- Playground: "Beat the Model" and "What-If" sliders

## Stack
Python, pandas, scikit-learn, XGBoost, LightGBM, Optuna, FastAPI, Streamlit, Plotly, Docker, GCP Cloud Run

## Run locally
```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
pytest
streamlit run frontend/streamlit_app.py
```

## Sample data
`sample_data/` has iris (classification), breast_cancer (binary), diabetes (regression) and messy.csv (deliberately ugly churn data for stress testing).

## Built with
Built with Claude Code using project subagents and slash commands in `.claude/`.
