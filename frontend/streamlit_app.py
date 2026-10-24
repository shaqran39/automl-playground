"""Streamlit frontend. Run: streamlit run frontend/streamlit_app.py"""

import hashlib
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:  # streamlit only adds frontend/ to the path
    sys.path.insert(0, str(ROOT))

from app.core.cleaning import CleaningLog, clean  # noqa: E402
from app.core.preprocessing import build_preprocessor  # noqa: E402
from app.core.profiling import DatasetProfile, profile, read_csv_raw  # noqa: E402

SAMPLE_DIR = ROOT / "sample_data"
NO_SAMPLE = "(none)"
EDITOR_COLUMNS = ["on", "column", "action", "reason", "rows_affected"]
DATASET_KEYS = (
    "dataset_key",
    "dataset_name",
    "raw_df",
    "raw_profile",
    "disabled_steps",
    "editor_version",
    "clean_df",
    "cleaning_log",
)

GROUP_LABELS: dict[str, str] = {
    "numeric": "Numbers: fill blanks (median or mean), then put on the same scale",
    "missing_flags": "Was it blank? Extra yes/no columns for numbers that had gaps",
    "boolean": "Yes/no columns: fill blanks with the most common answer",
    "onehot": "Categories with few options: one 0/1 column per option",
    "frequency": "Categories with lots of options: replaced by how common each value is",
}


# ---------- state ----------


def dataset_key(name: str, df: pd.DataFrame) -> str:
    """Stable key for a dataset: name, shape and a hash of the contents."""
    digest = hashlib.sha1(pd.util.hash_pandas_object(df, index=True).values.tobytes())
    return f"{name}|{df.shape[0]}x{df.shape[1]}|{digest.hexdigest()[:12]}"


def set_dataset(name: str, df: pd.DataFrame) -> None:
    """Store the raw df in session state; reset cleaning choices when it changes."""
    key = dataset_key(name, df)
    if st.session_state.get("dataset_key") == key:
        return
    st.session_state["dataset_key"] = key
    st.session_state["dataset_name"] = name
    st.session_state["raw_df"] = df
    st.session_state["raw_profile"] = profile(df)
    st.session_state["disabled_steps"] = set()
    st.session_state["editor_version"] = 0


def disabled_steps() -> set[str]:
    return set(st.session_state.get("disabled_steps", set()))


# ---------- data source ----------


def sample_files() -> list[str]:
    return sorted(p.name for p in SAMPLE_DIR.glob("*.csv"))


def render_data_source() -> None:
    st.header("1. Pick some data")
    left, right = st.columns(2)
    with left:
        uploaded = st.file_uploader("Upload a CSV", type="csv")
    with right:
        sample = st.selectbox("Or try a sample", [NO_SAMPLE, *sample_files()])
    if uploaded is not None:
        set_dataset(uploaded.name, read_csv_raw(uploaded))  # memory only, never saved
    elif sample != NO_SAMPLE:
        set_dataset(sample, read_csv_raw(SAMPLE_DIR / sample))
    else:
        clear_dataset()  # user removed the file, so drop it from memory too


def clear_dataset() -> None:
    for key in DATASET_KEYS:
        st.session_state.pop(key, None)


# ---------- profile ----------


def profile_frame(prof: DatasetProfile) -> pd.DataFrame:
    rows = [
        {
            "column": c.name,
            "type": c.dtype,
            "missing %": round(c.missing_pct, 1),
            "unique": c.n_unique,
            "why": " ".join(c.reasons),
        }
        for c in prof.columns
    ]
    return pd.DataFrame(rows)


def render_profile(prof: DatasetProfile) -> None:
    st.header("2. What's in it")
    st.write(
        f"**{prof.n_rows:,}** rows, **{prof.n_cols}** columns, "
        f"**{prof.n_duplicate_rows:,}** duplicate rows"
    )
    st.dataframe(
        profile_frame(prof),
        hide_index=True,
        width="stretch",
        column_config={
            "missing %": st.column_config.NumberColumn(
                help="Share of blank or null-looking values", format="%.1f"
            ),
            "unique": st.column_config.NumberColumn(help="How many different values"),
            "why": st.column_config.TextColumn(help="Why we picked this type", width="large"),
        },
    )


# ---------- cleaning log ----------


def fix_frame(log: CleaningLog) -> pd.DataFrame:
    frame = log.to_frame()
    fixes = frame[frame["kind"] == "fix"].set_index("step_id")
    return fixes.rename(columns={"enabled": "on"})[EDITOR_COLUMNS]


def render_flags(log: CleaningLog) -> None:
    flags = [s for s in log.steps if s.kind == "flag"]
    if not flags:
        return
    st.subheader("Heads up")
    for step in flags:
        st.warning(f"**{step.column}**: {step.reason}")


def render_fix_editor(log: CleaningLog) -> None:
    fixes = fix_frame(log)
    if fixes.empty:
        st.success("Nothing to fix. This data is already pretty tidy.")
        return
    st.caption("Untick a fix to skip it. Everything updates right away.")
    key = f"fix_editor|{st.session_state['dataset_key']}|{st.session_state['editor_version']}"
    edited = st.data_editor(
        fixes,
        key=key,
        hide_index=True,
        width="stretch",
        disabled=[c for c in EDITOR_COLUMNS if c != "on"],
        column_config={
            "on": st.column_config.CheckboxColumn(help="Apply this fix?"),
            "column": st.column_config.TextColumn(help="* means the whole table"),
            "reason": st.column_config.TextColumn(width="large"),
            "rows_affected": st.column_config.NumberColumn(
                "rows changed", help="How many rows or values this fix touches"
            ),
        },
    )
    new_disabled = set(edited.index[~edited["on"].astype(bool)])
    if new_disabled != disabled_steps():
        st.session_state["disabled_steps"] = new_disabled
        st.rerun()


def reset_fixes() -> None:
    st.session_state["disabled_steps"] = set()
    st.session_state["editor_version"] += 1  # fresh editor, old ticks forgotten


def render_cleaning(log: CleaningLog) -> None:
    st.header("3. Cleaning")
    st.write("Here's what we'd fix. Every step is optional.")
    render_fix_editor(log)
    if disabled_steps():
        st.button("Turn everything back on", on_click=reset_fixes)
    render_flags(log)


# ---------- result ----------


def render_row_ids(log: CleaningLog) -> None:
    if log.row_ids is None:
        st.info("No ID column here, so each row is tracked by its original row number.")
        return
    st.write(f"Rows are tracked by **{log.row_ids.name}** (kept aside, not used for training).")
    st.dataframe(log.row_ids.head(20), width="stretch")


def render_result(raw_df: pd.DataFrame, clean_df: pd.DataFrame, log: CleaningLog) -> None:
    st.header("4. The cleaned data")
    st.write(
        f"{raw_df.shape[0]:,} rows x {raw_df.shape[1]} columns "
        f"-> {clean_df.shape[0]:,} rows x {clean_df.shape[1]} columns"
    )
    st.dataframe(clean_df.head(20), width="stretch")
    render_row_ids(log)


# ---------- preprocessor preview ----------


def preprocessor_groups(clean_df: pd.DataFrame) -> list[tuple[str, list[str]]]:
    """(group name, columns) for each part of the unfitted ColumnTransformer."""
    ct = build_preprocessor(profile(clean_df), target=None, df=clean_df)
    return [(name, list(cols)) for name, _, cols in ct.transformers]


def render_model_view(clean_df: pd.DataFrame) -> None:
    with st.expander("What the model will see"):
        st.caption(
            "This is the plan, not fitted yet. It only learns from the training split "
            "in a later step, so nothing leaks from the test rows."
        )
        try:
            groups = preprocessor_groups(clean_df)
        except ValueError:
            st.warning("No usable columns left for a model. Try turning some fixes back on.")
            return
        for name, cols in groups:
            st.markdown(f"**{GROUP_LABELS.get(name, name)}**")
            st.write(", ".join(cols))


# ---------- main ----------


def main() -> None:
    st.set_page_config(page_title="AutoML Playground", layout="wide")
    st.title("AutoML Playground")
    st.caption("Drop in a CSV. We'll figure out the rest.")
    render_data_source()
    if "raw_df" not in st.session_state:
        st.info("Upload a CSV or pick a sample to get started.")
        return
    raw_df: pd.DataFrame = st.session_state["raw_df"]
    prof: DatasetProfile = st.session_state["raw_profile"]
    render_profile(prof)
    clean_df, log = clean(raw_df, prof, disabled_steps())
    st.session_state["clean_df"] = clean_df
    st.session_state["cleaning_log"] = log
    render_cleaning(log)
    render_result(raw_df, clean_df, log)
    render_model_view(clean_df)


main()
