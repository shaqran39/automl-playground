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
from app.core.preprocessing import (  # noqa: E402
    NoUsableFeaturesError,
    PreprocessOptions,
    build_preprocessor,
)
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
    "outlier_choices",
    "numeric_imputer",
)
OUTLIER_LABELS: dict[str, str] = {
    "auto": "auto",
    "keep": "keep",
    "cap": "cap",
    "log": "log",
    "missing": "set to missing",
}
OUTLIER_VALUES = {label: value for value, label in OUTLIER_LABELS.items()}
IMPUTER_LABELS: dict[str, str] = {
    "auto": "Auto (median or mean, whichever fits the column)",
    "knn": "KNN (copy from the most similar rows)",
    "iterative": "Iterative (predict each blank from the other columns)",
}

GROUP_LABELS: dict[str, str] = {
    "numeric": "Numbers: handle outliers, put on the same scale, then fill blanks",
    "dates": "Dates and times: split into year, month, weekday, hour and gaps between dates",
    "was_missing": "Was it blank? Extra yes/no columns for columns with over 5% gaps",
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
    st.session_state["outlier_choices"] = {}
    st.session_state["numeric_imputer"] = "auto"


def disabled_steps() -> set[str]:
    return set(st.session_state.get("disabled_steps", set()))


def outlier_choices() -> dict[str, str]:
    return dict(st.session_state.get("outlier_choices", {}))


def preprocess_options(numeric_cols: list[str]) -> PreprocessOptions:
    """Current outlier and imputer picks, limited to columns that are still numeric."""
    choices = {c: v for c, v in outlier_choices().items() if c in numeric_cols}
    imputer = st.session_state.get("numeric_imputer", "auto")
    return PreprocessOptions(outliers=choices, numeric_imputer=imputer)


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


# ---------- outliers ----------


def numeric_columns(clean_prof: DatasetProfile) -> list[str]:
    return [c.name for c in clean_prof.columns if c.dtype == "numeric"]


def values_removed(log: CleaningLog, column: str) -> int:
    """Impossible or placeholder values the cleaning step turned into blanks."""
    wanted = {f"impossible:{column}", f"placeholder:{column}"}
    return sum(s.rows_affected for s in log.steps if s.step_id in wanted and s.enabled)


def outlier_frame(clean_prof: DatasetProfile, log: CleaningLog) -> pd.DataFrame:
    choices = outlier_choices()
    rows = [
        {
            "column": c.name,
            "outliers": c.n_outliers,
            "removed": values_removed(log, c.name),
            "choice": OUTLIER_LABELS.get(choices.get(c.name, "auto"), "auto"),
        }
        for c in clean_prof.columns
        if c.dtype == "numeric"
    ]
    return pd.DataFrame(rows, columns=["column", "outliers", "removed", "choice"])


def render_outlier_editor(clean_prof: DatasetProfile, log: CleaningLog) -> None:
    frame = outlier_frame(clean_prof, log)
    if frame.empty:
        st.info("No number columns here, so there's nothing to tweak.")
        return
    st.caption(
        "Auto leaves a column alone if it has no outliers, uses log for skewed positive "
        "numbers, and caps the rest. Cap squeezes extreme values to the 1st and 99th "
        "percentile. Log shrinks long tails. Set to missing blanks them out."
    )
    st.caption(
        "Cap and log only apply to linear, SVM and KNN models. "
        "Tree models get the raw values, since big numbers don't throw them off."
    )
    edited = st.data_editor(
        frame,
        key=f"outlier_editor|{st.session_state['dataset_key']}",
        hide_index=True,
        width="stretch",
        disabled=["column", "outliers", "removed"],
        column_config={
            "outliers": st.column_config.NumberColumn(
                help="Values far outside the middle half of the data (the IQR rule). "
                "Just a count, nothing is removed."
            ),
            "removed": st.column_config.NumberColumn(
                "impossible or placeholder",
                help="Values that can't be real (like age 150) or filler codes (like 999). "
                "Cleaning already turned these into blanks.",
            ),
            "choice": st.column_config.SelectboxColumn(
                "what to do",
                options=list(OUTLIER_LABELS.values()),
                required=True,
                help="How the model handles outliers. Learned from the training rows only.",
            ),
        },
    )
    new = {
        col: OUTLIER_VALUES[label]
        for col, label in zip(edited["column"], edited["choice"], strict=True)
        if OUTLIER_VALUES.get(label, "auto") != "auto"
    }
    numeric = set(frame["column"])
    old = {c: v for c, v in outlier_choices().items() if c in numeric and v != "auto"}
    if new != old:
        st.session_state["outlier_choices"] = new
        st.rerun()


def render_imputer_choice() -> None:
    st.radio(
        "How should we fill blanks in number columns?",
        list(IMPUTER_LABELS),
        format_func=IMPUTER_LABELS.get,
        key="numeric_imputer",
        help="Whatever you pick only learns from the training rows, so nothing leaks.",
    )


def render_outliers(clean_prof: DatasetProfile, log: CleaningLog) -> None:
    st.header("4. Outliers and blanks")
    st.write("Pick how the model should treat extreme numbers and gaps.")
    render_outlier_editor(clean_prof, log)
    render_imputer_choice()


# ---------- result ----------


def render_row_ids(log: CleaningLog) -> None:
    if log.row_ids is None:
        st.info("No ID column here, so each row is tracked by its original row number.")
        return
    st.write(f"Rows are tracked by **{log.row_ids.name}** (kept aside, not used for training).")
    st.dataframe(log.row_ids.head(20), width="stretch")


def render_result(raw_df: pd.DataFrame, clean_df: pd.DataFrame, log: CleaningLog) -> None:
    st.header("5. The cleaned data")
    st.write(
        f"{raw_df.shape[0]:,} rows x {raw_df.shape[1]} columns "
        f"-> {clean_df.shape[0]:,} rows x {clean_df.shape[1]} columns"
    )
    st.dataframe(clean_df.head(20), width="stretch")
    render_row_ids(log)


# ---------- preprocessor preview ----------


def preprocessor_groups(
    clean_df: pd.DataFrame, clean_prof: DatasetProfile
) -> list[tuple[str, list[str]]]:
    """(group name, columns) for each part of the unfitted ColumnTransformer."""
    options = preprocess_options(numeric_columns(clean_prof))
    ct = build_preprocessor(clean_prof, target=None, df=clean_df, options=options)
    return [(name, list(cols)) for name, _, cols in ct.transformers]


def render_model_view(clean_df: pd.DataFrame, clean_prof: DatasetProfile) -> None:
    with st.expander("What the model will see"):
        st.caption(
            "This is the plan, not fitted yet. It only learns from the training split "
            "in a later step, so nothing leaks from the test rows."
        )
        try:
            groups = preprocessor_groups(clean_df, clean_prof)
        except NoUsableFeaturesError:
            st.warning("No usable columns left for a model. Try turning some fixes back on.")
            return
        except ValueError as e:
            st.error(str(e))
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
    clean_prof = profile(clean_df)
    render_outliers(clean_prof, log)
    render_result(raw_df, clean_df, log)
    render_model_view(clean_df, clean_prof)


main()
