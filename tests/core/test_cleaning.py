"""Phase 2 cleaning tests (written test-first against the agreed contract).

Hand-built frames are always all-string, like read_csv_raw gives, and get
profiled with the real profiler, so these tests also pin down which profile
types trigger which cleaning step.
"""

import copy
import io
from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest

from app.core import cleaning as C
from app.core import profiling as P
from tests.conftest import DATA, MESSY

STEP_ORDER = [
    "dedupe",
    "normalize",
    "to_numeric",
    "to_bool",
    "merge_case",
    "presence_flag",
    "date_parts",
    "drop_id",
    "drop_constant",
    "high_missing",
]
LOG_COLUMNS = ["step_id", "column", "action", "reason", "kind", "enabled", "rows_affected"]

# every step that applies to the kitchen_sink_raw fixture
SINK_FIX_STEPS = [
    "dedupe:*",
    "normalize:state",
    "normalize:age",
    "normalize:notes",
    "normalize:sparse",
    "to_numeric:monthly_spend",
    "to_numeric:age",
    "to_numeric:sparse",
    "to_bool:churned",
    "merge_case:state",
    "presence_flag:notes",
    "date_parts:signup_date",
    "drop_id:customer_id",
    "drop_constant:country",
]
SINK_FLAG_STEPS = ["high_missing:sparse"]


# ---------------------------------------------------------------- helpers


def raw(cols: dict[str, list[str]], index: list[int] | None = None) -> pd.DataFrame:
    """All-string frame, the same shape read_csv_raw gives (blanks are "")."""
    return pd.DataFrame({k: pd.Series(v, dtype=str, index=index) for k, v in cols.items()})


def with_row(cols: dict[str, list[str]]) -> pd.DataFrame:
    """raw() plus a unique float column "r", so no row is an exact duplicate."""
    n = len(next(iter(cols.values())))
    return raw({**cols, "r": [f"{i}.5" for i in range(n)]})


def run(
    df: pd.DataFrame, disabled: set[str] | None = None
) -> tuple[pd.DataFrame, C.CleaningLog, P.DatasetProfile]:
    prof = P.profile(df)
    clean_df, log = C.clean(df, prof, disabled)
    return clean_df, log, prof


def ids(log: C.CleaningLog) -> list[str]:
    return [s.step_id for s in log.steps]


def expect_type(prof: P.DatasetProfile, name: str, dtype: str) -> None:
    """Guard: the test relies on the profiler typing this column a certain way."""
    assert prof.column(name).dtype == dtype, f"profiler typed {name!r} unexpectedly"


def floats(series: pd.Series) -> list[float | None]:
    return [None if pd.isna(v) else float(v) for v in series]


def is_float64(series: pd.Series) -> bool:
    return series.dtype == np.float64


def log_signature(log: C.CleaningLog) -> list[tuple[object, ...]]:
    """Everything about the log except `enabled`."""
    return [(s.step_id, s.column, s.action, s.reason, s.kind, s.rows_affected) for s in log.steps]


def assert_log_invariants(log: C.CleaningLog) -> None:
    for s in log.steps:
        assert s.step_id == f"{s.action}:{s.column}"
        assert s.action in STEP_ORDER, s.action
        assert s.kind in ("fix", "flag")
        assert isinstance(s.reason, str) and s.reason.strip(), s.step_id
        assert "—" not in s.reason, f"em dash in reason for {s.step_id}"
        assert s.rows_affected >= 0
        if s.action == "dedupe":
            assert s.column == "*"
        if s.action == "high_missing":
            assert s.kind == "flag"
        else:
            assert s.kind == "fix"


# ---------------------------------------------------------------- dedupe


def test_dedupe_drops_later_duplicates_and_keeps_index() -> None:
    df = raw({"a": ["1", "2", "1", "3"], "b": ["x", "y", "x", "z"]}, index=[10, 20, 30, 40])
    clean_df, log, prof = run(df)
    assert prof.n_duplicate_rows == 1
    step = log.step("dedupe:*")
    assert step.column == "*"
    assert step.rows_affected == 1
    assert list(clean_df.index) == [10, 20, 40]


def test_dedupe_only_on_exact_raw_duplicates() -> None:
    # " x" and "x" only match after trimming, so they're not exact duplicates
    df = raw({"a": ["1", "1", "2"], "b": ["x", " x", "y"]})
    clean_df, log, prof = run(df)
    assert prof.n_duplicate_rows == 0
    assert "dedupe:*" not in ids(log)
    assert len(clean_df) == 3


def test_no_dedupe_step_without_duplicates() -> None:
    df = raw({"a": ["1", "2", "3"], "b": ["x", "y", "z"]})
    _, log, _ = run(df)
    assert all(s.action != "dedupe" for s in log.steps)


# ---------------------------------------------------------------- normalize


def test_normalize_trims_and_turns_null_tokens_into_nan() -> None:
    df = raw({"city": [" Sydney", "Perth ", "N/A", " null ", "Perth", "Sydney"]})
    clean_df, log, prof = run(df)
    expect_type(prof, "city", "categorical")
    step = log.step("normalize:city")
    assert step.rows_affected == 4
    out = clean_df["city"]
    assert out.iloc[0] == "Sydney"
    assert out.iloc[1] == "Perth"
    assert pd.isna(out.iloc[2]) and pd.isna(out.iloc[3])
    assert out.iloc[4] == "Perth"


@pytest.mark.parametrize("token", ["NULL", " none ", "?", "-", "NaN", "na", "n/a"])
def test_normalize_null_tokens_case_insensitive(token: str) -> None:
    df = raw({"c": ["a", "b", token, "a", "b"]})
    clean_df, log, _ = run(df)
    assert "normalize:c" in ids(log)
    assert pd.isna(clean_df["c"].iloc[2])


def test_normalize_not_logged_when_nothing_changes() -> None:
    df = with_row({"c": ["a", "b", "a", "c"]})  # extra unique column so dedupe drops nothing
    clean_df, log, _ = run(df)
    assert "normalize:c" not in ids(log)
    assert clean_df["c"].tolist() == ["a", "b", "a", "c"]


def test_normalize_skips_non_string_columns() -> None:
    df = pd.DataFrame({"n": [1.0, np.nan, 3.0, 4.0], "c": ["a", "b", "a", "b"]})
    _, log, _ = run(df)
    assert "normalize:n" not in ids(log)


# ---------------------------------------------------------------- to_numeric


def test_to_numeric_strips_disguises() -> None:
    df = raw({"m": ["$1,200", "(5)", "12%", "7"]})
    clean_df, log, prof = run(df)
    expect_type(prof, "m", "numeric")
    assert "to_numeric:m" in ids(log)
    assert is_float64(clean_df["m"])
    assert floats(clean_df["m"]) == [1200.0, -5.0, 12.0, 7.0]


def test_to_numeric_leftover_values_become_missing() -> None:
    values = [str(i + 0.5) for i in range(19)] + ["oops"]
    df = raw({"m": values})
    clean_df, log, prof = run(df)
    expect_type(prof, "m", "numeric")
    step = log.step("to_numeric:m")
    assert step.reason.strip()
    assert is_float64(clean_df["m"])
    assert pd.isna(clean_df["m"].iloc[19])
    assert clean_df["m"].iloc[:19].notna().all()


def test_to_numeric_not_logged_for_numeric_dtype() -> None:
    df = pd.DataFrame({"n": [1.5, 2.5, 3.5, 4.5]})
    clean_df, log, _ = run(df)
    assert "to_numeric:n" not in ids(log)
    assert floats(clean_df["n"]) == [1.5, 2.5, 3.5, 4.5]


# ---------------------------------------------------------------- to_bool


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        (["yes", "no", "YES", "No"], [1.0, 0.0, 1.0, 0.0]),
        (["Y", "N", "y", ""], [1.0, 0.0, 1.0, None]),
        (["1.0", "0", "1", "0"], [1.0, 0.0, 1.0, 0.0]),
        (["true", "False", "TRUE", "false"], [1.0, 0.0, 1.0, 0.0]),
        (["off", "on", "on", "off"], [0.0, 1.0, 1.0, 0.0]),
        (["F", "T", "t", "f"], [0.0, 1.0, 1.0, 0.0]),
    ],
)
def test_to_bool(values: list[str], expected: list[float | None]) -> None:
    df = with_row({"b": values})
    clean_df, log, prof = run(df)
    expect_type(prof, "b", "boolean")
    assert "to_bool:b" in ids(log)
    assert is_float64(clean_df["b"])
    assert floats(clean_df["b"]) == expected


# ---------------------------------------------------------------- merge_case


def test_merge_case_uppercase_wins_even_when_less_common() -> None:
    df = with_row({"s": ["nsw", "nsw", "NSW", "VIC", "VIC", "nsw"]})
    clean_df, log, prof = run(df)
    expect_type(prof, "s", "categorical")
    step = log.step("merge_case:s")
    assert step.rows_affected == 3
    assert clean_df["s"].tolist() == ["NSW", "NSW", "NSW", "VIC", "VIC", "NSW"]


def test_merge_case_most_common_wins_without_uppercase() -> None:
    df = with_row({"s": ["Perth", "Perth", "pERTH", "x", "y"]})
    clean_df, log, _ = run(df)
    assert clean_df["s"].tolist() == ["Perth", "Perth", "Perth", "x", "y"]
    assert log.step("merge_case:s").rows_affected == 1


def test_merge_case_tie_goes_to_uppercase() -> None:
    df = with_row({"s": ["nsw", "NSW", "Nsw", "NSW", "nsw", "VIC"]})
    clean_df, log, _ = run(df)
    assert clean_df["s"].tolist() == ["NSW", "NSW", "NSW", "NSW", "NSW", "VIC"]
    assert log.step("merge_case:s").rows_affected == 3


def test_merge_case_tie_without_uppercase_goes_alphabetical() -> None:
    # neither spelling is all uppercase; sorted() puts "Perth" before "pERTH"
    df = with_row({"s": ["pERTH", "Perth", "x", "y", "x"]})
    clean_df, log, _ = run(df)
    assert clean_df["s"].tolist() == ["Perth", "Perth", "x", "y", "x"]
    assert log.step("merge_case:s").rows_affected == 1


def test_merge_case_not_logged_without_variants() -> None:
    df = raw({"s": ["a", "b", "a", "c"]})
    _, log, _ = run(df)
    assert "merge_case:s" not in ids(log)


# ---------------------------------------------------------------- presence_flag


def test_presence_flag_replaces_column_in_place() -> None:
    df = raw(
        {
            "x": ["1.5", "2.5", "3.5", "4.5", "5.5", "6.5"],
            "notes": ["called", "", "N/A", "called", "", ""],
            "y": ["a", "b", "a", "b", "a", "b"],
        }
    )
    clean_df, log, prof = run(df)
    expect_type(prof, "notes", "categorical")
    assert prof.column("notes").n_unique == 1
    assert "presence_flag:notes" in ids(log)
    assert list(clean_df.columns) == ["x", "notes_present", "y"]
    assert clean_df["notes_present"].dtype == np.int64
    assert clean_df["notes_present"].tolist() == [1, 0, 0, 1, 0, 0]
    # 66% missing, but presence flag columns don't get a high_missing flag
    assert "high_missing:notes" not in ids(log)
    assert "high_missing:notes_present" not in ids(log)


# ---------------------------------------------------------------- date_parts


def test_date_parts_iso() -> None:
    df = raw({"a": ["1", "2", "3", "4"], "d": ["2024-01-15", "2024-03-04", "2023-12-31", ""]})
    clean_df, log, prof = run(df)
    expect_type(prof, "d", "datetime")
    assert "date_parts:d" in ids(log)
    assert list(clean_df.columns) == ["a", "d_year", "d_month", "d_dayofweek"]
    for c in ("d_year", "d_month", "d_dayofweek"):
        assert is_float64(clean_df[c]), c
    assert floats(clean_df["d_year"]) == [2024.0, 2024.0, 2023.0, None]
    assert floats(clean_df["d_month"]) == [1.0, 3.0, 12.0, None]
    assert floats(clean_df["d_dayofweek"]) == [0.0, 0.0, 6.0, None]  # Mon=0, Sun=6


def test_date_parts_day_first_and_position() -> None:
    df = raw(
        {
            "d": ["15/01/2024", "16/02/2024", "03/03/2024"],
            "z": ["p", "q", "p"],
        }
    )
    clean_df, _, prof = run(df)
    expect_type(prof, "d", "datetime")
    assert prof.column("d").dayfirst is True
    assert list(clean_df.columns) == ["d_year", "d_month", "d_dayofweek", "z"]
    assert floats(clean_df["d_year"]) == [2024.0, 2024.0, 2024.0]
    assert floats(clean_df["d_month"]) == [1.0, 2.0, 3.0]
    assert floats(clean_df["d_dayofweek"]) == [0.0, 4.0, 6.0]


# ---------------------------------------------------------------- drop_id / row_ids


def _id_frame() -> pd.DataFrame:
    cids = [f"C{1000 + i}" for i in range(12)]
    cids[3] = " C1003 "
    return raw({"cid": cids, "x": [f"{i}.5" for i in range(12)]})


def test_drop_id_and_row_ids() -> None:
    df = _id_frame()
    clean_df, log, prof = run(df)
    expect_type(prof, "cid", "id")
    assert "drop_id:cid" in ids(log)
    assert "cid" not in clean_df.columns
    assert log.row_ids is not None
    assert log.row_ids.name == "cid"
    assert log.row_ids.tolist() == [f"C{1000 + i}" for i in range(12)]  # normalized
    assert log.row_ids.index.equals(clean_df.index)


def test_row_ids_set_even_when_drop_id_disabled() -> None:
    clean_df, log, _ = run(_id_frame(), disabled={"drop_id:cid"})
    assert "cid" in clean_df.columns
    assert log.step("drop_id:cid").enabled is False
    assert log.row_ids is not None
    assert log.row_ids.name == "cid"
    assert log.row_ids.index.equals(clean_df.index)


def test_row_ids_none_without_id_column() -> None:
    _, log, _ = run(raw({"a": ["1", "2", "3"], "b": ["x", "y", "x"]}))
    assert log.row_ids is None


def test_row_ids_follow_dedupe(kitchen_sink_raw: pd.DataFrame) -> None:
    clean_df, log, _ = run(kitchen_sink_raw)
    assert len(clean_df) == len(kitchen_sink_raw) - 1
    assert log.row_ids is not None
    assert log.row_ids.index.equals(clean_df.index)
    assert log.row_ids.name == "customer_id"


# ---------------------------------------------------------------- drop_constant / high_missing


def test_drop_constant_including_all_missing() -> None:
    df = raw(
        {
            "k": ["same", "same", "same", "same"],
            "e": ["", "", "", ""],
            "v": ["1.5", "2.5", "3.5", "4.5"],
        }
    )
    clean_df, log, prof = run(df)
    expect_type(prof, "k", "constant")
    expect_type(prof, "e", "constant")
    assert prof.column("e").all_missing
    assert {"drop_constant:k", "drop_constant:e"} <= set(ids(log))
    assert list(clean_df.columns) == ["v"]
    assert "high_missing:e" not in ids(log)
    assert "high_missing:k" not in ids(log)


def test_drop_constant_all_missing_float_column() -> None:
    df = pd.DataFrame({"e": [np.nan] * 4, "v": ["a", "b", "a", "b"]})
    clean_df, log, _ = run(df)
    assert "drop_constant:e" in ids(log)
    assert "e" not in clean_df.columns


def test_high_missing_flag_leaves_data_alone() -> None:
    s = ["1", "", "", "2", "", "", "3", "", "4", ""]  # 60% missing
    df = with_row({"s": s})
    clean_df, log, prof = run(df)
    expect_type(prof, "s", "numeric")
    step = log.step("high_missing:s")
    assert step.kind == "flag"
    assert step.enabled is True
    assert C.HIGH_MISSING_PCT == 50.0
    assert "s" in clean_df.columns
    assert floats(clean_df["s"]) == [1.0, None, None, 2.0, None, None, 3.0, None, 4.0, None]


def test_high_missing_strictly_above_threshold() -> None:
    s = ["1", "", "2", "", "3", "", "4", "", "5", ""]  # exactly 50%
    _, log, _ = run(with_row({"s": s}))
    assert "high_missing:s" not in ids(log)


# ---------------------------------------------------------------- messy.csv end to end


@pytest.fixture
def messy_clean(messy_raw: pd.DataFrame) -> tuple[pd.DataFrame, C.CleaningLog]:
    clean_df, log, _ = run(messy_raw)
    return clean_df, log


def test_messy_columns_and_order(messy_clean: tuple[pd.DataFrame, C.CleaningLog]) -> None:
    clean_df, _ = messy_clean
    assert list(clean_df.columns) == [
        "signup_date_year",
        "signup_date_month",
        "signup_date_dayofweek",
        "monthly_spend",
        "state",
        "age",
        "notes_present",
        "plan",
        "churned",
    ]
    assert len(clean_df) == 300


def test_messy_id_dropped_and_in_row_ids(
    messy_raw: pd.DataFrame, messy_clean: tuple[pd.DataFrame, C.CleaningLog]
) -> None:
    clean_df, log = messy_clean
    assert "customer_id" not in clean_df.columns
    assert "drop_id:customer_id" in ids(log)
    assert log.row_ids is not None
    assert log.row_ids.name == "customer_id"
    assert log.row_ids.tolist() == messy_raw["customer_id"].tolist()
    assert log.row_ids.index.equals(clean_df.index)


def test_messy_values(messy_clean: tuple[pd.DataFrame, C.CleaningLog]) -> None:
    clean_df, log = messy_clean
    assert is_float64(clean_df["monthly_spend"])
    assert clean_df["monthly_spend"].iloc[0] == pytest.approx(79.03)
    assert clean_df["monthly_spend"].notna().all()

    assert is_float64(clean_df["age"])
    assert int(clean_df["age"].isna().sum()) == 46

    assert is_float64(clean_df["churned"])
    assert set(clean_df["churned"].unique()) == {0.0, 1.0}
    assert int(clean_df["churned"].sum()) == 102

    assert clean_df["notes_present"].dtype == np.int64
    assert int(clean_df["notes_present"].sum()) == 112

    assert "country" not in clean_df.columns
    assert "drop_constant:country" in ids(log)

    assert sorted(clean_df["plan"].unique()) == ["basic", "premium", "pro"]


def test_messy_state_case_merged(messy_clean: tuple[pd.DataFrame, C.CleaningLog]) -> None:
    clean_df, log = messy_clean
    present = set(clean_df["state"].dropna())
    # nsw and NSW collapse into a single spelling (see note on which one in the report)
    spellings = present & {"nsw", "NSW"}
    assert len(spellings) == 1
    winner = spellings.pop()
    assert int((clean_df["state"] == winner).sum()) == 100
    assert present - {winner} == {"QLD", "VIC", "WA"}
    assert int(clean_df["state"].isna().sum()) == 49
    raw_count = {"nsw": 51, "NSW": 49}[winner]
    assert log.step("merge_case:state").rows_affected == 100 - raw_count
    assert log.step("normalize:state").rows_affected == 49


def test_messy_state_merges_to_uppercase_nsw(
    messy_clean: tuple[pd.DataFrame, C.CleaningLog],
) -> None:
    """The spec example and the task brief say nsw/NSW -> NSW for messy.csv.

    Note: messy.csv has 51 'nsw' and 49 'NSW', so the 'most common spelling wins'
    rule alone would give 'nsw'. This test pins the brief's expected outcome.
    """
    clean_df, _ = messy_clean
    assert "nsw" not in set(clean_df["state"].dropna())
    assert "NSW" in set(clean_df["state"].dropna())


def test_messy_signup_date_parts(messy_clean: tuple[pd.DataFrame, C.CleaningLog]) -> None:
    clean_df, log = messy_clean
    assert "signup_date" not in clean_df.columns
    assert "date_parts:signup_date" in ids(log)
    first, second = clean_df.iloc[0], clean_df.iloc[1]
    # 2023-02-23 is a Thursday
    assert (first["signup_date_year"], first["signup_date_month"]) == (2023.0, 2.0)
    assert first["signup_date_dayofweek"] == 3.0
    # 09/04/2024 is day first: 9 April 2024, a Tuesday
    assert (second["signup_date_year"], second["signup_date_month"]) == (2024.0, 4.0)
    assert second["signup_date_dayofweek"] == 1.0


def test_messy_no_imputation(
    messy_raw: pd.DataFrame, messy_clean: tuple[pd.DataFrame, C.CleaningLog]
) -> None:
    """clean() runs before the split, so missing values must stay missing."""
    clean_df, _ = messy_clean
    raw_blank_age = messy_raw["age"].str.strip() == ""
    assert clean_df.loc[raw_blank_age, "age"].isna().all()
    assert clean_df.loc[~raw_blank_age, "age"].notna().all()
    assert int(clean_df["state"].isna().sum()) == 49


def test_messy_log_invariants(messy_clean: tuple[pd.DataFrame, C.CleaningLog]) -> None:
    _, log = messy_clean
    assert_log_invariants(log)
    assert all(s.enabled for s in log.steps)


def test_steps_logged_in_contract_order(kitchen_sink_raw: pd.DataFrame) -> None:
    _, log, _ = run(kitchen_sink_raw)
    positions = [STEP_ORDER.index(s.action) for s in log.steps]
    assert positions == sorted(positions)


# ---------------------------------------------------------------- disabling steps


def test_kitchen_sink_logs_every_step(kitchen_sink_raw: pd.DataFrame) -> None:
    _, log, _ = run(kitchen_sink_raw)
    assert set(ids(log)) == set(SINK_FIX_STEPS + SINK_FLAG_STEPS)
    assert len(ids(log)) == len(set(ids(log)))
    assert log.step("dedupe:*").rows_affected == 1
    assert_log_invariants(log)


def _null_like(v: object) -> bool:
    return isinstance(v, str) and (v != v.strip() or v.strip().lower() in P.MISSING_TOKENS)


def _applied(step: C.CleaningStep, clean_df: pd.DataFrame, n_raw: int) -> bool:
    col = step.column
    checks: dict[str, Callable[[], bool]] = {
        "dedupe": lambda: len(clean_df) == n_raw - step.rows_affected,
        "normalize": lambda: (
            col not in clean_df.columns
            or pd.api.types.is_numeric_dtype(clean_df[col])
            or not clean_df[col].map(_null_like).any()
        ),
        "to_numeric": lambda: is_float64(clean_df[col]),
        "to_bool": lambda: is_float64(clean_df[col]),
        "merge_case": lambda: (
            clean_df[col].dropna().str.lower().nunique() == clean_df[col].dropna().nunique()
        ),
        "presence_flag": lambda: f"{col}_present" in clean_df and col not in clean_df,
        "date_parts": lambda: f"{col}_year" in clean_df and col not in clean_df,
        "drop_id": lambda: col not in clean_df.columns,
        "drop_constant": lambda: col not in clean_df.columns,
    }
    return checks[step.action]()


def _not_applied(step: C.CleaningStep, clean_df: pd.DataFrame, n_raw: int) -> bool:
    col = step.column
    checks: dict[str, Callable[[], bool]] = {
        "dedupe": lambda: len(clean_df) == n_raw,
        # raw blanks stay as "" strings when normalize is off
        "normalize": lambda: (
            col not in clean_df.columns
            or pd.api.types.is_numeric_dtype(clean_df[col])
            or bool(clean_df[col].map(_null_like).any())
        ),
        "to_numeric": lambda: not pd.api.types.is_numeric_dtype(clean_df[col]),
        "to_bool": lambda: not pd.api.types.is_numeric_dtype(clean_df[col]),
        "merge_case": lambda: (
            clean_df[col].dropna().str.lower().nunique() < clean_df[col].dropna().nunique()
        ),
        "presence_flag": lambda: col in clean_df and f"{col}_present" not in clean_df,
        "date_parts": lambda: col in clean_df and f"{col}_year" not in clean_df,
        "drop_id": lambda: col in clean_df.columns,
        "drop_constant": lambda: col in clean_df.columns,
    }
    return checks[step.action]()


@pytest.mark.parametrize("step_id", SINK_FIX_STEPS)
def test_disabling_one_step(kitchen_sink_raw: pd.DataFrame, step_id: str) -> None:
    n_raw = len(kitchen_sink_raw)
    _, full_log, _ = run(kitchen_sink_raw)
    clean_df, log, _ = run(kitchen_sink_raw, disabled={step_id})

    assert log_signature(log) == log_signature(full_log)
    for s in log.steps:
        assert s.enabled is (s.step_id != step_id), s.step_id

    off = log.step(step_id)
    assert _not_applied(off, clean_df, n_raw), f"{step_id} was applied while disabled"
    for s in log.steps:
        if s.kind == "fix" and s.step_id != step_id:
            assert _applied(s, clean_df, n_raw), f"{s.step_id} not applied"


def test_disabled_normalize_keeps_raw_strings() -> None:
    values = [" Sydney", "Perth ", "N/A", "Perth", "Sydney"]
    clean_df, log, _ = run(raw({"city": values}), disabled={"normalize:city"})
    assert log.step("normalize:city").enabled is False
    assert clean_df["city"].tolist() == values


def test_disabled_to_numeric_keeps_strings() -> None:
    clean_df, log, _ = run(raw({"m": ["$1,200", "(5)", "12%", "7"]}), disabled={"to_numeric:m"})
    assert log.step("to_numeric:m").enabled is False
    assert clean_df["m"].tolist() == ["$1,200", "(5)", "12%", "7"]


def test_disabled_dedupe_keeps_rows() -> None:
    df = raw({"a": ["1", "2", "1", "3"], "b": ["x", "y", "x", "z"]})
    clean_df, log, _ = run(df, disabled={"dedupe:*"})
    assert log.step("dedupe:*").enabled is False
    assert log.step("dedupe:*").rows_affected == 1
    assert len(clean_df) == 4


def test_flag_steps_ignore_disabled(kitchen_sink_raw: pd.DataFrame) -> None:
    full_df, _, _ = run(kitchen_sink_raw)
    clean_df, log, _ = run(kitchen_sink_raw, disabled=set(SINK_FLAG_STEPS))
    for step_id in SINK_FLAG_STEPS:
        assert log.step(step_id).enabled is True
    pd.testing.assert_frame_equal(clean_df, full_df)


def test_unknown_disabled_ids_ignored(kitchen_sink_raw: pd.DataFrame) -> None:
    full_df, full_log, _ = run(kitchen_sink_raw)
    clean_df, log, _ = run(kitchen_sink_raw, disabled={"nope:x", "drop_id:not_a_column", ""})
    pd.testing.assert_frame_equal(clean_df, full_df)
    assert log.to_frame().equals(full_log.to_frame())


def test_disable_everything(kitchen_sink_raw: pd.DataFrame) -> None:
    n_raw = len(kitchen_sink_raw)
    clean_df, log, _ = run(kitchen_sink_raw, disabled=set(SINK_FIX_STEPS))
    assert set(ids(log)) == set(SINK_FIX_STEPS + SINK_FLAG_STEPS)
    assert len(clean_df) == n_raw
    assert list(clean_df.columns) == list(kitchen_sink_raw.columns)
    for s in log.steps:
        assert s.enabled is (s.kind == "flag")


# ---------------------------------------------------------------- CleaningLog API


def test_log_step_lookup_and_keyerror(messy_clean: tuple[pd.DataFrame, C.CleaningLog]) -> None:
    _, log = messy_clean
    assert log.step("drop_id:customer_id").column == "customer_id"
    with pytest.raises(KeyError):
        log.step("drop_id:nope")


def test_log_to_frame(messy_clean: tuple[pd.DataFrame, C.CleaningLog]) -> None:
    _, log = messy_clean
    frame = log.to_frame()
    assert list(frame.columns) == LOG_COLUMNS
    assert len(frame) == len(log.steps)
    assert frame["step_id"].tolist() == ids(log)
    assert frame["enabled"].tolist() == [s.enabled for s in log.steps]


def test_empty_log_to_frame_has_columns() -> None:
    frame = C.CleaningLog().to_frame()
    assert list(frame.columns) == LOG_COLUMNS
    assert len(frame) == 0
    with pytest.raises(KeyError):
        C.CleaningLog().step("dedupe:*")


def test_cleaning_step_defaults() -> None:
    step = C.CleaningStep(step_id="drop_id:x", column="x", action="drop_id", reason="r")
    assert step.kind == "fix"
    assert step.enabled is True
    assert step.rows_affected == 0


# ---------------------------------------------------------------- purity and determinism


def test_clean_is_pure(kitchen_sink_raw: pd.DataFrame) -> None:
    df = kitchen_sink_raw
    prof = P.profile(df)
    df_before = df.copy(deep=True)
    prof_before = copy.deepcopy(prof)
    disabled = {"drop_id:customer_id"}
    C.clean(df, prof, disabled)
    pd.testing.assert_frame_equal(df, df_before)
    assert prof == prof_before
    assert disabled == {"drop_id:customer_id"}


def test_clean_output_is_not_a_view(messy_raw: pd.DataFrame) -> None:
    before = messy_raw.copy(deep=True)
    clean_df, _ = C.clean(messy_raw, P.profile(messy_raw))
    clean_df.loc[clean_df.index[0], "plan"] = "changed"
    pd.testing.assert_frame_equal(messy_raw, before)


def test_clean_is_deterministic(kitchen_sink_raw: pd.DataFrame) -> None:
    a_df, a_log, _ = run(kitchen_sink_raw)
    b_df, b_log, _ = run(kitchen_sink_raw)
    pd.testing.assert_frame_equal(a_df, b_df)
    pd.testing.assert_frame_equal(a_log.to_frame(), b_log.to_frame())
    assert a_log.row_ids is not None and b_log.row_ids is not None
    pd.testing.assert_series_equal(a_log.row_ids, b_log.row_ids)


def test_original_index_kept() -> None:
    df = raw({"a": ["1", "2", "3"], "b": ["x", "y", "x"]}, index=[7, 3, 99])
    clean_df, _, _ = run(df)
    assert list(clean_df.index) == [7, 3, 99]


# ---------------------------------------------------------------- edge cases


def test_empty_rows() -> None:
    df = raw({"a": [], "b": []})
    clean_df, log, _ = run(df)
    assert len(clean_df) == 0
    assert "dedupe:*" not in ids(log)
    assert_log_invariants(log)


def test_no_columns_at_all() -> None:
    clean_df, log, _ = run(pd.DataFrame())
    assert clean_df.shape == (0, 0)
    assert log.steps == []
    assert log.row_ids is None


def test_single_row() -> None:
    clean_df, log, _ = run(raw({"a": ["1"], "b": ["x"]}))
    assert len(clean_df) == 1
    assert_log_invariants(log)


def test_single_column() -> None:
    clean_df, log, _ = run(raw({"m": ["$1", "$2", "$3", ""]}))
    assert list(clean_df.columns) == ["m"]
    assert floats(clean_df["m"]) == [1.0, 2.0, 3.0, None]
    assert_log_invariants(log)


def test_all_columns_constant() -> None:
    df = raw({"k1": ["a"] * 5, "k2": ["7"] * 5})
    clean_df, log, _ = run(df)
    assert log.step("dedupe:*").rows_affected == 4
    assert {"drop_constant:k1", "drop_constant:k2"} <= set(ids(log))
    assert clean_df.shape == (1, 0)


def test_plain_read_csv_frame() -> None:
    """Plain pd.read_csv already made numbers and NaN. Clean should still work."""
    df = pd.read_csv(MESSY)
    clean_df, log, _ = run(df)
    assert_log_invariants(log)
    assert "to_numeric:age" not in ids(log)
    assert "normalize:age" not in ids(log)
    assert "to_numeric:monthly_spend" in ids(log)
    assert "customer_id" not in clean_df.columns
    assert "country" not in clean_df.columns
    assert is_float64(clean_df["monthly_spend"])
    np.testing.assert_array_equal(clean_df["age"].to_numpy(), df["age"].to_numpy())
    assert set(clean_df["churned"].unique()) == {0.0, 1.0}


def test_unicode_headers() -> None:
    df = raw({"prix €": ["$1", "$2", "$3", "$4"], "état": ["a", "B", "b", "a"]})
    clean_df, log, _ = run(df)
    assert "to_numeric:prix €" in ids(log)
    assert is_float64(clean_df["prix €"])
    assert "état" in clean_df.columns


def test_duplicated_headers_from_csv() -> None:
    df = P.read_csv_raw(io.StringIO("x,x\n1.5,a\n2.5,b\n3.5,a\n"))
    clean_df, log, _ = run(df)
    assert list(clean_df.columns) == ["x", "x.1"]
    assert is_float64(clean_df["x"])
    assert_log_invariants(log)


def test_mixed_types_stay_categorical() -> None:
    values = ["1", "2", "abc", "3", "def", "4"]
    clean_df, log, prof = run(raw({"m": values}))
    expect_type(prof, "m", "categorical")
    assert "to_numeric:m" not in ids(log)
    assert clean_df["m"].tolist() == values


def test_mixed_python_objects_do_not_crash() -> None:
    df = pd.DataFrame({"m": [1, "a", 2.5, None, "b"], "v": ["x", "y", "x", "y", "x"]})
    clean_df, log, _ = run(df)
    assert len(clean_df) == 5
    assert_log_invariants(log)


def test_huge_cardinality() -> None:
    n = 3000
    df = raw({"code": [f"item{i}" for i in range(n)], "v": [f"{i % 50}.5" for i in range(n)]})
    clean_df, log, _ = run(df)
    assert len(clean_df) == n
    assert_log_invariants(log)


# ---------------------------------------------------------------- sample data


def test_sample_csvs_clean(sample_name: str) -> None:
    df = P.read_csv_raw(DATA / sample_name)
    prof = P.profile(df)
    clean_df, log = C.clean(df, prof)
    assert_log_invariants(log)
    assert clean_df.index.isin(df.index).all()
    assert ("dedupe:*" in ids(log)) == (prof.n_duplicate_rows > 0)
    assert len(clean_df) == len(df) - prof.n_duplicate_rows
    for col in prof.columns:
        if col.dtype == "numeric" and col.name in clean_df.columns:
            assert is_float64(clean_df[col.name]), col.name


# ---------------------------------------------------------------- review fixes


def test_high_missing_not_flagged_for_text() -> None:
    n = 20
    words = "this is a fairly long free text note"
    df = raw(
        {
            "note": [f"{words} {i}" if i < 8 else "" for i in range(n)],
            "x": [f"{i}.5" for i in range(n)],
        }
    )
    _, log, prof = run(df)
    assert prof.column("note").dtype == "text"
    assert "high_missing:note" not in ids(log)


def test_high_missing_datetime_counts_missing_rows() -> None:
    n = 20
    df = raw(
        {
            "when": [f"2024-01-{i + 1:02d}" if i < 8 else "" for i in range(n)],
            "x": [f"{i}.5" for i in range(n)],
        }
    )
    _, log, prof = run(df)
    expect_type(prof, "when", "datetime")
    assert log.step("high_missing:when").rows_affected == 12
