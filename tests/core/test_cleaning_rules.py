"""Phase 2 rework: new cleaning rules (written test-first against the approved spec, B).

Covers: only logging real changes, placeholder codes, impossible values with the
10% guard, invalid dates (with a fixed reference_date) and invalid times.
Every frame is all-string like read_csv_raw gives and goes through the real profiler.
"""

import datetime as dt
import inspect

import numpy as np
import pandas as pd
import pytest

from app.core import cleaning as C
from app.core import profiling as P
from tests.conftest import DATA, REFERENCE_DATE, SAMPLE_NAMES

REF = pd.Timestamp(REFERENCE_DATE)
FLAG_ACTIONS = {"high_missing", "impossible_skipped", "unparseable_dates"}
PLACEHOLDER_CODES = [-1, 99, 999, 9999, -99, -999, -9999]


# ---------------------------------------------------------------- helpers


def raw(cols: dict[str, list[str]]) -> pd.DataFrame:
    return pd.DataFrame({k: pd.Series(v, dtype=str) for k, v in cols.items()})


def with_row(cols: dict[str, list[str]]) -> pd.DataFrame:
    """raw() plus a unique float column "r", so no row is an exact duplicate."""
    n = len(next(iter(cols.values())))
    return raw({**cols, "r": [f"{i}.5" for i in range(n)]})


def run(
    df: pd.DataFrame, disabled: set[str] | None = None, reference_date: object = REF
) -> tuple[pd.DataFrame, C.CleaningLog, P.DatasetProfile]:
    prof = P.profile(df)
    clean_df, log = C.clean(df, prof, disabled, reference_date=reference_date)
    return clean_df, log, prof


def ids(log: C.CleaningLog) -> list[str]:
    return [s.step_id for s in log.steps]


def floats(series: pd.Series) -> list[float | None]:
    return [None if pd.isna(v) else float(v) for v in series]


def numbers(n: int, start: float = 10.0) -> list[str]:
    """n distinct plain numbers, start, start+1, ... (no placeholder codes in 10..60)."""
    return [f"{start + i:g}" for i in range(n)]


def dates(n: int, start: str = "2023-01-01") -> list[str]:
    return [d.date().isoformat() for d in pd.date_range(start, periods=n, freq="7D")]


def log_signature(log: C.CleaningLog) -> list[tuple[object, ...]]:
    return [(s.step_id, s.column, s.action, s.reason, s.kind, s.rows_affected) for s in log.steps]


def assert_fix_steps_change_something(log: C.CleaningLog) -> None:
    for s in log.steps:
        assert s.step_id == f"{s.action}:{s.column}"
        assert s.reason.strip() and "—" not in s.reason, s.step_id
        if s.action in FLAG_ACTIONS:
            assert s.kind == "flag", s.step_id
        else:
            assert s.kind == "fix", s.step_id
            assert s.rows_affected > 0, f"{s.step_id} logged but changed nothing"


# ---------------------------------------------------------------- signature


def test_clean_takes_reference_date_defaulting_to_none() -> None:
    param = inspect.signature(C.clean).parameters["reference_date"]
    assert param.default is None


@pytest.mark.parametrize(
    "ref",
    [REFERENCE_DATE, pd.Timestamp(REFERENCE_DATE), dt.date(2025, 1, 1)],
    ids=["str", "Timestamp", "date"],
)
def test_reference_date_accepts_str_timestamp_and_date(ref: object) -> None:
    df = with_row({"visit": dates(19) + ["2025-06-01"]})
    clean_df, log, _ = run(df, reference_date=ref)
    assert log.step("invalid_dates:visit").rows_affected == 1
    assert pd.isna(clean_df["visit"].iloc[19])


def test_reference_date_none_means_today() -> None:
    df = with_row({"visit": dates(19) + ["2099-06-01"]})
    clean_df, log, _ = run(df, reference_date=None)
    assert log.step("invalid_dates:visit").rows_affected == 1
    assert clean_df["visit"].iloc[:19].notna().all()


# ---------------------------------------------------------------- only log real changes


def test_iris_logs_no_to_numeric_steps() -> None:
    df = P.read_csv_raw(DATA / "iris.csv")
    clean_df, log, _ = run(df)
    assert not [s for s in log.steps if s.action == "to_numeric"]
    assert ids(log) == ["dedupe:*"]
    for c in clean_df.columns[:4]:
        assert clean_df[c].dtype == np.float64, c  # still parsed, just not logged


def test_breast_cancer_zero_one_target_is_silent() -> None:
    df = P.read_csv_raw(DATA / "breast_cancer.csv")
    clean_df, log, _ = run(df)
    assert "to_bool:diagnosis" not in ids(log)
    assert clean_df["diagnosis"].dtype == np.float64
    assert set(clean_df["diagnosis"].unique()) == {0.0, 1.0}


@pytest.mark.parametrize("name", SAMPLE_NAMES)
def test_every_fix_step_changes_something(name: str) -> None:
    _, log, _ = run(P.read_csv_raw(DATA / name))
    assert_fix_steps_change_something(log)


def test_plain_numbers_parsed_silently() -> None:
    df = with_row({"m": ["5.1", "4.9", "6.0", "7.25"]})
    clean_df, log, _ = run(df)
    assert "to_numeric:m" not in ids(log)
    assert clean_df["m"].dtype == np.float64
    assert floats(clean_df["m"]) == [5.1, 4.9, 6.0, 7.25]


def test_to_numeric_counts_only_disguised_values() -> None:
    df = with_row({"m": ["$1,200", "$5", "7", "8"]})
    clean_df, log, _ = run(df)
    assert log.step("to_numeric:m").rows_affected == 2
    assert floats(clean_df["m"]) == [1200.0, 5.0, 7.0, 8.0]


def test_to_numeric_counts_values_that_became_missing() -> None:
    df = with_row({"m": numbers(19) + ["oops"]})
    clean_df, log, _ = run(df)
    assert log.step("to_numeric:m").rows_affected == 1
    assert pd.isna(clean_df["m"].iloc[19])


def test_to_numeric_counts_disguised_plus_failed() -> None:
    df = with_row({"m": ["$10", "(5)"] + numbers(17) + ["oops"]})
    _, log, _ = run(df)
    assert log.step("to_numeric:m").rows_affected == 3


def test_zero_one_strings_parsed_silently() -> None:
    df = with_row({"b": ["1", "0", "1", "0", ""]})
    clean_df, log, _ = run(df)
    assert "to_bool:b" not in ids(log)
    assert clean_df["b"].dtype == np.float64
    assert floats(clean_df["b"]) == [1.0, 0.0, 1.0, 0.0, None]


def test_to_bool_logged_for_words_and_counts_them() -> None:
    df = with_row({"b": ["yes", "no", "YES", "No", ""]})
    clean_df, log, _ = run(df)
    assert log.step("to_bool:b").rows_affected == 4
    assert floats(clean_df["b"]) == [1.0, 0.0, 1.0, 0.0, None]


def test_date_strings_parsed_silently_and_kept_in_place() -> None:
    df = with_row({"a": ["x", "y", "x", "z"], "d": ["2024-01-15", "2024-03-04", "2023-12-31", ""]})
    clean_df, log, _ = run(df)
    assert [s for s in log.steps if s.column == "d" and s.action != "normalize"] == []
    assert "date_parts:d" not in ids(log)
    assert list(clean_df.columns) == ["a", "d", "r"]
    assert str(clean_df["d"].dtype) == "datetime64[ns]"
    assert clean_df["d"].iloc[0] == pd.Timestamp("2024-01-15")
    assert pd.isna(clean_df["d"].iloc[3])


def test_no_date_part_columns_anywhere() -> None:
    df = with_row({"d": dates(10), "v": numbers(10)})
    clean_df, log, _ = run(df)
    assert not [s for s in log.steps if s.action == "date_parts"]
    assert not [c for c in clean_df.columns if c.startswith("d_")]


def test_tz_strings_become_utc_naive_in_clean_df() -> None:
    df = with_row({"t": [f"2024-01-{d:02d}T05:00:00+10:00" for d in range(1, 21)]})
    clean_df, _, _ = run(df)
    out = clean_df["t"]
    assert str(out.dtype) == "datetime64[ns]"
    assert out.iloc[0] == pd.Timestamp("2023-12-31 19:00:00")


def test_tz_aware_dtype_becomes_utc_naive_in_clean_df() -> None:
    values = pd.date_range("2024-01-01", periods=20, tz="Australia/Sydney")
    df = pd.DataFrame({"t": values, "v": numbers(20)})
    clean_df, _, _ = run(df)
    out = clean_df["t"]
    assert str(out.dtype) == "datetime64[ns]"
    assert out.dt.tz is None
    assert out.iloc[0] == pd.Timestamp("2023-12-31 13:00:00")


def test_time_strings_parsed_silently() -> None:
    df = with_row({"t": ["14:30", "9:05", "2:30 PM", "2:30pm", "14:30:15"] * 4})
    clean_df, log, prof = run(df)
    assert prof.column("t").dtype == "time"
    assert "invalid_times:t" not in ids(log)
    assert clean_df["t"].iloc[:5].tolist() == [
        "14:30:00",
        "09:05:00",
        "14:30:00",
        "14:30:00",
        "14:30:15",
    ]


# ---------------------------------------------------------------- placeholder codes


@pytest.mark.parametrize("code", PLACEHOLDER_CODES)
def test_each_placeholder_code(code: int) -> None:
    values = numbers(28) + [str(code), str(code)]  # others are 10..37
    clean_df, log, _ = run(with_row({"score": values}))
    step = log.step("placeholder:score")
    assert step.kind == "fix"
    assert step.rows_affected == 2
    assert str(code) in step.reason
    assert clean_df["score"].iloc[28:].isna().all()
    assert clean_df["score"].iloc[:28].notna().all()


def test_placeholder_needs_two_occurrences() -> None:
    clean_df, log, _ = run(with_row({"score": numbers(29) + ["999"]}))
    assert "placeholder:score" not in ids(log)
    assert clean_df["score"].iloc[29] == 999.0


@pytest.mark.parametrize(("top", "fires"), [(499.0, True), (499.5, False), (600.0, False)])
def test_positive_placeholder_must_be_over_twice_the_max(top: float, fires: bool) -> None:
    values = numbers(27) + [f"{top:g}", "999", "999"]
    _, log, _ = run(with_row({"score": values}))
    assert ("placeholder:score" in ids(log)) is fires


def test_negative_placeholder_needs_nonnegative_others() -> None:
    temps = [f"{t}" for t in range(-10, 18)]  # real negatives, so -1 is just a temperature
    clean_df, log, _ = run(with_row({"temp": [*temps, "-1", "-1"]}))
    assert "placeholder:temp" not in ids(log)
    assert int((clean_df["temp"] == -1).sum()) == 3


def test_placeholder_reason_names_codes_and_counts() -> None:
    values = numbers(26) + ["999", "999", "-1", "-1"]
    clean_df, log, _ = run(with_row({"score": values}))
    step = log.step("placeholder:score")
    assert step.rows_affected == 4
    assert "999" in step.reason and "-1" in step.reason
    assert "2" in step.reason
    assert int(clean_df["score"].isna().sum()) == 4


def test_placeholder_only_for_numeric_columns() -> None:
    values = ["a", "b", "c", "999", "999"] * 4
    clean_df, log, prof = run(with_row({"code": values}))
    assert prof.column("code").dtype == "categorical"
    assert "placeholder:code" not in ids(log)
    assert int((clean_df["code"] == "999").sum()) == 8


def test_non_codes_left_alone() -> None:
    clean_df, log, _ = run(with_row({"score": numbers(28) + ["888", "888"]}))
    assert "placeholder:score" not in ids(log)
    assert int((clean_df["score"] == 888).sum()) == 2


def test_placeholder_runs_before_impossible() -> None:
    ages = [str(20 + i) for i in range(27)] + ["999", "999", "999"]
    clean_df, log, _ = run(with_row({"age": ages}))
    assert log.step("placeholder:age").rows_affected == 3
    assert "impossible:age" not in ids(log)  # 999 is a code, not an impossible age
    assert int(clean_df["age"].isna().sum()) == 3
    order = ids(log)
    assert all(
        order.index(a) < order.index(b)
        for a in order
        if a.startswith("placeholder:")
        for b in order
        if b.startswith("impossible:")
    )


# ---------------------------------------------------------------- impossible values


def _two_bad(low: str, high: str, n: int = 30) -> list[str]:
    """n values, 2 of them bad (6.7%, under the 10% guard)."""
    return [str(20 + i) for i in range(n - 2)] + [low, high]


@pytest.mark.parametrize(
    "name", ["age", "Age", "AGE", "patient_age", "patientAge", "age (years)", "Age.At.Visit"]
)
def test_age_rule_matches_word_tokens(name: str) -> None:
    clean_df, log, _ = run(with_row({name: _two_bad("-3", "150")}))
    step = log.step(f"impossible:{name}")
    assert step.rows_affected == 2
    assert clean_df[name].iloc[-2:].isna().all()
    assert clean_df[name].iloc[:-2].notna().all()


@pytest.mark.parametrize("name", ["usage", "page", "average", "stage", "agent", "mileage", "x"])
def test_age_rule_ignores_substrings(name: str) -> None:
    clean_df, log, _ = run(with_row({name: _two_bad("-3", "150")}))
    assert f"impossible:{name}" not in ids(log)
    assert clean_df[name].notna().all()


def test_age_boundaries_kept() -> None:
    clean_df, log, _ = run(with_row({"age": _two_bad("0", "120")}))
    assert "impossible:age" not in ids(log)
    assert clean_df["age"].notna().all()


def test_percentage_matches_pct_rule_only() -> None:
    clean_df, log, _ = run(with_row({"percentage": _two_bad("-1.5", "110")}))
    assert log.step("impossible:percentage").rows_affected == 2
    assert clean_df["percentage"].iloc[-2:].isna().all()


@pytest.mark.parametrize("name", ["pct", "percent", "percentage", "discount_pct", "growthPct"])
def test_pct_rule(name: str) -> None:
    clean_df, log, _ = run(with_row({name: _two_bad("-4", "101")}))
    assert log.step(f"impossible:{name}").rows_affected == 2
    assert floats(clean_df[name].iloc[-2:]) == [None, None]


def test_pct_rule_from_percent_disguise() -> None:
    values = [f"{20 + i}%" for i in range(19)] + ["150%"]
    clean_df, log, prof = run(with_row({"rate": values}))
    assert "percent" in prof.column("rate").disguise
    assert log.step("impossible:rate").rows_affected == 1
    assert pd.isna(clean_df["rate"].iloc[19])


@pytest.mark.parametrize(
    "name",
    [
        "price",
        "cost",
        "qty",
        "quantity",
        "count",
        "weight",
        "height",
        "salary",
        "income",
        "unit_price",
        "itemCount",
    ],
)
def test_non_negative_rule(name: str) -> None:
    clean_df, log, _ = run(with_row({name: _two_bad("-5", "-7.5")}))
    assert log.step(f"impossible:{name}").rows_affected == 2
    assert clean_df[name].iloc[-2:].isna().all()


@pytest.mark.parametrize(
    "name",
    [
        "bmi",
        "blood_pressure",
        "bloodPressure",
        "bp",
        "systolic",
        "diastolic",
        "heart_rate",
        "HeartRate",
        "glucose",
        "weight",
        "height",
    ],
)
def test_zero_rule(name: str) -> None:
    clean_df, log, _ = run(with_row({name: _two_bad("0", "0.0")}))
    assert log.step(f"impossible:{name}").rows_affected == 2
    assert clean_df[name].iloc[-2:].isna().all()


@pytest.mark.parametrize("name", ["pressure", "rate", "heart", "blood", "score"])
def test_zero_rule_needs_the_full_token_pair(name: str) -> None:
    clean_df, log, _ = run(with_row({name: _two_bad("0", "0.0")}))
    assert f"impossible:{name}" not in ids(log)
    assert clean_df[name].notna().all()


def test_one_step_per_column_with_every_rule() -> None:
    clean_df, log, _ = run(with_row({"weight": _two_bad("0", "-2")}))
    steps = [s for s in log.steps if s.column == "weight" and s.action == "impossible"]
    assert len(steps) == 1
    assert steps[0].rows_affected == 2
    assert clean_df["weight"].iloc[-2:].isna().all()


def test_impossible_ignores_missing_values() -> None:
    values = _two_bad("-3", "150") + [""] * 5
    clean_df, log, _ = run(with_row({"age": values}))
    assert log.step("impossible:age").rows_affected == 2
    assert int(clean_df["age"].isna().sum()) == 7


# ---------------------------------------------------------------- 10% guard


def _standardised_ages(n: int = 30) -> list[str]:
    rng = np.random.default_rng(0)
    return [f"{v:.3f}" for v in rng.normal(0, 1, n)]


def test_guard_skips_standardised_age_and_flags_it() -> None:
    values = _standardised_ages()
    clean_df, log, _ = run(with_row({"age": values}))
    assert "impossible:age" not in ids(log)
    flag = log.step("impossible_skipped:age")
    assert flag.kind == "flag"
    assert flag.enabled is True
    assert "age" in flag.reason and "skip" in flag.reason.lower()
    assert "—" not in flag.reason
    np.testing.assert_allclose(clean_df["age"].to_numpy(), [float(v) for v in values])


def test_guard_flag_cannot_be_disabled() -> None:
    df = with_row({"age": _standardised_ages()})
    full_df, full_log, _ = run(df)
    clean_df, log, _ = run(df, disabled={"impossible_skipped:age"})
    assert log.step("impossible_skipped:age").enabled is True
    pd.testing.assert_frame_equal(clean_df, full_df)
    assert log_signature(log) == log_signature(full_log)


@pytest.mark.parametrize(("n_bad", "runs"), [(3, True), (4, False)])
def test_guard_boundary_is_ten_percent(n_bad: int, runs: bool) -> None:
    values = [str(20 + i) for i in range(30 - n_bad)] + ["-5"] * n_bad  # 10% vs 13%
    clean_df, log, _ = run(with_row({"price": values}))
    assert ("impossible:price" in ids(log)) is runs
    assert ("impossible_skipped:price" in ids(log)) is not runs
    assert int(clean_df["price"].isna().sum()) == (n_bad if runs else 0)


def test_guard_is_per_rule() -> None:
    # weight: one negative (fine, rule runs) and lots of zeros (guard skips the zero rule)
    values = [str(50 + i) for i in range(21)] + ["-2"] + ["0"] * 8
    clean_df, log, _ = run(with_row({"weight": values}))
    assert log.step("impossible:weight").rows_affected == 1
    assert log.step("impossible_skipped:weight").kind == "flag"
    assert pd.isna(clean_df["weight"].iloc[21])
    assert int((clean_df["weight"] == 0).sum()) == 8


# ---------------------------------------------------------------- invalid dates


def _visit(bad: str, name: str = "visit_date", n: int = 24) -> pd.DataFrame:
    return with_row({name: dates(n - 1) + [bad]})


@pytest.mark.parametrize(
    "bad",
    ["2025-01-02", "2099-05-01", "1899-12-31", "1850-03-02"],
    ids=["day after ref", "far future", "just before 1900", "1850"],
)
def test_invalid_dates_future_and_too_old(bad: str) -> None:
    clean_df, log, _ = run(_visit(bad))
    step = log.step("invalid_dates:visit_date")
    assert step.kind == "fix"
    assert step.rows_affected == 1
    assert pd.isna(clean_df["visit_date"].iloc[-1])
    assert clean_df["visit_date"].iloc[:-1].notna().all()


@pytest.mark.parametrize("ok", ["2025-01-01", "1900-01-02", "2024-12-31"])
def test_valid_edge_dates_kept(ok: str) -> None:
    clean_df, log, _ = run(_visit(ok))
    assert "invalid_dates:visit_date" not in ids(log)
    assert clean_df["visit_date"].iloc[-1] == pd.Timestamp(ok)


@pytest.mark.parametrize("placeholder", ["1900-01-01", "1970-01-01"])
def test_placeholder_dates(placeholder: str) -> None:
    clean_df, log, prof = run(_visit(placeholder))
    assert prof.column("visit_date").dtype == "datetime"
    step = log.step("invalid_dates:visit_date")
    assert step.rows_affected == 1
    assert pd.isna(clean_df["visit_date"].iloc[-1])


# 9999-12-31 and junk can't be stored as dates, so switching a step off can't bring
# them back. They're reported by a flag instead of the toggleable invalid_dates step.
@pytest.mark.parametrize("bad", ["9999-12-31", "not a date"])
def test_unparseable_dates_are_a_flag(bad: str) -> None:
    clean_df, log, prof = run(_visit(bad, n=30))
    assert prof.column("visit_date").dtype == "datetime"
    assert "invalid_dates:visit_date" not in ids(log)
    flag = log.step("unparseable_dates:visit_date")
    assert flag.kind == "flag" and flag.rows_affected == 1
    assert bad in flag.reason
    assert pd.isna(clean_df["visit_date"].iloc[-1])


def test_invalid_dates_one_step_counts_every_reason() -> None:
    values = dates(36) + ["2099-05-01", "1850-03-02", "1970-01-01", "9999-12-31"]
    clean_df, log, _ = run(with_row({"visit_date": values}))
    step = log.step("invalid_dates:visit_date")
    assert step.rows_affected == 3
    actions = [s.action for s in log.steps if s.column == "visit_date"]
    assert actions == ["invalid_dates", "unparseable_dates"]
    assert clean_df["visit_date"].iloc[-4:].isna().all()


def test_no_invalid_dates_step_when_all_fine() -> None:
    _, log, _ = run(_visit("2024-02-29"))
    assert "invalid_dates:visit_date" not in ids(log)


@pytest.mark.parametrize(
    "name",
    [
        "end_date",
        "endDate",
        "start_time",
        "renewal_date",
        "next_visit",
        "expected_delivery",
        "expiry",
        "expiration_date",
        "due",
        "deadline",
        "scheduled_for",
        "appointment",
    ],
)
def test_future_rule_exempt_names(name: str) -> None:
    clean_df, log, _ = run(_visit("2099-05-01", name=name))
    assert f"invalid_dates:{name}" not in ids(log)
    assert clean_df[name].iloc[-1] == pd.Timestamp("2099-05-01")


@pytest.mark.parametrize("name", ["weekend", "spend", "spending", "visit_date", "endorsed_on"])
def test_future_rule_not_exempt_names(name: str) -> None:
    clean_df, log, _ = run(_visit("2099-05-01", name=name))
    assert log.step(f"invalid_dates:{name}").rows_affected == 1
    assert pd.isna(clean_df[name].iloc[-1])


def test_exempt_names_still_drop_old_and_placeholder_dates() -> None:
    values = dates(27) + ["2099-05-01", "1850-03-02", "1970-01-01"]
    clean_df, log, _ = run(with_row({"end_date": values}))
    assert log.step("invalid_dates:end_date").rows_affected == 2
    assert clean_df["end_date"].iloc[-3] == pd.Timestamp("2099-05-01")
    assert clean_df["end_date"].iloc[-2:].isna().all()


# ---------------------------------------------------------------- invalid times


def test_invalid_times_logged_when_values_go_missing() -> None:
    values = ["14:30", "9:05", "2:30 PM", "18:00"] * 5
    values[7] = "25:61"
    clean_df, log, prof = run(with_row({"t": values}))
    assert prof.column("t").dtype == "time"
    step = log.step("invalid_times:t")
    assert step.kind == "fix"
    assert step.rows_affected == 1
    assert pd.isna(clean_df["t"].iloc[7])
    assert clean_df["t"].iloc[0] == "14:30:00"


def test_blank_times_are_not_invalid() -> None:
    values = ["14:30", "9:05", "", "18:00"] * 5
    _, log, _ = run(with_row({"t": values}))
    assert "invalid_times:t" not in ids(log)


# ---------------------------------------------------------------- disabling the new steps


def test_disable_placeholder_keeps_codes() -> None:
    clean_df, log, _ = run(with_row({"score": numbers(28) + ["999", "999"]}), {"placeholder:score"})
    assert log.step("placeholder:score").enabled is False
    assert floats(clean_df["score"].iloc[-2:]) == [999.0, 999.0]


def test_disable_impossible_keeps_values() -> None:
    clean_df, log, _ = run(with_row({"age": _two_bad("-3", "150")}), {"impossible:age"})
    assert log.step("impossible:age").enabled is False
    assert floats(clean_df["age"].iloc[-2:]) == [-3.0, 150.0]


def test_disable_invalid_dates_keeps_parseable_dates() -> None:
    values = dates(26) + ["2099-05-01", "1850-03-02", "1900-01-01", "1970-01-01"]
    clean_df, log, _ = run(with_row({"visit_date": values}), {"invalid_dates:visit_date"})
    assert log.step("invalid_dates:visit_date").enabled is False
    kept = clean_df["visit_date"].iloc[-4:].tolist()
    assert kept == [pd.Timestamp(v) for v in values[-4:]]


def test_disable_invalid_times_keeps_the_raw_value() -> None:
    values = ["14:30", "9:05", "2:30 PM", "18:00"] * 5
    values[7] = "25:61"
    clean_df, log, _ = run(with_row({"t": values}), {"invalid_times:t"})
    assert log.step("invalid_times:t").enabled is False
    assert not pd.isna(clean_df["t"].iloc[7])


# ---------------------------------------------------------------- messy_clinic.csv

CLINIC_FIX_STEPS = {
    "normalize:glucose": 6,
    "normalize:city": 5,
    "to_bool:smoker": 80,
    "to_bool:outcome": 80,
    "placeholder:age": 3,
    "placeholder:blood_pressure": 2,
    "placeholder:quantity": 3,
    "impossible:age": 2,
    "impossible:bmi": 3,
    "impossible:blood_pressure": 2,
    "impossible:discount_pct": 3,
    "impossible:price": 3,
    "invalid_dates:visit_date": 4,
    "unparseable_dates:visit_date": 1,  # 9999-12-31, a flag
    "invalid_times:checkin_time": 1,
    "drop_id:patient_id": 80,
}


@pytest.fixture
def clinic_clean(clinic_raw: pd.DataFrame) -> tuple[pd.DataFrame, C.CleaningLog]:
    clean_df, log, _ = run(clinic_raw)
    return clean_df, log


def test_clinic_logs_exactly_the_expected_steps(
    clinic_clean: tuple[pd.DataFrame, C.CleaningLog],
) -> None:
    _, log = clinic_clean
    assert {s.step_id: s.rows_affected for s in log.steps} == CLINIC_FIX_STEPS
    assert_fix_steps_change_something(log)
    assert not [s for s in log.steps if s.action == "to_numeric"]


def test_clinic_values(
    clinic_raw: pd.DataFrame, clinic_clean: tuple[pd.DataFrame, C.CleaningLog]
) -> None:
    clean_df, log = clinic_clean
    assert "patient_id" not in clean_df.columns
    assert log.row_ids is not None and log.row_ids.iloc[0] == "P0001"
    assert len(clean_df) == 80

    for c in ["age", "bmi", "blood_pressure", "glucose", "discount_pct", "price", "quantity"]:
        assert clean_df[c].dtype == np.float64, c
    assert clean_df["age"].iloc[[5, 12, 20, 40, 60]].isna().all()  # -3, 150, three 999s
    assert int(clean_df["age"].isna().sum()) == 5
    assert clean_df["bmi"].iloc[[7, 33, 66]].isna().all()
    assert clean_df["blood_pressure"].iloc[[9, 14, 51, 70]].isna().all()
    assert clean_df["discount_pct"].iloc[[11, 37, 64]].isna().all()
    assert clean_df["price"].iloc[[8, 27, 55]].isna().all()
    assert clean_df["quantity"].iloc[[4, 22, 49]].isna().all()
    assert int(clean_df["quantity"].isna().sum()) == 3

    # the big incomes are real, not placeholders
    assert clean_df["income"].notna().all()
    assert clean_df["income"].max() == 910000.0

    # two-value code stays as labels
    assert set(clean_df["sex"].dropna()) == {"1", "2"}
    assert set(clean_df["smoker"].unique()) == {0.0, 1.0}


def test_clinic_dates_and_times(clinic_clean: tuple[pd.DataFrame, C.CleaningLog]) -> None:
    clean_df, log = clinic_clean
    visit = clean_df["visit_date"]
    assert str(visit.dtype) == "datetime64[ns]"
    assert visit.iloc[[6, 15, 25, 35, 45]].isna().all()
    assert int(visit.isna().sum()) == 5
    assert visit.iloc[0] == pd.Timestamp("2023-01-01")

    discharge = clean_df["discharge_datetime"]
    assert str(discharge.dtype) == "datetime64[ns]"
    assert discharge.notna().all()
    assert "invalid_dates:discharge_datetime" not in [s.step_id for s in log.steps]
    assert discharge.iloc[0] == pd.Timestamp("2023-01-02 08:00:00")
    ok = visit.notna()
    assert (discharge[ok] > visit[ok]).all()

    times = clean_df["checkin_time"]
    assert times.iloc[1:5].tolist() == ["14:30:00", "09:05:00", "14:30:00", "14:30:00"]
    assert times.iloc[13] == "14:30:15"
    assert pd.isna(times.iloc[50])
    assert int(times.isna().sum()) == 1


def test_clinic_does_not_mutate_input(clinic_raw: pd.DataFrame) -> None:
    before = clinic_raw.copy(deep=True)
    prof = P.profile(clinic_raw)
    C.clean(clinic_raw, prof, reference_date=REF)
    pd.testing.assert_frame_equal(clinic_raw, before)


NEW_CLINIC_STEPS = [
    s
    for s in CLINIC_FIX_STEPS
    if s.split(":")[0] in {"placeholder", "impossible", "invalid_dates", "invalid_times"}
]


@pytest.mark.parametrize("step_id", NEW_CLINIC_STEPS)
def test_clinic_disable_each_new_step(clinic_raw: pd.DataFrame, step_id: str) -> None:
    full_df, full_log, _ = run(clinic_raw)
    clean_df, log, _ = run(clinic_raw, {step_id})
    # the log is built as if every step is on
    assert log_signature(log) == log_signature(full_log)
    for s in log.steps:
        assert s.enabled is (s.step_id != step_id or s.kind == "flag"), s.step_id
    col = step_id.split(":")[1]
    changed = full_df[col].isna() & clean_df[col].notna()
    assert int(changed.sum()) >= 1, f"{step_id} still applied while disabled"
    others = [c for c in full_df.columns if c != col]
    pd.testing.assert_frame_equal(clean_df[others], full_df[others])


def test_clinic_disable_placeholder_restores_exact_rows(clinic_raw: pd.DataFrame) -> None:
    # blood_pressure 999 isn't caught by the bp == 0 rule, so it's unambiguous
    clean_df, _, _ = run(clinic_raw, {"placeholder:blood_pressure"})
    assert floats(clean_df["blood_pressure"].iloc[[14, 70]]) == [999.0, 999.0]
    assert clean_df["blood_pressure"].iloc[[9, 51]].isna().all()  # zeros still removed


def test_clinic_disable_impossible_age_keeps_placeholders_removed(clinic_raw: pd.DataFrame) -> None:
    clean_df, _, _ = run(clinic_raw, {"impossible:age"})
    assert floats(clean_df["age"].iloc[[5, 12]]) == [-3.0, 150.0]
    assert clean_df["age"].iloc[[20, 40, 60]].isna().all()
