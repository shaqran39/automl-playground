"""Phase 2 rework: profiling changes (2-value codes, UTC dates, time-only columns).

Written test-first against the approved rework spec, section A.
"""

from typing import Any

import numpy as np
import pandas as pd
import pytest

from app.core import profiling as P
from tests.conftest import DATA


def _col(values: list[Any] | pd.Series, name: str = "col") -> P.ColumnProfile:
    return P.profile(pd.DataFrame({name: values})).column(name)


# ---------------------------------------------------------------- 2-value numeric -> categorical


@pytest.mark.parametrize(
    "values",
    [
        ["1", "2"] * 10,
        ["1.0", "2.0"] * 10,
        ["5", "10", "10"] * 7,
        ["-1", "1"] * 10,
        [1, 2, 2, 1] * 5,
        ["1", "2", ""] * 7,  # missing values don't change it
    ],
    ids=["1/2", "1.0/2.0", "5/10", "-1/1", "int dtype", "with blanks"],
)
def test_two_value_numeric_is_categorical(values: list[Any]) -> None:
    col = _col(values, "sex")
    assert col.dtype == "categorical"
    assert any("two values" in r.lower() for r in col.reasons), col.reasons


def test_two_value_reason_names_the_values() -> None:
    col = _col(["1", "2"] * 10, "sex")
    assert any("1" in r and "2" in r and "two values" in r.lower() for r in col.reasons)
    assert all("—" not in r for r in col.reasons)


@pytest.mark.parametrize(
    "values",
    [["0", "1"] * 10, ["1.0", "0.0"] * 10, [0, 1, 1, 0] * 5, ["1", "0", "01"] * 7],
    ids=["0/1 str", "1.0/0.0", "0/1 int", "padded 01"],
)
def test_zero_one_stays_boolean(values: list[Any]) -> None:
    assert _col(values).dtype == "boolean"


def test_three_values_stay_numeric() -> None:
    assert _col(["1", "2", "3"] * 10).dtype == "numeric"


def test_diabetes_sex_is_categorical() -> None:
    prof = P.profile(P.read_csv_raw(DATA / "diabetes.csv"))
    sex = prof.column("sex")
    assert sex.dtype == "categorical"
    assert sex.n_unique == 2
    assert any("two values" in r.lower() for r in sex.reasons)
    others = [c.dtype for c in prof.columns if c.name != "sex"]
    assert all(t == "numeric" for t in others)


# ---------------------------------------------------------------- timezones -> UTC naive


def test_tz_aware_series_converted_to_utc() -> None:
    # midnight in Sydney (UTC+11 in January) is 13:00 the day before in UTC
    col = _col(pd.date_range("2024-01-01", periods=20, tz="Australia/Sydney"))
    assert col.dtype == "datetime"
    assert col.date_range == ("2023-12-31", "2024-01-19")
    assert col.has_time is True


def test_offset_strings_converted_to_utc() -> None:
    values = [f"2024-01-{d:02d}T05:00:00+10:00" for d in range(1, 21)]
    col = _col(values)
    assert col.dtype == "datetime"
    assert col.date_range == ("2023-12-31", "2024-01-19")


def test_utc_series_unchanged() -> None:
    col = _col(pd.date_range("2024-01-01", periods=20, tz="UTC"))
    assert col.date_range == ("2024-01-01", "2024-01-20")
    assert col.has_time is False


# ---------------------------------------------------------------- has_time


def test_has_time_defaults_false() -> None:
    assert P.ColumnProfile(name="x", dtype="numeric", missing_pct=0.0, n_unique=1).has_time is False


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ([f"2024-01-{d:02d}" for d in range(1, 21)], False),
        ([f"2024-01-{d:02d} 00:00:00" for d in range(1, 21)], False),
        ([f"2024-01-{d:02d} 14:{d:02d}:00" for d in range(1, 21)], True),
        ([f"2024-01-{d:02d} 00:00:00" for d in range(1, 20)] + ["2024-01-20 09:30:00"], True),
    ],
    ids=["date only", "all midnight", "times", "one non-midnight"],
)
def test_has_time(values: list[str], expected: bool) -> None:
    col = _col(values)
    assert col.dtype == "datetime"
    assert col.has_time is expected


def test_has_time_on_datetime64_column() -> None:
    col = _col(pd.Series(pd.date_range("2024-01-01 08:00", periods=20, freq="h")))
    assert col.dtype == "datetime"
    assert col.has_time is True


# ---------------------------------------------------------------- time-only columns


@pytest.mark.parametrize(
    "values",
    [
        ["14:30", "9:05", "23:59", "0:00"] * 5,
        ["14:30:15", "09:05:00", "23:59:59", "00:00:01"] * 5,
        ["2:30 PM", "11:15 AM", "12:00 PM", "12:45 am"] * 5,
        ["2:30pm", "11:15am", "7:05pm", "1:00am"] * 5,
        ["14:30", "2:30 PM", "9:05:10", "7:05pm"] * 5,
    ],
    ids=["H:MM", "HH:MM:SS", "AM/PM spaced", "am/pm joined", "mixed"],
)
def test_time_only_column(values: list[str]) -> None:
    col = _col(values, "checkin")
    assert col.dtype == "time"
    assert col.date_range is None


def test_time_needs_95_percent_parsing() -> None:
    good = ["14:30", "9:05", "18:45", "7:10"] * 5
    assert _col(good[:19] + ["25:61"]).dtype == "time"  # 95%
    assert _col(good[:18] + ["25:61", "99:99"]).dtype != "time"  # 90%


@pytest.mark.parametrize(
    "bad",
    ["24:00", "12:60", "13:00 PM", "0:30 AM", "14:30:60", "abc"],
)
def test_time_invalid_values_do_not_parse(bad: str) -> None:
    out = P.parse_times(pd.Series([bad]))
    assert pd.isna(out.iloc[0]), bad


def test_time_with_missing_values() -> None:
    col = _col(["14:30", "", "9:05", None, "18:00"] * 5)
    assert col.dtype == "time"
    assert col.missing_pct == pytest.approx(40.0)


def test_datetimes_are_not_time() -> None:
    assert _col([f"2024-01-{d:02d} 14:30" for d in range(1, 21)]).dtype == "datetime"


def test_plain_numbers_are_not_time() -> None:
    assert _col(["1430", "905", "1800", "700"] * 5).dtype == "numeric"


def test_time_reasons_are_plain() -> None:
    col = _col(["14:30", "9:05"] * 10)
    assert col.reasons and all(r.strip() and "—" not in r for r in col.reasons)


# ---------------------------------------------------------------- parse_times


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("14:30", "14:30:00"),
        ("9:05", "09:05:00"),
        ("09:05", "09:05:00"),
        ("14:30:15", "14:30:15"),
        ("0:00", "00:00:00"),
        ("23:59:59", "23:59:59"),
        ("2:30 PM", "14:30:00"),
        ("2:30pm", "14:30:00"),
        ("2:30 am", "02:30:00"),
        ("12:15 AM", "00:15:00"),
        ("12:15 PM", "12:15:00"),
        ("11:59:30 PM", "23:59:30"),
    ],
)
def test_parse_times_values(raw: str, expected: str) -> None:
    out = P.parse_times(pd.Series([raw]))
    assert out.iloc[0] == expected


def test_parse_times_missing_and_invalid_are_none() -> None:
    values = pd.Series(["14:30", None, np.nan, "", "25:61", "nope"], index=[5, 6, 7, 8, 9, 10])
    out = P.parse_times(values)
    assert isinstance(out, pd.Series)
    assert list(out.index) == [5, 6, 7, 8, 9, 10]
    assert out.iloc[0] == "14:30:00"
    assert out.iloc[1:].isna().all()


def test_parse_times_does_not_mutate_input() -> None:
    values = pd.Series(["2:30 PM", "9:05"])
    before = values.copy()
    P.parse_times(values)
    pd.testing.assert_series_equal(values, before)


# ---------------------------------------------------------------- messy_clinic.csv


CLINIC_TYPES: dict[str, str] = {
    "patient_id": "id",
    "age": "numeric",
    "sex": "categorical",
    "bmi": "numeric",
    "blood_pressure": "numeric",
    "glucose": "numeric",
    "discount_pct": "numeric",
    "price": "numeric",
    "quantity": "numeric",
    "income": "numeric",
    "city": "categorical",
    "smoker": "boolean",
    "visit_date": "datetime",
    "discharge_datetime": "datetime",
    "checkin_time": "time",
    "outcome": "boolean",
}


def test_clinic_column_types(clinic_raw: pd.DataFrame) -> None:
    prof = P.profile(clinic_raw)
    assert prof.n_rows == 80
    assert {c.name: c.dtype for c in prof.columns} == CLINIC_TYPES


def test_clinic_has_time_flags(clinic_raw: pd.DataFrame) -> None:
    prof = P.profile(clinic_raw)
    assert prof.column("visit_date").has_time is False
    assert prof.column("discharge_datetime").has_time is True


def test_clinic_profile_is_read_only(clinic_raw: pd.DataFrame) -> None:
    before = clinic_raw.copy(deep=True)
    P.profile(clinic_raw)
    pd.testing.assert_frame_equal(clinic_raw, before)
