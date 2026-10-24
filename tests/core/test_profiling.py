"""Phase 1 profiling tests (written test-first against the agreed contract)."""

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
import pytest

from app.core import profiling as P
from tests.conftest import DATA, MESSY

ALLOWED_TYPES = {"numeric", "categorical", "boolean", "datetime", "text", "id", "constant"}
STAT_KEYS = {"min", "max", "mean", "median", "std", "skew"}
DISGUISES = {"currency", "percent", "thousands_sep", "parens_negative"}


# ---------------------------------------------------------------- helpers


def dtypes_of(df: pd.DataFrame) -> dict[str, str]:
    return {c.name: c.dtype for c in P.profile(df).columns}


def single_dtype(values: list[Any], name: str = "col") -> P.ColumnProfile:
    return P.profile(pd.DataFrame({name: values})).column(name)


def assert_profile_invariants(prof: Any, df: pd.DataFrame) -> None:
    assert prof.n_rows == len(df)
    assert prof.n_cols == df.shape[1]
    assert len(prof.columns) == df.shape[1]
    assert [c.name for c in prof.columns] == [str(c) for c in df.columns]
    for col in prof.columns:
        assert col.dtype in ALLOWED_TYPES, col.name
        assert 0.0 <= col.missing_pct <= 100.0, col.name
        assert 0.0 <= col.parse_rate <= 1.0, col.name
        assert col.n_unique >= 0
        assert col.reasons and all(isinstance(r, str) and r.strip() for r in col.reasons)
        assert len(col.sample_values) <= 5
        assert all(isinstance(v, str) for v in col.sample_values)
        assert len(col.top_values) <= 10
        assert set(col.disguise) <= DISGUISES
        if col.dtype == "numeric":
            assert col.stats is not None and set(col.stats) == STAT_KEYS, col.name
            assert col.outlier_bounds is not None
            assert col.outlier_bounds[0] <= col.outlier_bounds[1]
            assert 0.0 <= col.outlier_pct <= 100.0
            assert col.top_values == {}
        else:
            assert col.stats is None, col.name
            assert col.n_outliers == 0
            assert col.outlier_bounds is None
            assert col.disguise == []
        if col.dtype != "datetime":
            assert col.date_range is None


# ---------------------------------------------------------------- messy.csv

MESSY_TYPES: dict[str, str] = {
    "customer_id": "id",
    "signup_date": "datetime",
    "monthly_spend": "numeric",
    "state": "categorical",
    "age": "numeric",
    "notes": "categorical",
    "country": "constant",
    "plan": "categorical",
    "churned": "boolean",
}


def test_read_csv_raw_keeps_strings() -> None:
    df = P.read_csv_raw(MESSY)
    assert df.shape == (300, 9)
    assert df.isna().sum().sum() == 0  # keep_default_na=False: blanks stay ""
    assert all(isinstance(v, str) for v in df["age"])


def test_messy_shape_and_invariants(messy_df: pd.DataFrame) -> None:
    prof = P.profile(messy_df)
    assert (prof.n_rows, prof.n_cols) == (300, 9)
    assert prof.n_duplicate_rows == 0
    assert_profile_invariants(prof, messy_df)


@pytest.mark.parametrize(("column", "expected"), MESSY_TYPES.items())
def test_messy_column_types(messy_df: pd.DataFrame, column: str, expected: str) -> None:
    assert P.profile(messy_df).column(column).dtype == expected


def test_messy_customer_id(messy_df: pd.DataFrame) -> None:
    col = P.profile(messy_df).column("customer_id")
    assert col.n_unique == 300
    assert col.missing_pct == 0.0


def test_messy_signup_date(messy_df: pd.DataFrame) -> None:
    col = P.profile(messy_df).column("signup_date")
    assert col.dayfirst is True
    assert col.date_ambiguous is False
    assert col.date_format == "mixed"
    assert col.parse_rate == pytest.approx(1.0)
    assert col.date_range == ("2023-01-05", "2024-08-20")


def test_messy_monthly_spend(messy_df: pd.DataFrame) -> None:
    col = P.profile(messy_df).column("monthly_spend")
    assert "currency" in col.disguise
    assert col.parse_rate == pytest.approx(1.0)
    assert col.missing_pct == 0.0
    assert col.stats is not None
    assert col.stats["min"] == pytest.approx(5.57)
    assert col.stats["max"] == pytest.approx(603.63)


def test_messy_monthly_spend_outliers(messy_raw: pd.DataFrame, messy_df: pd.DataFrame) -> None:
    values = messy_raw["monthly_spend"].str.replace("$", "", regex=False).astype(float)
    q1, q3 = values.quantile(0.25), values.quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    expected = int(((values < lower) | (values > upper)).sum())

    col = P.profile(messy_df).column("monthly_spend")
    assert col.n_outliers == expected
    assert col.outlier_pct == pytest.approx(100 * expected / len(values))
    assert col.outlier_bounds == pytest.approx((lower, upper))


def test_messy_state(messy_df: pd.DataFrame) -> None:
    col = P.profile(messy_df).column("state")
    assert col.missing_pct == pytest.approx(100 * 49 / 300, abs=0.01)
    assert col.n_unique == 5  # case-sensitive: nsw and NSW both count
    assert col.case_variants == {"nsw": ["NSW", "nsw"]}
    assert col.top_values["nsw"] == 51
    assert col.top_values["NSW"] == 49


def test_messy_age(messy_df: pd.DataFrame) -> None:
    col = P.profile(messy_df).column("age")
    assert col.missing_pct == pytest.approx(100 * 46 / 300, abs=0.01)
    assert col.stats is not None
    assert col.stats["min"] == pytest.approx(18.0)
    assert col.stats["max"] == pytest.approx(74.0)
    assert col.stats["median"] == pytest.approx(46.0)


PRESENCE_REASON = "present vs missing is the signal"


def has_presence_reason(col: Any) -> bool:
    return any(PRESENCE_REASON in r for r in col.reasons)


def test_messy_notes_presence_signal(messy_df: pd.DataFrame) -> None:
    # One distinct value ("called support") but 62.67% missing -> categorical, not constant.
    col = P.profile(messy_df).column("notes")
    assert col.missing_pct == pytest.approx(100 * 188 / 300, abs=0.01)
    assert col.n_unique == 1
    assert col.dtype == "categorical"
    assert has_presence_reason(col)


def test_messy_country_constant(messy_df: pd.DataFrame) -> None:
    col = P.profile(messy_df).column("country")
    assert col.n_unique == 1
    assert col.missing_pct == 0.0
    assert col.all_missing is False
    assert not has_presence_reason(col)


def test_messy_plan_and_churned_counts(messy_df: pd.DataFrame) -> None:
    prof = P.profile(messy_df)
    assert prof.column("plan").top_values == {"pro": 109, "basic": 97, "premium": 94}
    assert prof.column("churned").top_values == {"no": 198, "yes": 102}


def test_profile_does_not_mutate_input(messy_df: pd.DataFrame) -> None:
    before = messy_df.copy(deep=True)
    P.profile(messy_df)
    pd.testing.assert_frame_equal(messy_df, before)


# ---------------------------------------------------------------- every sample CSV


def _iris(t: dict[str, str]) -> None:
    assert sum(v == "numeric" for v in t.values()) == 4
    assert t["species"] == "categorical"


def _breast_cancer(t: dict[str, str]) -> None:
    assert sum(v == "numeric" for v in t.values()) == 30
    assert t["diagnosis"] == "boolean"


def _diabetes(t: dict[str, str]) -> None:
    # sex holds 1.0 / 2.0: two values but not a boolean pair, so numeric.
    assert all(v == "numeric" for v in t.values()), t


def _messy(t: dict[str, str]) -> None:
    assert t == MESSY_TYPES


SAMPLE_EXPECTATIONS: dict[str, Callable[[dict[str, str]], None]] = {
    "iris.csv": _iris,
    "breast_cancer.csv": _breast_cancer,
    "diabetes.csv": _diabetes,
    "messy.csv": _messy,
}


@pytest.mark.parametrize("reader", ["pd.read_csv", "read_csv_raw"])
def test_every_sample_csv(sample_name: str, reader: str) -> None:
    path = DATA / sample_name
    df = pd.read_csv(path) if reader == "pd.read_csv" else P.read_csv_raw(path)
    prof = P.profile(df)
    assert_profile_invariants(prof, df)
    types = {c.name: c.dtype for c in prof.columns}
    SAMPLE_EXPECTATIONS[sample_name](types)


def test_sample_df_fixture_profiles(sample_df: pd.DataFrame) -> None:
    assert_profile_invariants(P.profile(sample_df), sample_df)


def test_iris_has_one_duplicate_row() -> None:
    assert P.profile(pd.read_csv(DATA / "iris.csv")).n_duplicate_rows == 1


# ---------------------------------------------------------------- parse_numeric


@pytest.mark.parametrize(
    ("raw", "expected", "markers"),
    [
        ("$1,200", 1200.0, {"currency", "thousands_sep"}),
        ("€3.5", 3.5, {"currency"}),
        ("£10", 10.0, {"currency"}),
        ("¥500", 500.0, {"currency"}),
        ("12%", 12.0, {"percent"}),
        ("(45)", -45.0, {"parens_negative"}),
        (" 7 ", 7.0, set()),
        ("-3.25", -3.25, set()),
    ],
)
def test_parse_numeric_values(raw: str, expected: float, markers: set[str]) -> None:
    parsed, found = P.parse_numeric(pd.Series([raw]))
    assert parsed.iloc[0] == pytest.approx(expected)
    assert set(found) == markers


def test_parse_numeric_unparseable_is_nan() -> None:
    parsed, found = P.parse_numeric(pd.Series(["abc"]))
    assert np.isnan(parsed.iloc[0])
    assert found == []


def test_parse_numeric_series_keeps_index_and_dtype() -> None:
    s = pd.Series(["$1,000", "2", "oops", "(3)"], index=[10, 11, 12, 13])
    parsed, found = P.parse_numeric(s)
    assert list(parsed.index) == [10, 11, 12, 13]
    assert parsed.dtype.kind == "f"
    assert parsed.tolist()[:2] == [1000.0, 2.0]
    assert np.isnan(parsed.iloc[2])
    assert parsed.iloc[3] == -3.0
    assert set(found) == {"currency", "thousands_sep", "parens_negative"}


# ---------------------------------------------------------------- parse_dates


@pytest.mark.parametrize(
    ("values", "expected", "dayfirst", "ambiguous"),
    [
        (["05/03/2024"], ["2024-03-05"], True, True),
        (["31/12/2024"], ["2024-12-31"], True, False),
        (["12/31/2024"], ["2024-12-31"], False, False),
        (["31-12-2024"], ["2024-12-31"], True, False),
        (["31.12.2024"], ["2024-12-31"], True, False),
        (
            ["2024-01-15", "31/12/2023", "05/03/2024"],
            ["2024-01-15", "2023-12-31", "2024-03-05"],
            True,
            False,
        ),
    ],
)
def test_parse_dates(
    values: list[str], expected: list[str], dayfirst: bool, ambiguous: bool
) -> None:
    parsed, got_dayfirst, got_ambiguous = P.parse_dates(pd.Series(values))
    assert list(parsed) == [pd.Timestamp(v) for v in expected]
    assert got_dayfirst is dayfirst
    assert got_ambiguous is ambiguous


def test_parse_dates_iso_only() -> None:
    parsed, _, ambiguous = P.parse_dates(pd.Series(["2024-03-05", "2023-12-31"]))
    assert list(parsed) == [pd.Timestamp("2024-03-05"), pd.Timestamp("2023-12-31")]
    assert ambiguous is False


def test_parse_dates_unparseable_is_nat() -> None:
    parsed, _, _ = P.parse_dates(pd.Series(["2024-03-05", "hello"]))
    assert parsed.iloc[0] == pd.Timestamp("2024-03-05")
    assert pd.isna(parsed.iloc[1])
    assert pd.api.types.is_datetime64_any_dtype(parsed)


# ---------------------------------------------------------------- edge cases


def test_empty_dataframe() -> None:
    prof = P.profile(pd.DataFrame())
    assert (prof.n_rows, prof.n_cols, prof.n_duplicate_rows) == (0, 0, 0)
    assert prof.columns == []


def test_zero_rows_with_columns() -> None:
    df = pd.DataFrame({"a": pd.Series([], dtype=float), "b": pd.Series([], dtype=object)})
    prof = P.profile(df)
    assert (prof.n_rows, prof.n_cols) == (0, 2)
    for col in prof.columns:
        assert col.dtype == "constant"
        assert col.all_missing is True
        assert col.n_unique == 0


def test_single_row() -> None:
    df = pd.DataFrame({"a": [1.5], "b": ["x"]})
    prof = P.profile(df)
    assert prof.n_rows == 1
    assert_profile_invariants(prof, df)
    assert all(c.dtype == "constant" for c in prof.columns)
    assert all(c.all_missing is False for c in prof.columns)


def test_single_column() -> None:
    df = pd.DataFrame({"only": ["a", "b", "c", "a", "b", "c"]})
    prof = P.profile(df)
    assert prof.n_cols == 1
    assert prof.column("only").dtype == "categorical"


def test_all_missing_column() -> None:
    df = pd.DataFrame({"x": [1, 2, 3, 4], "empty": [None, np.nan, None, np.nan]})
    col = P.profile(df).column("empty")
    assert col.dtype == "constant"
    assert col.all_missing is True
    assert col.missing_pct == pytest.approx(100.0)
    assert col.n_unique == 0
    assert col.sample_values == []


@pytest.mark.parametrize(
    ("n_missing", "expected"),
    [
        (0, "constant"),
        (4, "constant"),  # 4% missing
        (5, "categorical"),  # 5% boundary -> presence is the signal
        (60, "categorical"),
    ],
)
def test_single_value_with_missingness(n_missing: int, expected: str) -> None:
    values: list[Any] = ["gold"] * (100 - n_missing) + [None] * n_missing
    col = single_dtype(values)
    assert col.missing_pct == pytest.approx(float(n_missing))
    assert col.dtype == expected
    assert col.all_missing is False
    assert has_presence_reason(col) is (expected == "categorical")


def test_single_value_with_null_tokens_counts_as_missing() -> None:
    col = single_dtype(["gold"] * 40 + ["n/a"] * 60)
    assert col.dtype == "categorical"
    assert has_presence_reason(col)


def test_null_tokens_only_column() -> None:
    tokens = ["NA", "n/a", "NaN", "null", "None", "-", "?", "", "   ", "N/A", "NULL"]
    col = single_dtype(tokens)
    assert col.dtype == "constant"
    assert col.all_missing is True
    assert col.missing_pct == pytest.approx(100.0)


def test_null_tokens_count_as_missing() -> None:
    col = single_dtype(["1", "2", "na", "3", "?", "4", "5", "6", "7", "1"])
    assert col.dtype == "numeric"
    assert col.missing_pct == pytest.approx(20.0)


def test_unicode_headers() -> None:
    df = pd.DataFrame(
        {
            "prix €": ["€1", "€2", "€3", "€2", "€1", "€4"],
            "名前": ["太郎", "花子", "次郎", "太郎", "花子", "次郎"],
        }
    )
    prof = P.profile(df)
    assert_profile_invariants(prof, df)
    assert prof.column("prix €").dtype == "numeric"
    assert "currency" in prof.column("prix €").disguise
    assert prof.column("名前").dtype == "categorical"


def test_duplicate_headers() -> None:
    df = pd.DataFrame(
        [[1.5, "a"], [2.5, "b"], [1.5, "c"], [3.5, "a"], [2.5, "b"], [1.5, "c"]],
        columns=["x", "x"],
    )
    prof = P.profile(df)
    assert prof.n_cols == 2
    assert [c.name for c in prof.columns] == ["x", "x"]
    assert [c.dtype for c in prof.columns] == ["numeric", "categorical"]


@pytest.mark.parametrize(
    "values",
    [
        ["Y", "n", "Y", "n", "y"],
        ["True", "false", "True", "false"],
        ["1", "0", "1", "0", "1"],
        ["yes", "no", "no"],
        ["t", "f", "t"],
        ["on", "off", "ON"],
        [True, False, True, True],
        [1, 0, 1, 1, 0],
    ],
    ids=["Y/n", "True/false", "1/0 str", "yes/no", "t/f", "on/off", "bool dtype", "0/1 int"],
)
def test_boolean_variants(values: list[Any]) -> None:
    assert single_dtype(values).dtype == "boolean"


def test_boolean_case_variants() -> None:
    col = single_dtype(["Y", "n", "Y", "n", "y"])
    assert col.case_variants == {"y": ["Y", "y"]}


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        (["1.0", "2.0", "1.0", "2.0"], "numeric"),  # two values, not a boolean pair
        (["a", "b", "a", "b"], "categorical"),
        (["yes", "maybe", "no", "yes"], "categorical"),
    ],
)
def test_two_or_three_values_not_boolean(values: list[str], expected: str) -> None:
    assert single_dtype(values).dtype == expected


def test_year_only_column_is_numeric_not_datetime() -> None:
    years = [str(y) for y in [2019, 2020, 2021, 2022, 2023, 2024]] * 5
    col = single_dtype(years, "year")
    assert col.dtype == "numeric"
    assert col.date_format is None


def test_free_text_column() -> None:
    sentences = [
        "the delivery arrived two days late again",
        "great service and very friendly support staff",
        "I would like to cancel my plan next month",
        "app keeps crashing when I open the billing page",
        "price went up without any warning at all",
        "happy with the product but shipping is slow",
        "please call me back about my last invoice",
        "could not log in for most of the weekend",
        "the new dashboard is much easier to use now",
        "support solved my problem in under ten minutes",
    ] * 2  # repeated so the column is not unique enough to look like an id
    col = single_dtype(sentences, "feedback")
    assert col.dtype == "text"


def test_high_cardinality_short_strings_are_text() -> None:
    prefixes = ["North", "South", "East", "West", "New", "Old", "Port", "Lake", "Mount", "Fort"]
    prefixes += ["Glen", "Bay", "Saint", "Upper", "Lower"]
    suffixes = ["field", "ton", "ville", "burg", "ford", "haven", "wood", "dale", "port", "view"]
    cities = [p + s for p in prefixes for s in suffixes]  # 150 unique
    values = cities + cities[:50]  # 200 rows, 75% unique
    col = single_dtype(values, "city")
    assert col.n_unique == 150
    assert col.dtype == "text"


def test_sequential_ints_are_id() -> None:
    assert single_dtype(list(range(1, 101)), "row").dtype == "id"


def test_sequential_int_strings_are_id() -> None:
    assert single_dtype([str(i) for i in range(1, 101)], "row").dtype == "id"


def test_high_unique_floats_stay_numeric() -> None:
    rng = np.random.default_rng(0)
    assert single_dtype(rng.normal(50, 10, 200).tolist(), "score").dtype == "numeric"


@pytest.mark.parametrize(
    ("values", "expected", "parse_rate"),
    [
        (["1", "2", "3"] * 8 + ["oops"], "numeric", 24 / 25),  # 96% parse -> numeric
        (["10", "20", "30", "abc"] * 5, "categorical", 1.0),  # 75% parse -> not numeric
    ],
)
def test_mixed_types(values: list[str], expected: str, parse_rate: float) -> None:
    col = single_dtype(values)
    assert col.dtype == expected
    assert col.parse_rate == pytest.approx(parse_rate)


def test_n_duplicate_rows() -> None:
    df = pd.DataFrame({"a": [1, 1, 1, 2, 3], "b": ["x", "x", "x", "y", "z"]})
    assert P.profile(df).n_duplicate_rows == 2


def test_column_lookup_raises_keyerror() -> None:
    prof = P.profile(pd.DataFrame({"a": [1, 2, 3]}))
    assert prof.column("a").name == "a"
    with pytest.raises(KeyError):
        prof.column("missing")


def test_sample_values_capped_at_five() -> None:
    col = single_dtype([f"v{i % 20}" for i in range(100)])
    assert 0 < len(col.sample_values) <= 5
