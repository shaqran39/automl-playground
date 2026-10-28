"""The speedups: distinct-value parsing, sample gates and parse reuse. Same results, less work."""

import dataclasses
import time

import numpy as np
import pandas as pd
import pytest

from app.core import cleaning as C
from app.core import profiling as P
from tests.conftest import SAMPLE_NAMES
from tests.golden import DATA, REFERENCE_DATE


def _old_clean_strings(values: pd.Series) -> pd.Series:
    """The original row-by-row to_clean_strings, kept here as the reference."""
    out = []
    for v in values:
        if v is None or P._is_missing_scalar(v):
            out.append(None)
            continue
        text = str(v).strip()
        out.append(None if text.lower() in P.MISSING_TOKENS else text)
    return pd.Series(out, index=values.index, dtype=object)


def _counting(monkeypatch: pytest.MonkeyPatch, name: str) -> list[int]:
    calls: list[int] = []
    real = getattr(P, name)

    def counted(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(P, name, counted)
    return calls


# ---------------------------------------------------------------- distinct values


NASTY = pd.Series(
    [None, np.nan, pd.NA, pd.NaT, " x ", "N/A", "x", 1.5, 1, True, "", "  ", "None", 0.0, -0.0],
    dtype=object,
)


def test_to_clean_strings_matches_row_by_row() -> None:
    assert P.to_clean_strings(NASTY).tolist() == _old_clean_strings(NASTY).tolist()


@pytest.mark.parametrize(
    "values",
    [
        pd.Series([1.0, -0.0, 0.0, np.nan, 2.5, 1e20, 1.0]),
        pd.Series([3, 1, 3, 2], dtype="int64"),
        pd.Series(pd.to_datetime(["2024-01-01", None, "2024-01-01 10:30"], format="ISO8601")),
        pd.Series([True, False, True]),
        pd.Series([" a", "a", "A ", None, "na", "b"], dtype="str"),
    ],
)
def test_to_clean_strings_matches_row_by_row_for_every_dtype(values: pd.Series) -> None:
    assert P.to_clean_strings(values).tolist() == _old_clean_strings(values).tolist()


def test_negative_zero_stays_its_own_value() -> None:
    d = P.distinct(pd.Series([0.0, -0.0, 0.0]))
    assert d.uniques.tolist() == ["0.0", "-0.0"]
    assert d.counts.tolist() == [2, 1]


def test_mixed_objects_are_not_merged() -> None:
    """1, 1.0 and True hash the same, but they print differently."""
    d = P.distinct(pd.Series([1, 1.0, True], dtype=object))
    assert d.uniques.tolist() == ["1", "1.0", "True"]


def test_distinct_keeps_first_seen_order_and_merges_after_trimming() -> None:
    d = P.distinct(pd.Series(["b ", "a", "b", " a", "c"], dtype="str"))
    assert d.uniques.tolist() == ["b", "a", "c"]
    assert d.counts.tolist() == [2, 2, 1]


def test_top_values_ties_match_value_counts() -> None:
    values = pd.Series([f"v{i % 13}" for i in range(130)] + ["v5", "v7"], dtype="str")
    prof = P.profile_column("c", values)
    expected = values.value_counts().head(P.MAX_TOP_VALUES)
    assert list(prof.top_values) == [str(k) for k in expected.index]


def test_parse_numeric_matches_row_level() -> None:
    values = pd.Series(["$1,200", "(5)", "12%", "7", " 7 ", None, "abc", "1e3", "7"], dtype=object)
    fast, found_fast = P.parse_numeric(values)
    slow, found_slow = P._parse_numeric_rows(values)
    pd.testing.assert_series_equal(fast, slow, check_names=False)
    assert found_fast == found_slow


def test_day_month_order_is_decided_by_rows_not_distinct_values() -> None:
    """One day-first date seen 5 times beats two month-first dates seen once each."""
    values = pd.Series(["13/01/2020"] * 5 + ["01/13/2020", "02/14/2020"], dtype="str")
    _, dayfirst, ambiguous = P.parse_dates(values)
    assert dayfirst is True
    assert ambiguous is True


# ---------------------------------------------------------------- sample gates


def test_free_text_never_tries_the_date_or_time_parser(monkeypatch: pytest.MonkeyPatch) -> None:
    dates = _counting(monkeypatch, "_parse_dates_detail")
    times = _counting(monkeypatch, "_parse_time")
    words = pd.Series([f"cosy flat number {i} near the park" for i in range(20_000)], dtype="str")
    assert P.profile_column("name", words).dtype == "text"
    assert dates == [] and times == []


def test_rare_junk_still_counts_on_the_full_column(monkeypatch: pytest.MonkeyPatch) -> None:
    """Types come from a sample, but parse_rate and outliers use every row."""
    monkeypatch.setattr(P, "TYPE_SAMPLE_ROWS", 100)
    values = [str(i % 500) for i in range(20_000)]
    values[123] = "oops"
    col = P.profile_column("n", pd.Series(values, dtype="str"))
    assert col.dtype == "numeric"
    assert col.parse_rate == pytest.approx(19_999 / 20_000)


def test_rare_placeholder_codes_are_still_found(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(P, "TYPE_SAMPLE_ROWS", 100)
    ages = [str(20 + i % 60) for i in range(20_000)]
    ages[50], ages[19_000] = "999", "999"
    df = pd.DataFrame({"visit": [f"v{i}" for i in range(20_000)], "age": ages}, dtype="str")
    _, log = C.clean(df, P.profile(df), reference_date=REFERENCE_DATE)
    assert log.step("placeholder:age").rows_affected == 2


# ---------------------------------------------------------------- parse reuse


def test_clean_reuses_the_profile_parses(monkeypatch: pytest.MonkeyPatch) -> None:
    raw = P.read_csv_raw(DATA / "messy_clinic.csv")
    prof = P.profile(raw)
    numbers = _counting(monkeypatch, "_parse_numeric_detail")
    dates = _counting(monkeypatch, "_parse_dates_detail")
    C.clean(raw, prof, reference_date=REFERENCE_DATE)
    assert numbers == [] and dates == []


@pytest.mark.parametrize("name", SAMPLE_NAMES)
def test_clean_without_the_cache_gives_the_same_result(name: str) -> None:
    raw = P.read_csv_raw(DATA / name)
    prof = P.profile(raw)
    bare = dataclasses.replace(prof)  # same profile, no parsed columns attached
    assert P.parsed_columns(bare) is None
    a_df, a_log = C.clean(raw, prof, reference_date=REFERENCE_DATE)
    b_df, b_log = C.clean(raw, bare, reference_date=REFERENCE_DATE)
    pd.testing.assert_frame_equal(a_df, b_df)
    assert a_log.to_frame().equals(b_log.to_frame())


def test_a_changed_frame_is_parsed_again() -> None:
    raw = P.read_csv_raw(DATA / "iris.csv")
    prof = P.profile(raw)
    changed = raw.copy()
    changed.iloc[0, 0] = "$5.1"
    _, log = C.clean(changed, prof)  # the old parse would miss the $
    assert "to_numeric:" + changed.columns[0] in {s.step_id for s in log.steps}


@pytest.mark.parametrize("name", SAMPLE_NAMES)
def test_profile_reuse_gives_the_same_profile(name: str) -> None:
    raw = P.read_csv_raw(DATA / name)
    prof = P.profile(raw)
    clean_df, _ = C.clean(raw, prof, reference_date=REFERENCE_DATE)
    fresh = P.profile(clean_df)
    reused = P.profile(clean_df, reuse=prof)
    assert dataclasses.asdict(reused) == dataclasses.asdict(fresh)


def test_profile_reuse_copies_so_edits_dont_leak() -> None:
    raw = P.read_csv_raw(DATA / "iris.csv")
    prof = P.profile(raw)
    again = P.profile(raw, reuse=prof)
    again.columns[0].reasons.append("edited")
    assert "edited" not in prof.columns[0].reasons


def test_duplicate_groups_with_codes_match_plain_groupby() -> None:
    raw = P.read_csv_raw(DATA / "messy.csv")
    raw = pd.concat([raw, raw.iloc[:20]], ignore_index=True)
    prof = P.profile(raw)
    ctx = C._Parsed(raw, prof)
    fast = C.duplicate_groups(raw, ctx.row_codes())
    slow = C.duplicate_groups(raw)
    # same partition of rows, whatever the numbering
    assert pd.crosstab(fast, slow).astype(bool).sum(axis=1).eq(1).all()
    assert fast.nunique() == slow.nunique()


def test_progress_is_reported_per_column() -> None:
    raw = P.read_csv_raw(DATA / "iris.csv")
    seen: list[tuple[int, int, str]] = []
    P.profile(raw, progress=lambda i, n, name: seen.append((i, n, name)))
    assert [s[2] for s in seen] == list(raw.columns)
    assert all(n == raw.shape[1] for _, n, _ in seen)


# ---------------------------------------------------------------- speed


def test_big_frame_is_fast() -> None:
    """60k rows x 8 columns took ~15 s with row-by-row loops. Generous bound for slow CI."""
    n = 60_000
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "id": [f"R{i:06d}" for i in range(n)],
            "price": [f"${v:,.2f}" for v in rng.uniform(5, 5000, n)],
            "score": [f"{v:.6f}" for v in rng.random(n)],
            "city": rng.choice(["Oslo", "oslo ", "Rome", "Lima", ""], n),
            "when": [f"{1 + i % 28}/{1 + i % 12}/2021" for i in range(n)],
            "clock": [f"{i % 24}:{i % 60:02d}" for i in range(n)],
            "note": [f"lovely place number {i} with a view" for i in range(n)],
            "flag": rng.choice(["yes", "no"], n),
        },
        dtype="str",
    )
    start = time.perf_counter()
    prof = P.profile(df)
    clean_df, _ = C.clean(df, prof, reference_date=REFERENCE_DATE)
    P.profile(clean_df, reuse=prof)
    assert time.perf_counter() - start < 10
    assert [c.dtype for c in prof.columns] == [
        "id", "numeric", "numeric", "categorical", "datetime", "time", "text", "boolean"
    ]  # fmt: skip
