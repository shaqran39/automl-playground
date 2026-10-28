"""Edge cases found while building the Phase 2 rework (not pinned by the spec tests).

Covers switched-off conversions next to the new value rules, frames loaded with plain
pd.read_csv, an all-blank train column with the KNN imputer, real datetime64 input
and one ugly inline CSV.
"""

import io
import warnings

import numpy as np
import pandas as pd
import pytest

from app.core import cleaning as C
from app.core import preprocessing as PP
from app.core import profiling as P
from tests.conftest import DATA, REFERENCE_DATE


def run(df: pd.DataFrame, disabled: set[str] | None = None) -> tuple[pd.DataFrame, C.CleaningLog]:
    return C.clean(df, P.profile(df), disabled, reference_date=REFERENCE_DATE)


def ids(log: C.CleaningLog) -> list[str]:
    return [s.step_id for s in log.steps]


def test_to_numeric_off_keeps_raw_text_even_with_value_rules_on() -> None:
    values = ["$5", "$7", *[str(i) for i in range(10, 36)], "999", "999", "-3"]
    df = pd.DataFrame({"price": values, "r": [f"{i}.5" for i in range(len(values))]})
    clean_df, log = run(df, {"to_numeric:price"})
    assert {"placeholder:price", "impossible:price"} <= set(ids(log))
    assert clean_df["price"].tolist() == values  # nothing blanked inside the text


def test_placeholder_needs_real_values_to_compare_with() -> None:
    df = pd.DataFrame({"score": ["999", "-1"] * 10, "r": [f"{i}.5" for i in range(20)]})
    clean_df, log = run(df)
    assert "placeholder:score" not in ids(log)
    assert clean_df["score"].notna().all()


def test_numeric_categories_from_plain_read_csv_become_labels() -> None:
    raw = pd.read_csv(DATA / "diabetes.csv")
    assert raw["sex"].dtype == np.float64
    clean_df, log = C.clean(raw, P.profile(raw))
    assert set(clean_df["sex"].unique()) <= {"1", "2"}
    assert not [s for s in log.steps if s.column == "sex"]  # silent, like any plain parse
    pre = PP.build_preprocessor(P.profile(clean_df), target="disease_progression", df=clean_df)
    out = pre.fit_transform(clean_df)
    assert np.isfinite(out).all()


@pytest.mark.parametrize("imputer", ["auto", "knn", "iterative"])
def test_all_blank_train_column_keeps_names_lined_up(imputer: str) -> None:
    X = pd.DataFrame({"a": [np.nan] * 12, "b": np.arange(12.0)})
    prof = P.DatasetProfile(
        n_rows=12,
        n_cols=2,
        columns=[
            P.ColumnProfile(name="a", dtype="numeric", missing_pct=100.0, n_unique=0),
            P.ColumnProfile(name="b", dtype="numeric", missing_pct=0.0, n_unique=12),
        ],
    )
    opts = PP.PreprocessOptions(numeric_imputer=imputer)  # type: ignore[arg-type]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pre = PP.build_preprocessor(prof, df=X, options=opts).fit(X)
        out = pre.transform(X)
    assert out.shape[1] == len(pre.get_feature_names_out())
    assert list(pre.get_feature_names_out()) == ["a", "b", "a_was_missing"]


def test_invalid_dates_on_real_datetime64_column() -> None:
    when = pd.Series(pd.date_range("2023-01-01", periods=29, freq="7D"))
    when = pd.concat([when, pd.Series(pd.to_datetime(["2099-05-01"]))], ignore_index=True)
    df = pd.DataFrame({"visit": when, "r": [f"{i}.5" for i in range(30)]})
    clean_df, log = run(df)
    assert log.step("invalid_dates:visit").rows_affected == 1
    assert pd.isna(clean_df["visit"].iloc[-1])
    assert str(clean_df["visit"].dtype) == "datetime64[ns]"


def test_time_columns_get_the_high_missing_flag() -> None:
    times = ["14:30", "", "", "9:05", ""] * 4
    df = pd.DataFrame({"t": times, "r": [f"{i}.5" for i in range(20)]})
    _, log = run(df)
    assert log.step("high_missing:t").kind == "flag"


@pytest.mark.parametrize(
    ("name", "tokens"),
    [
        ("patientAge", ["patient", "age"]),
        ("HeartRate", ["heart", "rate"]),
        ("Age.At.Visit", ["age", "at", "visit"]),
        ("BMIValue", ["bmi", "value"]),
        ("usage", ["usage"]),
    ],
)
def test_name_tokens(name: str, tokens: list[str]) -> None:
    assert C.name_tokens(name) == tokens


UGLY_HEAD = """id,Age ,when,clock,qty,flag
1, 34 ,2024-01-05T09:00:00+10:00,9:05 am,3,Yes
2,-1,2024-01-06T09:00:00+10:00,  14:30 ,-1,no
3,41,N/A,2:30PM,4,YES
4,-1,2024-01-08T09:00:00+10:00,25:61,5,
5,29,9999-12-31,07:15:30,-1,No
"""
# plain rows so each bad value stays under the 95% parse bar and the 10% guard
UGLY_CSV = UGLY_HEAD + "".join(
    f"{i},{20 + i},2024-02-{i:02d}T09:00:00+10:00,{i % 12 + 1}:15 pm,{i},{('no', 'yes')[i % 2]}\n"
    for i in range(6, 25)
)


def test_ugly_inline_csv_end_to_end() -> None:
    raw = P.read_csv_raw(io.StringIO(UGLY_CSV))
    prof = P.profile(raw)
    assert prof.column("clock").dtype == "time"
    assert prof.column("when").dtype == "datetime"
    clean_df, log = C.clean(raw, prof, reference_date=REFERENCE_DATE)
    assert "id" not in clean_df.columns
    # -1 shows up twice in both columns and nothing else is negative, so it's a code
    assert log.step("placeholder:Age ").rows_affected == 2
    assert log.step("placeholder:qty").rows_affected == 2
    assert log.step("invalid_times:clock").rows_affected == 1
    assert clean_df["clock"].iloc[:3].tolist() == ["09:05:00", "14:30:00", "14:30:00"]
    # offsets go to UTC: 09:00 at +10:00 is 23:00 the day before
    assert clean_df["when"].iloc[0] == pd.Timestamp("2024-01-04 23:00:00")
    assert pd.isna(clean_df["when"].iloc[4])  # 9999-12-31 placeholder
    pre = PP.build_preprocessor(P.profile(clean_df), target="flag", df=clean_df)
    out = pre.fit_transform(clean_df)
    assert np.isfinite(out).all()
    assert "when_hour" in pre.get_feature_names_out()


# ---------------------------------------------------------------- review fixes


def with_copy(cols: dict[str, list[str]]) -> pd.DataFrame:
    """Unique rows (a unique "r" column) plus one exact copy of the last row at the end."""
    n = len(next(iter(cols.values())))
    df = pd.DataFrame({**cols, "r": [f"{i}.5" for i in range(n)]})
    return pd.concat([df, df.iloc[[-1]]], ignore_index=True)


def signature(log: C.CleaningLog) -> list[tuple[object, ...]]:
    return [(s.step_id, s.reason, s.kind, s.rows_affected) for s in log.steps]


def iso_days(n: int) -> list[str]:
    return [d.date().isoformat() for d in pd.date_range("2023-01-01", periods=n)]


@pytest.mark.parametrize("family", ["linear", "tree"])
def test_h1_numeric_column_that_profiles_as_categorical_fits(family: str) -> None:
    from sklearn.model_selection import train_test_split

    df = pd.DataFrame(
        {
            "quantity": ["1", "2"] * 20 + ["-1"] * 3,
            "x": [f"{i * 1.5}" for i in range(43)],
            "y": ["yes", "no", "no"] * 14 + ["yes"],
        }
    )
    clean_df, log = run(df)
    assert log.step("placeholder:quantity").rows_affected == 3
    assert clean_df["quantity"].dtype == np.float64  # still numbers, only 1.0 / 2.0 left
    X = clean_df.drop(columns="y")
    X_train, X_test = train_test_split(X, test_size=0.25, random_state=0)
    prof = P.profile(X_train)
    assert prof.column("quantity").dtype == "categorical"
    pre = PP.build_preprocessor(prof, df=X_train, family=family)  # type: ignore[arg-type]
    out = pre.fit(X_train).transform(X_test)
    assert np.isfinite(out).all()
    assert {"quantity_1", "quantity_2"} <= set(pre.get_feature_names_out())


def test_h1_to_labels_transformer() -> None:
    out = PP.ToLabels().fit_transform(pd.DataFrame({"a": [2.0, np.nan, 1.5, 3]}))
    assert (out[0, 0], out[2, 0], out[3, 0]) == ("2", "1.5", "3")
    assert pd.isna(out[1, 0])


DEDUPE_CASES: dict[str, tuple[dict[str, list[str]], str]] = {
    "placeholder:score": ({"score": [f"{10 + i}" for i in range(28)] + ["999", "999"]}, "score"),
    "impossible:age": ({"age": [str(20 + i) for i in range(28)] + ["-3", "150"]}, "age"),
    "invalid_dates:visit": ({"visit": [*iso_days(23), "2099-05-01"]}, "visit"),
    "invalid_times:t": ({"t": ["14:30", "9:05", "2:30 PM", "18:00"] * 10 + ["25:61"]}, "t"),
}


@pytest.mark.parametrize("step_id", list(DEDUPE_CASES))
def test_m1_dedupe_off_copies_get_the_same_fix(step_id: str) -> None:
    cols, col = DEDUPE_CASES[step_id]
    df = with_copy(cols)  # the copy repeats a row with a bad value
    full_df, full_log = run(df)
    clean_df, log = run(df, {"dedupe:*"})
    assert signature(log) == signature(full_log)  # log built as if every step is on
    assert len(clean_df) == len(df)
    assert log.step(step_id).enabled
    assert pd.isna(clean_df[col].iloc[-1]), "the duplicate copy kept its bad value"
    assert pd.isna(clean_df[col].iloc[-2])
    assert int(clean_df[col].isna().sum()) == int(full_df[col].isna().sum()) + 1


def test_m1_placeholder_off_still_keeps_999_in_every_copy() -> None:
    df = with_copy({"age": [str(20 + i) for i in range(27)] + ["999", "999", "999"]})
    clean_df, log = run(df, {"dedupe:*", "placeholder:age"})
    assert "impossible:age" not in ids(log)
    assert int((clean_df["age"] == 999).sum()) == 4


def test_m2_disabled_invalid_dates_brings_back_everything_it_claims() -> None:
    values = [*iso_days(38), "2099-05-01", "9999-12-31"]
    df = pd.DataFrame({"visit": values, "r": [f"{i}.5" for i in range(40)]})
    clean_df, log = run(df, {"invalid_dates:visit"})
    fix, flag = log.step("invalid_dates:visit"), log.step("unparseable_dates:visit")
    assert (fix.enabled, fix.rows_affected) == (False, 1)
    assert (flag.kind, flag.enabled, flag.rows_affected) == ("flag", True, 1)
    assert flag.reason == "1 value isn't a real date (like '9999-12-31'), so it's missing."
    assert clean_df["visit"].iloc[38] == pd.Timestamp("2099-05-01")  # really restored
    assert pd.isna(clean_df["visit"].iloc[39])  # can't be stored, as the flag says


def test_m2_unparseable_flag_plural_reason() -> None:
    df = pd.DataFrame({"visit": [*iso_days(48), "rubbish", "9999-12-31"]})
    _, log = run(df)
    reason = log.step("unparseable_dates:visit").reason
    assert reason == "2 values aren't real dates (like '9999-12-31'), so they're missing."


def test_m3_generated_names_dodge_real_columns() -> None:
    n = 30
    df = pd.DataFrame(
        {
            "visit": pd.date_range("2023-01-01", periods=n, freq="3D"),
            "visit_year": np.arange(n) * 1.5 + 0.25,  # not an ID
            "leave": pd.date_range("2023-02-01", periods=n, freq="3D"),
            "days_between_visit_leave": np.arange(n, dtype=float) * 2,
            "a": [np.nan if i % 4 == 0 else float(i) for i in range(n)],
            "a_was_missing": np.arange(n, dtype=float) % 3,
        }
    )
    pre = PP.build_preprocessor(P.profile(df), df=df).fit(df)
    names = list(pre.get_feature_names_out())
    assert len(names) == len(set(names)), "duplicate feature names"
    assert {"visit_year", "visit_year_2", "a_was_missing", "a_was_missing_2"} <= set(names)
    assert {"days_between_visit_leave", "days_between_visit_leave_2"} <= set(names)
    assert pre.transform(df).shape[1] == len(names)


def test_m3_names_param_must_match_feature_count() -> None:
    est = PP.DateFeatures(kinds=("time",), names=("only_one",)).fit(pd.DataFrame({"t": ["10:00"]}))
    with pytest.raises(ValueError):
        est.get_feature_names_out()


def test_l1_non_unique_index_is_rejected() -> None:
    df = pd.DataFrame({"a": ["1", "2", "3"]}, index=[0, 0, 1])
    with pytest.raises(ValueError, match="unique"):
        C.clean(df, P.profile(df))


def test_l6_default_reference_date_is_utc_today() -> None:
    assert C._reference_day(None) == pd.Timestamp.now(tz="UTC").normalize().tz_localize(None)


@pytest.mark.parametrize(
    ("text", "hit"), [("9999-12-31", True), ("31/12/9999", True), ("19999-01-01", False)]
)
def test_l6_year_9999_is_a_whole_token(text: str, hit: bool) -> None:
    assert bool(pd.Series([text]).str.contains(C._YEAR_9999_RE, regex=True).iloc[0]) is hit


def test_l7_shared_rule_labels_count_once() -> None:
    values = [str(20 + i) for i in range(28)] + ["-3", "-5"]
    _, log = run(pd.DataFrame({"age_count": values, "r": [f"{i}.5" for i in range(30)]}))
    step = log.step("impossible:age_count")  # matches both the age and the count rule
    assert step.rows_affected == 2
    assert step.reason.endswith(": 2 negative values.")


def test_l8_reason_grammar() -> None:
    _, log = run(
        pd.DataFrame({"bmi": [str(20 + i) for i in range(28)] + ["0", "0"], "r": iso_days(30)})
    )
    assert log.step("impossible:bmi").reason.endswith(": 2 zero values.")
    _, log = run(pd.DataFrame({"age": [str(20 + i) for i in range(29)] + ["150"]}))
    assert log.step("impossible:age").reason.endswith(": 1 value over 120.")
    _, log = run(pd.DataFrame({"visit": [*iso_days(29), "1970-01-01"]}))
    assert ": 1 placeholder date (" in log.step("invalid_dates:visit").reason


def test_l3_no_usable_features_has_its_own_error() -> None:
    prof = P.DatasetProfile(
        n_rows=3,
        n_cols=1,
        columns=[P.ColumnProfile(name="cid", dtype="id", missing_pct=0.0, n_unique=3)],
    )
    with pytest.raises(PP.NoUsableFeaturesError):
        PP.build_preprocessor(prof)
    assert issubclass(PP.NoUsableFeaturesError, ValueError)  # old callers still work
