"""Phase 2 rework: OutlierHandler, DateFeatures and the new build_preprocessor groups.

Written test-first against the approved rework spec, section C. Everything learned
from data (percentiles, fences, skew, fills, scaler means) must come from the rows
passed to fit, never from the test rows or the full-data profile.
"""

import copy
import math

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, KNNImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from app.core import cleaning as C
from app.core import preprocessing as PP
from app.core import profiling as P
from tests.conftest import REFERENCE_DATE

TAU = 2 * math.pi


# ---------------------------------------------------------------- helpers


def col(
    name: str, dtype: str, missing_pct: float = 0.0, n_unique: int = 5, has_time: bool = False
) -> P.ColumnProfile:
    # has_time is only passed when set, so module-level profiles still build today
    extra = {"has_time": True} if has_time else {}
    return P.ColumnProfile(
        name=name,
        dtype=dtype,  # type: ignore[arg-type]
        missing_pct=missing_pct,
        n_unique=n_unique,
        **extra,
    )


def dprof(*cols: P.ColumnProfile, n_rows: int = 40) -> P.DatasetProfile:
    return P.DatasetProfile(n_rows=n_rows, n_cols=len(cols), columns=list(cols))


def groups(pre: ColumnTransformer) -> dict[str, tuple[object, list[str]]]:
    return {name: (trans, list(cols)) for name, trans, cols in pre.transformers}


def arr(out: object) -> np.ndarray:
    return np.asarray(out, dtype=float)


def names(est: object) -> list[str]:
    return [str(n) for n in est.get_feature_names_out()]  # type: ignore[attr-defined]


def fences(values: np.ndarray) -> tuple[float, float]:
    q1, q3 = np.nanpercentile(values, [25, 75])
    return q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)


KEEP = [float(v) for v in range(1, 21)]  # no outliers
LOG = [1.0, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 7, 8, 9, 10, 200]  # right skew, min >= 0
CAP_NEG = [-5.0, *range(1, 19), 500.0]  # skewed but has a negative, so cap
CAP_SYM = [-100.0, *range(1, 19), 100.0]  # outliers both sides, no skew


def frame(**cols: list[float]) -> pd.DataFrame:
    n = max(len(v) for v in cols.values())
    return pd.DataFrame({k: v + [np.nan] * (n - len(v)) for k, v in cols.items()})


# ---------------------------------------------------------------- OutlierHandler: auto


@pytest.mark.parametrize(
    ("values", "action"),
    [(KEEP, "keep"), (LOG, "log"), (CAP_NEG, "cap"), (CAP_SYM, "cap")],
    ids=["keep", "log", "cap negative", "cap symmetric"],
)
def test_auto_resolution(values: list[float], action: str) -> None:
    handler = PP.OutlierHandler().fit(pd.DataFrame({"x": values}))
    assert list(handler.actions_) == [action]


def test_auto_resolution_per_column() -> None:
    X = frame(a=KEEP, b=LOG, c=CAP_NEG)
    handler = PP.OutlierHandler(choices=("auto", "auto", "auto")).fit(X)
    assert list(handler.actions_) == ["keep", "log", "cap"]


def test_default_choices_mean_auto() -> None:
    X = frame(a=KEEP, b=LOG)
    assert list(PP.OutlierHandler().fit(X).actions_) == ["keep", "log"]


def test_explicit_choices_win_over_auto() -> None:
    X = frame(a=LOG, b=LOG, c=LOG, d=LOG)
    handler = PP.OutlierHandler(choices=("keep", "cap", "log", "missing")).fit(X)
    assert list(handler.actions_) == ["keep", "cap", "log", "missing"]


# ---------------------------------------------------------------- OutlierHandler: transforms


def test_keep_leaves_values_alone() -> None:
    X = pd.DataFrame({"x": LOG})
    out = arr(PP.OutlierHandler(choices=("keep",)).fit(X).transform(X))
    np.testing.assert_array_equal(out[:, 0], LOG)


def test_cap_clips_to_train_p1_p99() -> None:
    train = pd.DataFrame({"x": CAP_SYM})
    lo, hi = np.nanpercentile(CAP_SYM, [1, 99])
    handler = PP.OutlierHandler(choices=("cap",)).fit(train)
    test = pd.DataFrame({"x": [-1e6, 5.0, 1e6]})
    out = arr(handler.transform(test))[:, 0]
    assert out[0] == pytest.approx(lo)
    assert out[1] == 5.0
    assert out[2] == pytest.approx(hi)


def test_log_is_log1p_of_non_negative() -> None:
    handler = PP.OutlierHandler(choices=("log",)).fit(pd.DataFrame({"x": LOG}))
    out = arr(handler.transform(pd.DataFrame({"x": [-2.0, 0.0, 1.0, math.e - 1]})))[:, 0]
    np.testing.assert_allclose(out, [0.0, 0.0, math.log(2), 1.0])


def test_missing_sets_values_outside_train_fences_to_nan() -> None:
    lo, hi = fences(np.array(LOG))
    handler = PP.OutlierHandler(choices=("missing",)).fit(pd.DataFrame({"x": LOG}))
    test = pd.DataFrame({"x": [lo - 1, 5.0, hi + 1, 200.0]})
    out = arr(handler.transform(test))[:, 0]
    assert np.isnan(out[0]) and np.isnan(out[2]) and np.isnan(out[3])
    assert out[1] == 5.0


@pytest.mark.parametrize("choice", ["keep", "cap", "log", "missing", "auto"])
def test_nan_stays_nan(choice: str) -> None:
    train = pd.DataFrame({"x": [*LOG, np.nan]})
    handler = PP.OutlierHandler(choices=(choice,)).fit(train)
    out = arr(handler.transform(pd.DataFrame({"x": [np.nan, 3.0, np.nan]})))[:, 0]
    assert np.isnan(out[0]) and np.isnan(out[2])
    assert not np.isnan(out[1])


def test_handler_ignores_nan_when_fitting() -> None:
    with_nan = PP.OutlierHandler(choices=("cap",)).fit(pd.DataFrame({"x": [*CAP_SYM, np.nan]}))
    without = PP.OutlierHandler(choices=("cap",)).fit(pd.DataFrame({"x": CAP_SYM}))
    test = pd.DataFrame({"x": [-1e6, 1e6]})
    np.testing.assert_allclose(arr(with_nan.transform(test)), arr(without.transform(test)))


def test_handler_feature_names_are_input_names() -> None:
    handler = PP.OutlierHandler().fit(frame(a=KEEP, b=LOG))
    assert names(handler) == ["a", "b"]


def test_handler_output_shape() -> None:
    X = frame(a=KEEP, b=LOG, c=CAP_NEG)
    assert arr(PP.OutlierHandler().fit(X).transform(X)).shape == X.shape


# ---------------------------------------------------------------- OutlierHandler: tree family


def test_tree_family_keeps_raw_values_except_missing() -> None:
    X = frame(a=LOG, b=LOG, c=LOG, d=LOG, e=LOG)
    handler = PP.OutlierHandler(choices=("auto", "cap", "log", "missing", "keep"), family="tree")
    handler.fit(X)
    assert list(handler.actions_) == ["keep", "keep", "keep", "missing", "keep"]
    out = arr(handler.transform(X))
    np.testing.assert_array_equal(out[:, :3], X.iloc[:, :3].to_numpy())
    assert np.isnan(out[-1, 3])  # 200 is outside the fences


def test_family_is_a_param() -> None:
    handler = PP.OutlierHandler(choices=("cap",), family="tree")
    params = handler.get_params()
    assert params["family"] == "tree"
    assert tuple(params["choices"]) == ("cap",)
    assert clone(handler).get_params()["family"] == "tree"


# ---------------------------------------------------------------- OutlierHandler: train only


def _fitted_attrs(est: object) -> dict[str, object]:
    return {k: copy.deepcopy(v) for k, v in vars(est).items() if k.endswith("_")}


def _is_data(v: object) -> bool:
    """Plain fitted data (arrays, numbers, strings, containers of them), not estimators."""
    if isinstance(v, (np.ndarray, np.generic, int, float, str, bool)):
        return True
    if isinstance(v, (list, tuple)):
        return all(_is_data(i) for i in v)
    if isinstance(v, dict):
        return all(_is_data(i) for i in v.values())
    return False


def _assert_same(a: dict[str, object], b: dict[str, object]) -> None:
    assert a.keys() == b.keys()
    for k in a:
        if not _is_data(a[k]):
            continue
        try:
            np.testing.assert_allclose(arr(a[k]), arr(b[k]), equal_nan=True, err_msg=k)
        except (TypeError, ValueError):
            assert repr(a[k]) == repr(b[k]), k


def test_handler_learns_from_train_only() -> None:
    train = frame(a=CAP_SYM, b=LOG)
    handler = PP.OutlierHandler(choices=("cap", "missing")).fit(train)
    before = _fitted_attrs(handler)
    crazy = pd.DataFrame({"a": [1e9, -1e9, 3.0], "b": [1e9, 0.0, np.nan]})
    handler.transform(crazy)
    _assert_same(before, _fitted_attrs(handler))
    out = arr(handler.transform(crazy))
    assert out[0, 0] == pytest.approx(np.nanpercentile(CAP_SYM, 99))


# ---------------------------------------------------------------- DateFeatures

DATE_SUFFIXES = [
    "year",
    "month",
    "quarter",
    "dayofweek",
    "is_weekend",
    "month_sin",
    "month_cos",
    "dayofweek_sin",
    "dayofweek_cos",
]
TIME_SUFFIXES = ["hour", "minute", "hour_sin", "hour_cos"]


def _features(X: pd.DataFrame, kinds: tuple[str, ...]) -> pd.DataFrame:
    est = PP.DateFeatures(kinds=kinds).fit(X)
    return pd.DataFrame(arr(est.transform(X)), columns=names(est), index=X.index)


def test_date_feature_names() -> None:
    X = pd.DataFrame({"d": pd.to_datetime(["2024-01-06", "2024-08-14"])})
    est = PP.DateFeatures(kinds=("date",)).fit(X)
    assert names(est) == [f"d_{s}" for s in DATE_SUFFIXES]


def test_datetime_feature_names() -> None:
    X = pd.DataFrame({"d": pd.to_datetime(["2024-01-06 10:00", "2024-08-14 18:30"])})
    est = PP.DateFeatures(kinds=("datetime",)).fit(X)
    assert names(est) == [f"d_{s}" for s in DATE_SUFFIXES + TIME_SUFFIXES]


def test_time_feature_names() -> None:
    X = pd.DataFrame({"t": ["06:15:00", "18:30:00"]})
    est = PP.DateFeatures(kinds=("time",)).fit(X)
    assert names(est) == [f"t_{s}" for s in TIME_SUFFIXES]


def test_date_feature_values() -> None:
    # 2024-01-06 is a Saturday, 2024-01-07 a Sunday, 2024-08-14 a Wednesday
    X = pd.DataFrame({"d": pd.to_datetime(["2024-01-06", "2024-01-07", "2024-08-14"])})
    f = _features(X, ("date",))
    assert f["d_year"].tolist() == [2024.0, 2024.0, 2024.0]
    assert f["d_month"].tolist() == [1.0, 1.0, 8.0]
    assert f["d_quarter"].tolist() == [1.0, 1.0, 3.0]
    assert f["d_dayofweek"].tolist() == [5.0, 6.0, 2.0]
    assert f["d_is_weekend"].tolist() == [1.0, 1.0, 0.0]
    np.testing.assert_allclose(f["d_month_sin"], [0.0, 0.0, math.sin(TAU * 7 / 12)], atol=1e-12)
    np.testing.assert_allclose(f["d_month_cos"], [1.0, 1.0, math.cos(TAU * 7 / 12)], atol=1e-12)
    np.testing.assert_allclose(
        f["d_dayofweek_sin"], [math.sin(TAU * d / 7) for d in (5, 6, 2)], atol=1e-12
    )
    np.testing.assert_allclose(
        f["d_dayofweek_cos"], [math.cos(TAU * d / 7) for d in (5, 6, 2)], atol=1e-12
    )


@pytest.mark.parametrize(
    ("day", "quarter"), [("2024-03-31", 1), ("2024-04-01", 2), ("2024-12-31", 4)]
)
def test_quarter(day: str, quarter: int) -> None:
    f = _features(pd.DataFrame({"d": pd.to_datetime([day])}), ("date",))
    assert f["d_quarter"].iloc[0] == quarter


def test_datetime_hour_values() -> None:
    X = pd.DataFrame({"d": pd.to_datetime(["2024-07-15 18:30:00", "2024-07-15 06:05:00"])})
    f = _features(X, ("datetime",))
    assert f["d_hour"].tolist() == [18.0, 6.0]
    assert f["d_minute"].tolist() == [30.0, 5.0]
    np.testing.assert_allclose(f["d_hour_sin"], [-1.0, 1.0], atol=1e-12)
    np.testing.assert_allclose(f["d_hour_cos"], [0.0, 0.0], atol=1e-12)


def test_time_values() -> None:
    X = pd.DataFrame({"t": ["06:15:00", "00:00:00", "12:45:30"]})
    f = _features(X, ("time",))
    assert f["t_hour"].tolist() == [6.0, 0.0, 12.0]
    assert f["t_minute"].tolist() == [15.0, 0.0, 45.0]
    np.testing.assert_allclose(f["t_hour_sin"], [1.0, 0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(f["t_hour_cos"], [0.0, 1.0, -1.0], atol=1e-12)


def test_days_between_two_dates() -> None:
    X = pd.DataFrame(
        {
            "a": pd.to_datetime(["2024-01-01", "2024-03-01"]),
            "b": pd.to_datetime(["2024-01-11 12:00", "2024-02-28"], format="ISO8601"),
        }
    )
    est = PP.DateFeatures(kinds=("date", "datetime")).fit(X)
    out_names = names(est)
    assert out_names[-1] == "days_between_a_b"
    assert out_names.count("days_between_a_b") == 1
    f = pd.DataFrame(arr(est.transform(X)), columns=out_names)
    np.testing.assert_allclose(f["days_between_a_b"], [10.5, -2.0])


def test_days_between_every_pair_in_order_and_skips_times() -> None:
    X = pd.DataFrame(
        {
            "a": pd.to_datetime(["2024-01-01"]),
            "t": ["10:00:00"],
            "b": pd.to_datetime(["2024-01-03"]),
            "c": pd.to_datetime(["2024-01-10"]),
        }
    )
    est = PP.DateFeatures(kinds=("date", "time", "date", "date")).fit(X)
    between = [n for n in names(est) if n.startswith("days_between_")]
    assert between == ["days_between_a_b", "days_between_a_c", "days_between_b_c"]
    expected = {
        *(f"a_{s}" for s in DATE_SUFFIXES),
        *(f"t_{s}" for s in TIME_SUFFIXES),
        *(f"b_{s}" for s in DATE_SUFFIXES),
        *(f"c_{s}" for s in DATE_SUFFIXES),
        *between,
    }
    assert set(names(est)) == expected
    f = pd.DataFrame(arr(est.transform(X)), columns=names(est))
    assert f[between].iloc[0].tolist() == [2.0, 9.0, 7.0]


def test_single_date_has_no_days_between() -> None:
    X = pd.DataFrame({"a": pd.to_datetime(["2024-01-01"]), "t": ["10:00:00"]})
    est = PP.DateFeatures(kinds=("date", "time")).fit(X)
    assert not [n for n in names(est) if n.startswith("days_between")]


def test_nan_in_nan_out() -> None:
    X = pd.DataFrame(
        {
            "a": pd.to_datetime(["2024-01-01", None, "2024-01-05"]),
            "b": pd.to_datetime(["2024-01-02", "2024-01-03", None]),
            "t": ["10:00:00", None, "11:30:00"],
        }
    )
    f = _features(X, ("date", "date", "time"))
    a_cols = [f"a_{s}" for s in DATE_SUFFIXES]
    assert f.loc[1, a_cols].isna().all()
    assert f.loc[0, a_cols].notna().all()
    assert f.loc[2, [f"b_{s}" for s in DATE_SUFFIXES]].isna().all()
    assert f.loc[1, [f"t_{s}" for s in TIME_SUFFIXES]].isna().all()
    assert np.isnan(f.loc[1, "days_between_a_b"]) and np.isnan(f.loc[2, "days_between_a_b"])
    assert f.loc[0, "days_between_a_b"] == 1.0


def test_strings_are_coerced_like_datetimes() -> None:
    as_dt = pd.DataFrame({"d": pd.to_datetime(["2024-01-06 10:00", "2024-08-14 18:30"])})
    as_str = pd.DataFrame({"d": ["2024-01-06 10:00:00", "2024-08-14 18:30:00"]})
    np.testing.assert_allclose(
        _features(as_dt, ("datetime",)).to_numpy(), _features(as_str, ("datetime",)).to_numpy()
    )


def test_unparseable_strings_become_nan() -> None:
    f = _features(pd.DataFrame({"d": ["2024-01-06", "rubbish"]}), ("date",))
    assert f.iloc[1].isna().all()
    assert f.iloc[0].notna().all()


def test_tz_aware_input_becomes_utc() -> None:
    sydney = pd.DataFrame({"d": pd.date_range("2024-01-01", periods=2, tz="Australia/Sydney")})
    f = _features(sydney, ("datetime",))
    assert f["d_year"].iloc[0] == 2023.0  # 13:00 on 31 Dec in UTC
    assert f["d_hour"].iloc[0] == 13.0
    offsets = pd.DataFrame({"d": ["2024-01-01T05:00:00+10:00"]})
    g = _features(offsets, ("datetime",))
    assert (g["d_year"].iloc[0], g["d_hour"].iloc[0]) == (2023.0, 19.0)


def test_date_features_are_stateless_and_float() -> None:
    a = pd.DataFrame({"d": pd.to_datetime(["2024-01-06"])})
    b = pd.DataFrame({"d": pd.to_datetime(["1999-12-31"])})
    est = PP.DateFeatures(kinds=("date",)).fit(a)
    out = np.asarray(est.transform(b))
    assert out.dtype.kind == "f"
    assert out[0, 0] == 1999.0
    assert clone(est).get_params()["kinds"] == ("date",)


# ---------------------------------------------------------------- build_preprocessor: options

NUM_PROFILE = dprof(
    col("num_a", "numeric", missing_pct=10.0, n_unique=30),
    col("num_b", "numeric", n_unique=40),
    col("c", "categorical", n_unique=3),
)


def num_frame(n: int = 40, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    a = rng.normal(50, 10, n)
    a[::10] = np.nan
    b = rng.exponential(5, n)
    b[3] = 500.0  # a real outlier
    return pd.DataFrame({"num_a": a, "num_b": b, "c": rng.choice(["x", "y", "z"], n)})


def test_preprocess_options_defaults() -> None:
    opts = PP.PreprocessOptions()
    assert opts.outliers == {}
    assert opts.numeric_imputer == "auto"
    assert PP.PreprocessOptions().outliers is not opts.outliers


def test_ct_settings() -> None:
    pre = PP.build_preprocessor(NUM_PROFILE)
    assert pre.verbose_feature_names_out is False
    assert pre.sparse_threshold == 0.0
    assert pre.remainder == "drop"


def test_excluded_types() -> None:
    assert frozenset(PP.EXCLUDED_TYPES) == frozenset({"id", "constant", "text"})
    assert PP.MISSING_FILL == "Missing"


def test_outlier_choices_line_up_with_numeric_columns() -> None:
    opts = PP.PreprocessOptions(outliers={"num_b": "cap"})
    pre = PP.build_preprocessor(NUM_PROFILE, options=opts)
    handler = groups(pre)["numeric"][0].named_steps["outliers"]  # type: ignore[attr-defined]
    assert groups(pre)["numeric"][1] == ["num_a", "num_b"]
    assert tuple(handler.choices) == ("auto", "cap")
    assert handler.family == "linear"


@pytest.mark.parametrize("family", ["linear", "tree"])
def test_family_reaches_the_outlier_handler(family: str) -> None:
    pre = PP.build_preprocessor(NUM_PROFILE, family=family)  # type: ignore[arg-type]
    handler = groups(pre)["numeric"][0].named_steps["outliers"]  # type: ignore[attr-defined]
    assert handler.family == family


def test_tree_family_fitted_actions() -> None:
    df = num_frame()
    pre = PP.build_preprocessor(NUM_PROFILE, df=df, family="tree")
    pre.fit(df)
    handler = pre.named_transformers_["numeric"].named_steps["outliers"]
    assert list(handler.actions_) == ["keep", "keep"]
    linear = PP.build_preprocessor(NUM_PROFILE, df=df).fit(df)
    lin_handler = linear.named_transformers_["numeric"].named_steps["outliers"]
    assert lin_handler.actions_[1] == "log"  # exponential plus a big value, min >= 0


@pytest.mark.parametrize(
    ("choice", "kind"),
    [("auto", PP.SkewAwareImputer), ("knn", KNNImputer), ("iterative", IterativeImputer)],
)
def test_numeric_imputer_options(choice: str, kind: type) -> None:
    opts = PP.PreprocessOptions(numeric_imputer=choice)  # type: ignore[arg-type]
    pre = PP.build_preprocessor(NUM_PROFILE, options=opts)
    imp = groups(pre)["numeric"][0].named_steps["impute"]  # type: ignore[attr-defined]
    assert isinstance(imp, kind)
    if choice == "iterative":
        assert imp.random_state == 0
    df = num_frame()
    train, test = df.iloc[:30], df.iloc[30:].copy()
    test.loc[test.index[:2], "num_b"] = np.nan
    out_train = arr(pre.fit_transform(train))
    out_test = arr(pre.transform(test))
    assert np.isfinite(out_train).all() and np.isfinite(out_test).all()
    assert out_train.shape[1] == out_test.shape[1]


def test_dates_group_kinds_follow_the_profile() -> None:
    prof = dprof(
        col("d", "datetime"),
        col("dt", "datetime", has_time=True),
        col("t", "time"),
        col("x", "numeric"),
    )
    g = groups(PP.build_preprocessor(prof))
    assert g["dates"][1] == ["d", "dt", "t"]
    features = g["dates"][0].named_steps["features"]  # type: ignore[attr-defined]
    assert tuple(features.kinds) == ("date", "datetime", "time")


# ---------------------------------------------------------------- was_missing


@pytest.mark.parametrize(
    ("dtype", "pct", "flagged"),
    [
        ("numeric", 5.0, False),
        ("numeric", 5.1, True),
        ("numeric", 0.0, False),
        ("boolean", 6.0, True),
        ("boolean", 4.0, False),
        ("datetime", 10.0, True),
        ("time", 10.0, True),
        ("categorical", 30.0, False),
    ],
)
def test_was_missing_only_above_five_percent(dtype: str, pct: float, flagged: bool) -> None:
    prof = dprof(col("c", dtype, missing_pct=pct, n_unique=3), col("x", "numeric"))
    g = groups(PP.build_preprocessor(prof))
    assert ("c" in g.get("was_missing", (None, []))[1]) is flagged


def test_was_missing_output() -> None:
    df = pd.DataFrame(
        {
            "n": [1.0, np.nan, 3.0, 4.0, np.nan, 6.0, 7.0, 8.0, 9.0, 10.0],
            "b": [1.0, 0.0, np.nan, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
            "d": pd.to_datetime(["2024-01-01", None] + [f"2024-02-{i:02d}" for i in range(1, 9)]),
            "t": ["10:00:00", "11:00:00", None] + ["12:00:00"] * 7,
        }
    )
    prof = dprof(
        col("n", "numeric", missing_pct=20.0, n_unique=8),
        col("b", "boolean", missing_pct=10.0, n_unique=2),
        col("d", "datetime", missing_pct=10.0, n_unique=9),
        col("t", "time", missing_pct=10.0, n_unique=3),
        n_rows=10,
    )
    pre = PP.build_preprocessor(prof, df=df).fit(df)
    out = pd.DataFrame(arr(pre.transform(df)), columns=names(pre))
    for c in ["n", "b", "d", "t"]:
        flag = f"{c}_was_missing"
        assert flag in out.columns
        assert out[flag].tolist() == df[c].isna().astype(float).tolist()
    assert np.isfinite(out.to_numpy()).all()


# ---------------------------------------------------------------- "Missing" category


def test_blank_categories_become_missing_category() -> None:
    df = pd.DataFrame({"c": ["a", "b", None, "a", np.nan, "b"], "x": [1.0, 2, 3, 4, 5, 6]})
    prof = dprof(col("c", "categorical", missing_pct=33.0, n_unique=2), col("x", "numeric"))
    pre = PP.build_preprocessor(prof, df=df).fit(df)
    out = pd.DataFrame(arr(pre.transform(df)), columns=names(pre))
    assert "c_Missing" in out.columns
    assert out["c_Missing"].tolist() == [0.0, 0.0, 1.0, 0.0, 1.0, 0.0]


def test_frequency_encoder_learns_missing_share() -> None:
    values = [f"v{i % 12}" for i in range(36)] + [None] * 4
    df = pd.DataFrame({"c": values})
    prof = dprof(col("c", "categorical", missing_pct=10.0, n_unique=12), n_rows=40)
    pre = PP.build_preprocessor(prof, df=df).fit(df)
    out = arr(pre.transform(df.iloc[[36]]))  # a blank row, as pandas stores it
    assert out[0, 0] == pytest.approx(4 / 40)


# ---------------------------------------------------------------- target never leaks in

TARGET_PROFILE = dprof(
    col("num_a", "numeric", missing_pct=10.0, n_unique=30),
    col("num_b", "numeric", n_unique=40),
    col("when", "datetime", missing_pct=10.0, n_unique=40),
    col("other_day", "datetime", n_unique=40),
    col("c", "categorical", n_unique=3),
)


def target_frame(n: int = 40) -> pd.DataFrame:
    df = num_frame(n)
    when = pd.Series(pd.date_range("2024-01-01", periods=n, freq="D"))
    when[::10] = pd.NaT
    df["when"] = when
    df["other_day"] = pd.date_range("2023-06-01", periods=n, freq="3D")
    return df


@pytest.mark.parametrize("target", ["when", "num_a", "num_b", "c"])
def test_target_never_in_feature_names(target: str) -> None:
    df = target_frame()
    pre = PP.build_preprocessor(TARGET_PROFILE, target=target, df=df)
    for _, _, cols in pre.transformers:
        assert target not in list(cols)
    pre.fit(df)
    out = names(pre)
    assert target not in out
    assert not [n for n in out if n.startswith(f"{target}_")], out
    assert not [n for n in out if n.startswith("days_between") and target in n], out


def test_date_target_leaves_other_dates_in() -> None:
    df = target_frame()
    pre = PP.build_preprocessor(TARGET_PROFILE, target="when", df=df).fit(df)
    out = names(pre)
    assert "other_day_year" in out
    assert not [n for n in out if n.startswith("days_between")]  # only one date left


# ---------------------------------------------------------------- leakage


def _stats(pre: ColumnTransformer) -> dict[str, object]:
    """Every fitted attribute (name ending in _) of every step, keyed group.step.attr."""
    found: dict[str, object] = {}
    for group, trans in pre.named_transformers_.items():
        if group == "remainder":
            continue
        steps = trans.steps if isinstance(trans, Pipeline) else [("self", trans)]
        for step_name, est in steps:
            for k, v in vars(est).items():
                if k.endswith("_") and not k.startswith("_"):
                    found[f"{group}.{step_name}.{k}"] = copy.deepcopy(v)
    return found


def leak_frame(n: int = 60, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    income = rng.lognormal(10, 0.6, n)
    income[5] = 900000.0
    age = rng.normal(45, 12, n)
    age[::9] = np.nan
    return pd.DataFrame(
        {
            "income": income,
            "age": age,
            "visit": pd.date_range("2023-01-01", periods=n, freq="5D"),
            "checkin": [f"{8 + i % 10:02d}:{(i * 7) % 60:02d}:00" for i in range(n)],
            "smoker": rng.integers(0, 2, n).astype(float),
            "city": rng.choice(["Sydney", "Perth", "Hobart"], n),
        }
    )


def _changed_test_rows(df: pd.DataFrame, test_idx: pd.Index) -> pd.DataFrame:
    out = df.copy()
    present = out.loc[test_idx, "age"].notna()
    out.loc[test_idx[present], "age"] = 500.0
    out.loc[test_idx, "income"] = out.loc[test_idx, "income"] * 1000
    out.loc[test_idx, "visit"] = out.loc[test_idx, "visit"] + pd.Timedelta(days=3650)
    out.loc[test_idx, "smoker"] = 1.0
    return out


@pytest.mark.parametrize("family", ["linear", "tree"])
@pytest.mark.parametrize("imputer", ["auto", "knn", "iterative"])
def test_changing_test_rows_changes_no_fitted_statistic(family: str, imputer: str) -> None:
    df = leak_frame()
    train_idx, test_idx = df.index[:45], df.index[45:]
    changed = _changed_test_rows(df, test_idx)
    opts = PP.PreprocessOptions(numeric_imputer=imputer)  # type: ignore[arg-type]

    def fitted(full: pd.DataFrame) -> ColumnTransformer:
        # the profile and df here see every row, like a careless caller would pass
        pre = PP.build_preprocessor(P.profile(full), df=full, family=family, options=opts)  # type: ignore[arg-type]
        return pre.fit(full.loc[train_idx])

    a, b = fitted(df), fitted(changed)
    _assert_same(_stats(a), _stats(b))
    np.testing.assert_allclose(
        arr(a.transform(df.loc[train_idx])), arr(b.transform(df.loc[train_idx]))
    )


def test_transform_does_not_refit() -> None:
    df = leak_frame()
    pre = PP.build_preprocessor(P.profile(df.iloc[:45]), df=df.iloc[:45]).fit(df.iloc[:45])
    before = _stats(pre)
    pre.transform(_changed_test_rows(df, df.index[45:]).iloc[45:])
    _assert_same(before, _stats(pre))


# ---------------------------------------------------------------- messy_clinic end to end


@pytest.fixture
def clinic_split(
    clinic_raw: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    clean_df, _ = C.clean(clinic_raw, P.profile(clinic_raw), reference_date=REFERENCE_DATE)
    X = clean_df.drop(columns="outcome")
    y = clean_df["outcome"].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=0, stratify=y
    )
    return X_train, X_test, y_train, y_test


def _expected_auto(values: pd.Series) -> str:
    v = values.dropna().to_numpy(dtype=float)
    lo, hi = fences(v)
    if not ((v < lo) | (v > hi)).any():
        return "keep"
    return "log" if pd.Series(v).skew() > 1 and v.min() >= 0 else "cap"


@pytest.mark.parametrize("family", ["linear", "tree"])
def test_clinic_end_to_end(
    clinic_split: tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series], family: str
) -> None:
    X_train, X_test, y_train, y_test = clinic_split
    prof = P.profile(X_train)
    pre = PP.build_preprocessor(prof, df=X_train, family=family)  # type: ignore[arg-type]
    g = groups(pre)
    assert g["dates"][1] == ["visit_date", "discharge_datetime", "checkin_time"]
    assert "sex" in g["onehot"][1]
    assert {"smoker"} <= set(g["boolean"][1])

    out_train = arr(pre.fit_transform(X_train))
    out_test = arr(pre.transform(X_test))
    assert np.isfinite(out_train).all() and np.isfinite(out_test).all()
    assert out_train.shape == (len(X_train), out_test.shape[1])

    out = names(pre)
    for expected in [
        "visit_date_year",
        "visit_date_month_sin",
        "visit_date_is_weekend",
        "discharge_datetime_hour",
        "discharge_datetime_hour_cos",
        "checkin_time_hour",
        "checkin_time_minute",
        "days_between_visit_date_discharge_datetime",
        "sex_1",
        "sex_2",
        "smoker",
        "income",
    ]:
        assert expected in out, expected
    assert "visit_date_hour" not in out  # visit_date has no times
    assert not [n for n in out if n.startswith("patient_id")]
    assert not [n for n in out if "days_between" in n and "checkin_time" in n]

    flagged = {
        c.name
        for c in prof.columns
        if c.dtype in ("numeric", "boolean", "datetime", "time") and c.missing_pct > 5
    }
    assert {n[: -len("_was_missing")] for n in out if n.endswith("_was_missing")} == flagged

    numeric_cols = g["numeric"][1]
    handler = pre.named_transformers_["numeric"].named_steps["outliers"]
    actions = dict(zip(numeric_cols, handler.actions_, strict=True))
    if family == "tree":
        assert set(actions.values()) == {"keep"}
    else:
        assert actions["income"] == _expected_auto(X_train["income"]) == "log"
        for c in numeric_cols:
            assert actions[c] == _expected_auto(X_train[c]), c

    model = (
        DecisionTreeClassifier(random_state=0)
        if family == "tree"
        else LogisticRegression(max_iter=1000)
    )
    pipe = Pipeline([("pre", clone(pre)), ("model", model)]).fit(X_train, y_train)
    assert set(pipe.predict(X_test)) <= {0, 1}


def test_clinic_target_in_frame_never_becomes_a_feature(
    clinic_split: tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series],
) -> None:
    X_train, _, y_train, _ = clinic_split
    train = X_train.assign(outcome=y_train.astype(float))
    pre = PP.build_preprocessor(P.profile(train), target="outcome", df=train).fit(train)
    assert not [n for n in names(pre) if n == "outcome" or n.startswith("outcome_")]


def test_clinic_outlier_options_apply(
    clinic_split: tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series],
) -> None:
    X_train, X_test, _, _ = clinic_split
    opts = PP.PreprocessOptions(outliers={"income": "missing", "age": "cap"})
    pre = PP.build_preprocessor(P.profile(X_train), df=X_train, family="tree", options=opts)
    pre.fit(X_train)
    numeric_cols = groups(pre)["numeric"][1]
    handler = pre.named_transformers_["numeric"].named_steps["outliers"]
    actions = dict(zip(numeric_cols, handler.actions_, strict=True))
    assert actions["income"] == "missing"
    assert actions["age"] == "keep"  # tree models get raw values
    assert np.isfinite(arr(pre.transform(X_test))).all()
