"""Phase 2 preprocessing tests (written test-first against the agreed contract).

build_preprocessor only reads the profile, so most tests hand-build a
DatasetProfile to pin column assignment exactly. Integration tests use the
real profile -> clean -> profile(clean_df) chain.
"""

import copy

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import NotFittedError
from sklearn.impute import MissingIndicator, SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.validation import check_is_fitted

from app.core import cleaning as C
from app.core import preprocessing as PP
from app.core import profiling as P
from tests.conftest import DATA

SAMPLE_TARGETS: dict[str, str] = {
    "breast_cancer.csv": "diagnosis",
    "diabetes.csv": "disease_progression",
    "iris.csv": "species",
    "messy.csv": "churned",
}


# ---------------------------------------------------------------- helpers


def col(name: str, dtype: str, missing_pct: float = 0.0, n_unique: int = 5) -> P.ColumnProfile:
    return P.ColumnProfile(name=name, dtype=dtype, missing_pct=missing_pct, n_unique=n_unique)  # type: ignore[arg-type]


def dprof(*cols: P.ColumnProfile, n_rows: int = 40) -> P.DatasetProfile:
    return P.DatasetProfile(n_rows=n_rows, n_cols=len(cols), columns=list(cols))


def groups(pre: ColumnTransformer) -> dict[str, tuple[object, list[str]]]:
    """name -> (transformer, column names) from an unfitted ColumnTransformer."""
    return {name: (trans, list(cols)) for name, trans, cols in pre.transformers}


def last_step(trans: object) -> object:
    return trans.steps[-1][1] if isinstance(trans, Pipeline) else trans


def assert_clean_matrix(out: object) -> np.ndarray:
    assert isinstance(out, np.ndarray), "expected dense output"
    assert out.dtype.kind == "f"
    assert np.isfinite(out).all()
    return out


FULL_PROFILE = dprof(
    col("num_a", "numeric", missing_pct=10.0, n_unique=30),
    col("num_b", "numeric", n_unique=40),
    col("flag", "boolean", missing_pct=2.5, n_unique=2),
    col("small_cat", "categorical", missing_pct=5.0, n_unique=3),
    col("big_cat", "categorical", n_unique=15),
    col("cid", "id", n_unique=40),
    col("notes", "text", n_unique=40),
    col("when", "datetime", n_unique=40),
    col("k", "constant", n_unique=1),
    col("y", "numeric", n_unique=40),
)


def full_frame(n: int = 40, seed: int = 0) -> pd.DataFrame:
    """Already-clean frame matching FULL_PROFILE."""
    rng = np.random.default_rng(seed)
    num_a = rng.normal(50, 10, n)
    num_a[::10] = np.nan
    flag = rng.integers(0, 2, n).astype(float)
    flag[5] = np.nan
    small = rng.choice(["a", "b", "c"], n).astype(object)
    small[7] = np.nan
    return pd.DataFrame(
        {
            "num_a": num_a,
            "num_b": rng.exponential(5, n),
            "flag": flag,
            "small_cat": small,
            "big_cat": [f"v{i % 15}" for i in range(n)],
            "cid": [f"C{i}" for i in range(n)],
            "notes": [f"some long note number {i} here" for i in range(n)],
            "when": pd.date_range("2024-01-01", periods=n).astype(str),
            "k": ["same"] * n,
            "y": rng.normal(size=n),
        }
    )


# ---------------------------------------------------------------- constants


def test_constants() -> None:
    assert PP.ONEHOT_MAX_UNIQUE == 10
    assert PP.SKEW_THRESHOLD == 1.0


# ---------------------------------------------------------------- build_preprocessor


def test_returns_unfitted_column_transformer() -> None:
    pre = PP.build_preprocessor(FULL_PROFILE, target="y")
    assert isinstance(pre, ColumnTransformer)
    assert pre.remainder == "drop"
    with pytest.raises(NotFittedError):
        check_is_fitted(pre)


def test_column_assignment_per_type() -> None:
    g = groups(PP.build_preprocessor(FULL_PROFILE, target="y"))
    assert set(g) == {"numeric", "missing_flags", "boolean", "onehot", "frequency"}
    assert g["numeric"][1] == ["num_a", "num_b"]
    assert g["missing_flags"][1] == ["num_a"]
    assert g["boolean"][1] == ["flag"]
    assert g["onehot"][1] == ["small_cat"]
    assert g["frequency"][1] == ["big_cat"]
    used = {c for _, cols in g.values() for c in cols}
    assert not used & {"y", "cid", "notes", "when", "k"}


def test_transformer_types() -> None:
    g = groups(PP.build_preprocessor(FULL_PROFILE, target="y"))

    numeric = g["numeric"][0]
    assert isinstance(numeric, Pipeline)
    kinds = [type(step) for _, step in numeric.steps]
    assert kinds == [PP.SkewAwareImputer, StandardScaler]

    assert isinstance(g["missing_flags"][0], MissingIndicator)

    boolean = g["boolean"][0]
    assert isinstance(boolean, SimpleImputer)
    assert boolean.strategy == "most_frequent"

    onehot = g["onehot"][0]
    assert isinstance(onehot, Pipeline)
    imp = onehot.steps[0][1]
    assert isinstance(imp, SimpleImputer)
    assert imp.strategy == "constant" and imp.fill_value == "missing"
    enc = last_step(onehot)
    assert isinstance(enc, OneHotEncoder)
    assert enc.handle_unknown == "ignore"
    assert enc.sparse_output is False

    freq = g["frequency"][0]
    assert isinstance(freq, Pipeline)
    imp = freq.steps[0][1]
    assert isinstance(imp, SimpleImputer)
    assert imp.strategy == "constant" and imp.fill_value == "missing"
    assert isinstance(last_step(freq), PP.FrequencyEncoder)


def test_target_excluded_from_every_group() -> None:
    for target in ["num_a", "flag", "small_cat", "big_cat"]:
        g = groups(PP.build_preprocessor(FULL_PROFILE, target=target))
        used = {c for _, cols in g.values() for c in cols}
        assert target not in used, target
    g = groups(PP.build_preprocessor(FULL_PROFILE, target="num_a"))
    assert "missing_flags" not in g  # num_a was the only numeric column with missing values


def test_no_target_uses_every_usable_column() -> None:
    g = groups(PP.build_preprocessor(FULL_PROFILE))
    assert g["numeric"][1] == ["num_a", "num_b", "y"]


def test_missing_flags_only_when_numeric_missing() -> None:
    prof = dprof(col("a", "numeric"), col("b", "categorical", missing_pct=30.0, n_unique=3))
    g = groups(PP.build_preprocessor(prof))
    assert "missing_flags" not in g
    assert set(g) == {"numeric", "onehot"}


def test_empty_groups_left_out() -> None:
    g = groups(PP.build_preprocessor(dprof(col("a", "numeric"), col("b", "numeric"))))
    assert set(g) == {"numeric"}
    g = groups(PP.build_preprocessor(dprof(col("f", "boolean", n_unique=2))))
    assert set(g) == {"boolean"}


@pytest.mark.parametrize(("n_unique", "group"), [(10, "onehot"), (11, "frequency"), (2, "onehot")])
def test_onehot_frequency_boundary(n_unique: int, group: str) -> None:
    prof = dprof(col("c", "categorical", n_unique=n_unique))
    g = groups(PP.build_preprocessor(prof))
    assert set(g) == {group}
    assert g[group][1] == ["c"]


def test_huge_cardinality_goes_to_frequency() -> None:
    g = groups(PP.build_preprocessor(dprof(col("c", "categorical", n_unique=5000))))
    assert set(g) == {"frequency"}


def test_no_usable_columns_raises() -> None:
    prof = dprof(
        col("cid", "id"),
        col("notes", "text"),
        col("when", "datetime"),
        col("k", "constant", n_unique=1),
        col("y", "numeric"),
    )
    with pytest.raises(ValueError):
        PP.build_preprocessor(prof, target="y")
    with pytest.raises(ValueError):
        PP.build_preprocessor(dprof())


def test_build_preprocessor_does_not_mutate_profile() -> None:
    before = copy.deepcopy(FULL_PROFILE)
    PP.build_preprocessor(FULL_PROFILE, target="y")
    assert FULL_PROFILE == before


def test_fit_transform_dense_and_finite() -> None:
    df = full_frame()
    pre = PP.build_preprocessor(FULL_PROFILE, target="y")
    train, test = df.iloc[:30], df.iloc[30:].copy()
    test.loc[test.index[0], "small_cat"] = "never_seen"
    test.loc[test.index[1], "big_cat"] = "brand_new"
    test.loc[test.index[2], "num_b"] = np.nan
    out_train = assert_clean_matrix(pre.fit_transform(train))
    out_test = assert_clean_matrix(pre.transform(test))
    assert out_train.shape[1] == out_test.shape[1]
    assert out_train.shape[0] == 30 and out_test.shape[0] == 10


def test_selects_columns_by_name_regardless_of_order() -> None:
    df = full_frame()
    pre = PP.build_preprocessor(FULL_PROFILE, target="y")
    pre.fit(df.iloc[:30])
    shuffled = df.iloc[30:][list(reversed(df.columns))]
    a = pre.transform(df.iloc[30:])
    b = pre.transform(shuffled)
    np.testing.assert_allclose(a, b)


def test_missing_in_test_but_not_in_train() -> None:
    """A numeric column with no NaN in the train fold can still have NaN at test time."""
    df = full_frame()
    df["num_a"] = df["num_a"].fillna(1.0)
    train, test = df.iloc[:30].copy(), df.iloc[30:].copy()
    test.loc[test.index[:3], "num_a"] = np.nan
    pre = PP.build_preprocessor(FULL_PROFILE, target="y")
    out_train = assert_clean_matrix(pre.fit_transform(train))
    out_test = assert_clean_matrix(pre.transform(test))
    assert out_train.shape[1] == out_test.shape[1]


# ---------------------------------------------------------------- SkewAwareImputer

SKEWED = [1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 3.0, 100.0, np.nan]  # skew ~2.8, median 1.5
SYMMETRIC = [1.0, 2.0, 3.0, 4.0, 5.0, np.nan, 3.0, 3.0, 3.0]  # skew 0, mean 3.0


def test_imputer_picks_median_for_skewed_mean_for_symmetric() -> None:
    X = pd.DataFrame({"skewed": SKEWED, "sym": SYMMETRIC})
    imp = PP.SkewAwareImputer().fit(X)
    assert imp.strategies_ == ["median", "mean"]
    assert isinstance(imp.statistics_, np.ndarray)
    np.testing.assert_allclose(imp.statistics_, [1.5, 3.0])
    assert imp.n_features_in_ == 2
    out = imp.transform(X)
    assert isinstance(out, np.ndarray) and out.dtype.kind == "f"
    assert not np.isnan(out).any()
    assert out[8, 0] == pytest.approx(1.5)
    assert out[5, 1] == pytest.approx(3.0)


def test_imputer_accepts_ndarray() -> None:
    X = np.array([SKEWED, SYMMETRIC]).T
    imp = PP.SkewAwareImputer().fit(X)
    assert imp.strategies_ == ["median", "mean"]
    assert not np.isnan(imp.transform(X)).any()


def test_imputer_threshold_param() -> None:
    X = pd.DataFrame({"skewed": SKEWED})
    imp = PP.SkewAwareImputer(skew_threshold=5.0).fit(X)
    assert imp.strategies_ == ["mean"]
    assert imp.statistics_[0] == pytest.approx(np.nanmean(SKEWED))


def test_imputer_decision_comes_from_train_only() -> None:
    train = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0]})
    test = pd.DataFrame({"x": [1.0, 1.0, 1.0, 1.0, 100.0, np.nan]})
    imp = PP.SkewAwareImputer().fit(train)
    stats_before = imp.statistics_.copy()
    strategies_before = list(imp.strategies_)
    out = imp.transform(test)
    np.testing.assert_array_equal(imp.statistics_, stats_before)
    assert imp.strategies_ == strategies_before == ["mean"]
    assert out[5, 0] == pytest.approx(3.0)  # train mean, not test median


def test_imputer_all_nan_train_column() -> None:
    X = pd.DataFrame({"empty": [np.nan, np.nan, np.nan], "x": [1.0, 2.0, 3.0]})
    imp = PP.SkewAwareImputer().fit(X)
    assert imp.statistics_[0] == 0.0
    assert imp.strategies_[0] == "mean"
    out = imp.transform(X)
    np.testing.assert_array_equal(out[:, 0], [0.0, 0.0, 0.0])


def test_imputer_leaves_present_values_alone() -> None:
    X = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0]})
    out = PP.SkewAwareImputer().fit(X).transform(X)
    np.testing.assert_array_equal(out[:, 0], [1.0, 2.0, 3.0, 4.0])


def test_imputer_feature_names_out() -> None:
    X = pd.DataFrame({"a": [1.0, np.nan], "b": [2.0, 3.0]})
    imp = PP.SkewAwareImputer().fit(X)
    assert list(imp.get_feature_names_out()) == ["a", "b"]
    assert list(imp.get_feature_names_out(["p", "q"])) == ["p", "q"]


def test_imputer_transform_before_fit_raises() -> None:
    with pytest.raises(NotFittedError):
        PP.SkewAwareImputer().transform(pd.DataFrame({"x": [1.0]}))


# ---------------------------------------------------------------- FrequencyEncoder


def test_frequency_encoder_shares_from_train() -> None:
    train = pd.DataFrame({"c": ["a", "a", "b", "c"], "d": ["x", "y", "y", "y"]})
    test = pd.DataFrame({"c": ["a", "c", "zzz"], "d": ["y", "new", "x"]})
    enc = PP.FrequencyEncoder().fit(train)
    out = enc.transform(test)
    assert isinstance(out, np.ndarray)
    assert out.dtype.kind == "f"
    np.testing.assert_allclose(out, [[0.5, 0.75], [0.25, 0.0], [0.0, 0.25]])


def test_frequency_encoder_nan_free_output() -> None:
    train = pd.DataFrame({"c": ["a", "b", "b"]})
    test = pd.DataFrame({"c": ["a", np.nan, None, "q"]})
    out = PP.FrequencyEncoder().fit(train).transform(test)
    assert not np.isnan(out).any()
    np.testing.assert_allclose(out[:, 0], [1 / 3, 0.0, 0.0, 0.0])


def test_frequency_encoder_does_not_refit() -> None:
    train = pd.DataFrame({"c": ["a", "b"]})
    enc = PP.FrequencyEncoder().fit(train)
    enc.transform(pd.DataFrame({"c": ["b", "b", "b", "b"]}))
    np.testing.assert_allclose(enc.transform(pd.DataFrame({"c": ["a"]})), [[0.5]])


def test_frequency_encoder_feature_names_out() -> None:
    enc = PP.FrequencyEncoder().fit(pd.DataFrame({"c1": ["a"], "c2": ["b"]}))
    assert list(enc.get_feature_names_out()) == ["c1", "c2"]


# ---------------------------------------------------------------- clone


def test_clone_works() -> None:
    pre = PP.build_preprocessor(FULL_PROFILE, target="y")
    cloned = clone(pre)
    assert groups(cloned).keys() == groups(pre).keys()
    imp = clone(PP.SkewAwareImputer(skew_threshold=2.0))
    assert imp.get_params()["skew_threshold"] == 2.0
    assert isinstance(clone(PP.FrequencyEncoder()), PP.FrequencyEncoder)


def test_clone_of_fitted_is_unfitted() -> None:
    imp = PP.SkewAwareImputer().fit(pd.DataFrame({"x": [1.0, 2.0]}))
    with pytest.raises(NotFittedError):
        check_is_fitted(clone(imp))


# ---------------------------------------------------------------- integration


def _clean_messy(messy_raw: pd.DataFrame) -> tuple[pd.DataFrame, P.DatasetProfile]:
    clean_df, _ = C.clean(messy_raw, P.profile(messy_raw))
    return clean_df, P.profile(clean_df)


def test_messy_end_to_end(messy_raw: pd.DataFrame) -> None:
    clean_df, prof = _clean_messy(messy_raw)
    pre = PP.build_preprocessor(prof, target="churned")
    g = groups(pre)
    assert "age" in g["missing_flags"][1]
    assert {"monthly_spend", "age"} <= set(g["numeric"][1])
    assert {"state", "plan"} <= set(g["onehot"][1])
    assert "notes_present" in g["boolean"][1]
    assert all("churned" not in cols for _, cols in g.values())

    X = clean_df.drop(columns="churned")
    y = clean_df["churned"].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=0, stratify=y
    )
    out_train = assert_clean_matrix(pre.fit_transform(X_train))
    out_test = assert_clean_matrix(pre.transform(X_test))
    assert out_train.shape[1] == out_test.shape[1]
    assert out_train.shape[0] == len(X_train)
    assert out_test.shape[0] == len(X_test)

    # no leakage: the age fill value comes from the train rows only
    numeric_cols = g["numeric"][1]
    imp = pre.named_transformers_["numeric"].steps[0][1]
    age_train = X_train["age"].dropna()
    expected = age_train.median() if abs(age_train.skew()) > PP.SKEW_THRESHOLD else age_train.mean()
    assert imp.statistics_[numeric_cols.index("age")] == pytest.approx(expected)


def test_messy_pipeline_with_logistic_regression(messy_raw: pd.DataFrame) -> None:
    clean_df, prof = _clean_messy(messy_raw)
    X = clean_df.drop(columns="churned")
    y = clean_df["churned"].astype(int)
    X_train, X_test, y_train, _ = train_test_split(X, y, test_size=0.25, random_state=0)
    model = Pipeline(
        [
            ("pre", PP.build_preprocessor(prof, target="churned")),
            ("model", LogisticRegression(max_iter=1000)),
        ]
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    assert len(pred) == len(X_test)
    assert set(pred) <= {0, 1}


def test_target_column_in_frame_is_ignored(messy_raw: pd.DataFrame) -> None:
    clean_df, prof = _clean_messy(messy_raw)
    pre = PP.build_preprocessor(prof, target="churned")
    with_target = pre.fit_transform(clean_df)
    without_target = clone(pre).fit_transform(clean_df.drop(columns="churned"))
    np.testing.assert_allclose(with_target, without_target)


def test_sample_csvs_preprocess(sample_name: str) -> None:
    raw = P.read_csv_raw(DATA / sample_name)
    clean_df, _ = C.clean(raw, P.profile(raw))
    target = SAMPLE_TARGETS[sample_name]
    assert target in clean_df.columns
    pre = PP.build_preprocessor(P.profile(clean_df), target=target)
    X = clean_df.drop(columns=target)
    X_train, X_test = train_test_split(X, test_size=0.25, random_state=0)
    out_train = assert_clean_matrix(pre.fit_transform(X_train))
    out_test = assert_clean_matrix(pre.transform(X_test))
    assert out_train.shape[1] == out_test.shape[1] > 0


# ---------------------------------------------------------------- review fixes


@pytest.mark.parametrize(
    "step_id", ["to_numeric:monthly_spend", "to_numeric:age", "to_bool:churned"]
)
def test_disabled_conversion_still_fits(messy_raw: pd.DataFrame, step_id: str) -> None:
    """Switching off a type fix keeps strings, so they must go to the category encoders."""
    clean_df, _ = C.clean(messy_raw, P.profile(messy_raw), {step_id})
    column = step_id.split(":")[1]
    pre = PP.build_preprocessor(P.profile(clean_df), df=clean_df)
    groups = {name: cols for name, _, cols in pre.transformers}
    assert column not in groups.get("numeric", []) + groups.get("boolean", [])
    assert column in groups.get("onehot", []) + groups.get("frequency", [])
    train, test = train_test_split(clean_df, test_size=0.3, random_state=0)
    out = pre.fit(train).transform(test)
    assert out.dtype == np.float64
    assert np.isfinite(out).all()


def test_every_fix_disabled_still_fits(messy_raw: pd.DataFrame) -> None:
    prof = P.profile(messy_raw)
    _, full_log = C.clean(messy_raw, prof)
    clean_df, _ = C.clean(messy_raw, prof, {s.step_id for s in full_log.steps})
    pre = PP.build_preprocessor(P.profile(clean_df), target="churned", df=clean_df)
    out = pre.fit_transform(clean_df)
    assert out.dtype == np.float64
    assert np.isfinite(out).all()


def test_df_argument_keeps_real_numbers_numeric(messy_raw: pd.DataFrame) -> None:
    clean_df, prof = _clean_messy(messy_raw)
    with_df = PP.build_preprocessor(prof, target="churned", df=clean_df)
    without = PP.build_preprocessor(prof, target="churned")
    assert [t[2] for t in with_df.transformers] == [t[2] for t in without.transformers]


def test_custom_transformers_reject_reordered_columns() -> None:
    train = pd.DataFrame({"a": [1.0, 2.0, np.nan], "b": [3.0, 4.0, 5.0]})
    imp = PP.SkewAwareImputer().fit(train)
    with pytest.raises(ValueError):
        imp.transform(train[["b", "a"]])
    enc = PP.FrequencyEncoder().fit(train.astype(str))
    with pytest.raises(ValueError):
        enc.transform(train.astype(str)[["b", "a"]])


def test_custom_transformers_pass_sklearn_checks() -> None:
    from sklearn.utils.estimator_checks import check_estimator

    for est in (PP.SkewAwareImputer(), PP.FrequencyEncoder()):
        results = check_estimator(est, on_fail=None)
        failed = [r["check_name"] for r in results if r["status"] == "failed"]
        assert failed == [], f"{type(est).__name__} failed {failed}"
