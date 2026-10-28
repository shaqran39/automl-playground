"""Phase 2: the learned part of preprocessing, built as an unfitted ColumnTransformer.

Everything here is fitted on the training rows only, inside the model
Pipeline, so nothing leaks from the test set. That includes outlier handling
(percentiles, IQR fences, skew), date features, imputer fills and scaling.
build_preprocessor just reads a profile of the cleaned frame and decides which
columns go where.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils import Tags
from sklearn.utils.validation import check_is_fitted, validate_data

from app.core.cleaning import to_labels
from app.core.profiling import ColumnProfile, ColumnType, DatasetProfile, parse_times

ONEHOT_MAX_UNIQUE = 10
SKEW_THRESHOLD = 1.0
MISSING_FILL = "Missing"
WAS_MISSING_PCT = 5.0
IQR_FENCE = 1.5
CAP_PERCENTILES = (1.0, 99.0)
EXCLUDED_TYPES = frozenset({"id", "constant", "text"})

OutlierChoice = Literal["auto", "keep", "cap", "log", "missing"]
ImputerChoice = Literal["auto", "knn", "iterative"]
ModelFamily = Literal["linear", "tree"]  # linear = linear models, SVM, KNN
DateKind = Literal["date", "datetime", "time"]

OUTLIER_CHOICES: frozenset[str] = frozenset({"auto", "keep", "cap", "log", "missing"})
DATE_KINDS: frozenset[str] = frozenset({"date", "datetime", "time"})


class NoUsableFeaturesError(ValueError):
    """Every column is the target, an ID, text or a constant, so there's nothing to learn from."""


@dataclass
class PreprocessOptions:
    """User choices from the UI. A column missing from outliers means "auto"."""

    outliers: dict[str, OutlierChoice] = field(default_factory=dict)
    numeric_imputer: ImputerChoice = "auto"


# ---------------------------------------------------------------- shared helpers


def _validate(est: BaseEstimator, X: Any, numeric: bool, reset: bool) -> np.ndarray:
    """sklearn's input checks: 2D, width and column names match fit. NaN is allowed."""
    if numeric:
        return validate_data(est, X, dtype=float, ensure_all_finite="allow-nan", reset=reset)
    return validate_data(est, X, dtype=None, ensure_all_finite=False, reset=reset)


def _as_frame(est: BaseEstimator, X: Any, reset: bool) -> pd.DataFrame:
    """X as a positional DataFrame, keeping real dtypes (dates stay dates).

    Only the width and column names are checked, since numpy would turn mixed
    date and text columns into one object array.
    """
    validate_data(est, X, skip_check_array=True, reset=reset)
    if isinstance(X, pd.DataFrame):
        return X.set_axis(range(X.shape[1]), axis=1)
    arr = np.asarray(X, dtype=object)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D input, got {arr.ndim}D.")
    return pd.DataFrame(arr)


def _names_out(est: BaseEstimator, input_features: Any) -> np.ndarray:
    """Output names are the input names: the ones passed in, else the fitted ones."""
    check_is_fitted(est, "n_features_in_")
    if input_features is not None:
        names = np.asarray([str(n) for n in input_features], dtype=object)
        if len(names) != est.n_features_in_:
            raise ValueError(f"Expected {est.n_features_in_} feature names, got {len(names)}.")
        return names
    if hasattr(est, "feature_names_in_"):
        return np.asarray(est.feature_names_in_, dtype=object)
    return np.asarray([f"x{i}" for i in range(est.n_features_in_)], dtype=object)


def _per_column(
    values: tuple[str, ...], n: int, default: str, allowed: frozenset[str], what: str
) -> list[str]:
    """One setting per column. Empty means default everywhere."""
    if not values:
        return [default] * n
    if len(values) != n:
        raise ValueError(f"Got {len(values)} {what} for {n} columns.")
    bad = sorted(set(values) - allowed)
    if bad:
        raise ValueError(f"Unknown {what}: {bad}. Pick from {sorted(allowed)}.")
    return list(values)


# ---------------------------------------------------------------- SkewAwareImputer


def _pick_fill(column: np.ndarray, skew_threshold: float) -> tuple[str, float]:
    """(strategy, value) for one train column: median if skewed, mean otherwise."""
    present = pd.Series(column[~np.isnan(column)])
    if present.empty:
        return "mean", 0.0
    skew = present.skew()
    if pd.notna(skew) and abs(float(skew)) > skew_threshold:
        return "median", float(present.median())
    return "mean", float(present.mean())


class SkewAwareImputer(TransformerMixin, BaseEstimator):
    """Fill NaN with the median for skewed columns and the mean otherwise.

    The median vs mean choice is made at fit time from the training rows only,
    never from the full-data profile.
    """

    def __init__(self, skew_threshold: float = SKEW_THRESHOLD) -> None:
        self.skew_threshold = skew_threshold

    def __sklearn_tags__(self) -> Tags:
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = True
        return tags

    def fit(self, X: Any, y: Any = None) -> SkewAwareImputer:
        arr = _validate(self, X, numeric=True, reset=True)
        picks = [_pick_fill(arr[:, j], self.skew_threshold) for j in range(arr.shape[1])]
        self.strategies_ = [s for s, _ in picks]
        self.statistics_ = np.array([v for _, v in picks], dtype=float)
        return self

    def transform(self, X: Any) -> np.ndarray:
        check_is_fitted(self, "statistics_")
        arr = _validate(self, X, numeric=True, reset=False).copy()
        missing = np.isnan(arr)
        arr[missing] = np.take(self.statistics_, np.nonzero(missing)[1])
        return arr

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        return _names_out(self, input_features)


# ---------------------------------------------------------------- OutlierHandler


@dataclass(frozen=True)
class _ColumnStats:
    """What OutlierHandler learns about one train column. NaN when the column is empty."""

    low: float  # 1st percentile
    high: float  # 99th percentile
    fence_low: float
    fence_high: float
    skew: float
    min: float


def _column_stats(column: np.ndarray) -> _ColumnStats:
    present = column[~np.isnan(column)]
    if present.size == 0:
        return _ColumnStats(*([np.nan] * 6))
    low, q1, q3, high = np.percentile(present, [CAP_PERCENTILES[0], 25, 75, CAP_PERCENTILES[1]])
    iqr = q3 - q1
    skew = pd.Series(present).skew()
    return _ColumnStats(
        low=float(low),
        high=float(high),
        fence_low=float(q1 - IQR_FENCE * iqr),
        fence_high=float(q3 + IQR_FENCE * iqr),
        skew=float(skew) if pd.notna(skew) else 0.0,
        min=float(present.min()),
    )


def _auto_action(column: np.ndarray, stats: _ColumnStats) -> str:
    """keep with no train outliers, log for right skew on non-negatives, cap otherwise."""
    present = column[~np.isnan(column)]
    outside = (present < stats.fence_low) | (present > stats.fence_high)
    if present.size == 0 or not bool(outside.any()):
        return "keep"
    if stats.skew > SKEW_THRESHOLD and stats.min >= 0:
        return "log"
    return "cap"


def _resolve_action(choice: str, column: np.ndarray, stats: _ColumnStats, family: str) -> str:
    """Tree models split on raw values, so cap and log are pointless there."""
    if choice == "missing" or choice == "keep":
        return choice
    if family == "tree":
        return "keep"
    return _auto_action(column, stats) if choice == "auto" else choice


class OutlierHandler(TransformerMixin, BaseEstimator):
    """Per-column outlier handling, learned from the training rows only.

    choices holds one of auto / keep / cap / log / missing per column (empty
    means auto everywhere). fit learns the 1st/99th percentiles, the 1.5x IQR
    fences, skew and min of each column and resolves auto into actions_:
    keep when no train value is outside the fences, log for right skew
    (skew > 1) with no negatives, cap otherwise. For family="tree" only
    "missing" does anything.

    transform: cap clips to the train 1st/99th percentiles, log is
    log1p(max(x, 0)), missing turns values outside the train fences into NaN.
    NaN stays NaN.
    """

    def __init__(self, choices: tuple[str, ...] = (), family: str = "linear") -> None:
        self.choices = choices
        self.family = family

    def __sklearn_tags__(self) -> Tags:
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = True
        return tags

    def fit(self, X: Any, y: Any = None) -> OutlierHandler:
        if self.family not in ("linear", "tree"):
            raise ValueError(f"family must be 'linear' or 'tree', got {self.family!r}.")
        arr = _validate(self, X, numeric=True, reset=True)
        choices = _per_column(
            tuple(self.choices), arr.shape[1], "auto", OUTLIER_CHOICES, "outlier choices"
        )
        stats = [_column_stats(arr[:, j]) for j in range(arr.shape[1])]
        self.actions_ = [
            _resolve_action(c, arr[:, j], s, self.family)
            for j, (c, s) in enumerate(zip(choices, stats, strict=True))
        ]
        self.low_ = np.array([s.low for s in stats], dtype=float)
        self.high_ = np.array([s.high for s in stats], dtype=float)
        self.fence_low_ = np.array([s.fence_low for s in stats], dtype=float)
        self.fence_high_ = np.array([s.fence_high for s in stats], dtype=float)
        self.skew_ = np.array([s.skew for s in stats], dtype=float)
        self.min_ = np.array([s.min for s in stats], dtype=float)
        return self

    def _apply(self, column: np.ndarray, j: int) -> np.ndarray:
        action = self.actions_[j]
        if action == "cap":
            return np.clip(column, self.low_[j], self.high_[j])
        if action == "log":
            return np.log1p(np.maximum(column, 0.0))
        if action == "missing":
            outside = (column < self.fence_low_[j]) | (column > self.fence_high_[j])
            return np.where(outside, np.nan, column)
        return column

    def transform(self, X: Any) -> np.ndarray:
        check_is_fitted(self, "actions_")
        arr = _validate(self, X, numeric=True, reset=False).astype(float, copy=True)
        for j in range(arr.shape[1]):
            arr[:, j] = self._apply(arr[:, j], j)
        return arr

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        return _names_out(self, input_features)


# ---------------------------------------------------------------- DateFeatures

DATE_PARTS = ("year", "month", "quarter", "dayofweek", "is_weekend")
DATE_CYCLES = ("month_sin", "month_cos", "dayofweek_sin", "dayofweek_cos")
TIME_PARTS = ("hour", "minute", "hour_sin", "hour_cos")


def _to_datetimes(column: pd.Series) -> pd.Series:
    """datetime64, tz-naive in UTC. Strings are coerced, junk becomes NaT."""
    if isinstance(column.dtype, pd.DatetimeTZDtype):
        return column.dt.tz_convert("UTC").dt.tz_localize(None)
    if pd.api.types.is_datetime64_any_dtype(column):
        return column
    parsed = pd.to_datetime(column, errors="coerce", utc=True, format="mixed")
    return parsed.dt.tz_localize(None)


def _cycle(values: pd.Series, period: float) -> tuple[pd.Series, pd.Series]:
    angle = 2 * np.pi * values / period
    return np.sin(angle), np.cos(angle)


def _clock_features(hour: pd.Series, minute: pd.Series) -> list[pd.Series]:
    sin, cos = _cycle(hour, 24)
    return [hour, minute, sin, cos]


def _date_features(when: pd.Series, with_time: bool) -> list[pd.Series]:
    month = when.dt.month.astype(float)
    dow = when.dt.day_of_week.astype(float)
    parts = [
        when.dt.year.astype(float),
        month,
        when.dt.quarter.astype(float),
        dow,
        (dow >= 5).astype(float).where(when.notna()),
        *_cycle(month - 1, 12),
        *_cycle(dow, 7),
    ]
    if with_time:
        parts += _clock_features(when.dt.hour.astype(float), when.dt.minute.astype(float))
    return parts


def _time_features(column: pd.Series) -> list[pd.Series]:
    """Features from "HH:MM:SS" strings (raw times like "2:30 PM" work too)."""
    clock = parse_times(column)
    hour = pd.to_numeric(clock.str[:2], errors="coerce").astype(float)
    minute = pd.to_numeric(clock.str[3:5], errors="coerce").astype(float)
    return _clock_features(hour, minute)


def _feature_names(name: str, kind: str) -> list[str]:
    if kind == "time":
        suffixes: tuple[str, ...] = TIME_PARTS
    else:
        suffixes = DATE_PARTS + DATE_CYCLES + (TIME_PARTS if kind == "datetime" else ())
    return [f"{name}_{s}" for s in suffixes]


def date_feature_names(names: list[str], kinds: list[str]) -> list[str]:
    """Default DateFeatures output names: per-column parts, then days_between pairs."""
    out = [f for n, k in zip(names, kinds, strict=True) for f in _feature_names(n, k)]
    dated = [n for n, k in zip(names, kinds, strict=True) if k != "time"]
    return out + [f"days_between_{a}_{b}" for i, a in enumerate(dated) for b in dated[i + 1 :]]


def _override_names(default: list[str], names: tuple[str, ...]) -> np.ndarray:
    """names when given (to dodge clashes with real columns), else the defaults."""
    if names and len(names) != len(default):
        raise ValueError(f"Got {len(names)} output names for {len(default)} features.")
    return np.asarray(list(names) if names else default, dtype=object)


class DateFeatures(TransformerMixin, BaseEstimator):
    """Numbers a model can use from date, datetime and time-of-day columns.

    kinds holds "date", "datetime" (has a time part) or "time" per column, empty
    means "date" everywhere. Stateless: fit only records the input columns.
    Missing or unparseable values give NaN. With two or more date/datetime
    columns, days_between_{a}_{b} = (b - a) in days is added for every pair,
    at the end. names, when given, replaces the output names one for one.
    """

    def __init__(self, kinds: tuple[str, ...] = (), names: tuple[str, ...] = ()) -> None:
        self.kinds = kinds
        self.names = names

    def __sklearn_tags__(self) -> Tags:
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = True
        tags.input_tags.string = True
        tags.requires_fit = False
        return tags

    def _kinds(self, n: int) -> list[str]:
        return _per_column(tuple(self.kinds), n, "date", DATE_KINDS, "date kinds")

    def fit(self, X: Any, y: Any = None) -> DateFeatures:
        frame = _as_frame(self, X, reset=True)
        self._kinds(frame.shape[1])
        return self

    def transform(self, X: Any) -> np.ndarray:
        check_is_fitted(self, "n_features_in_")
        frame = _as_frame(self, X, reset=False)
        kinds = self._kinds(frame.shape[1])
        cols: list[pd.Series] = []
        dates: list[pd.Series] = []
        for j, kind in enumerate(kinds):
            if kind == "time":
                cols += _time_features(frame[j])
                continue
            when = _to_datetimes(frame[j])
            cols += _date_features(when, kind == "datetime")
            dates.append(when)
        for a in range(len(dates)):
            for b in range(a + 1, len(dates)):
                cols.append((dates[b] - dates[a]) / pd.Timedelta(days=1))
        if not cols:
            return np.empty((len(frame), 0))
        return np.column_stack([c.to_numpy(dtype=float, na_value=np.nan) for c in cols])

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        inputs = [str(n) for n in _names_out(self, input_features)]
        default = date_feature_names(inputs, self._kinds(len(inputs)))
        return _override_names(default, tuple(self.names))


# ---------------------------------------------------------------- WasMissing


def was_missing_names(names: list[str]) -> list[str]:
    return [f"{n}_was_missing" for n in names]


class WasMissing(TransformerMixin, BaseEstimator):
    """{col}_was_missing as 0/1 for every input column. Stateless.

    names, when given, replaces the output names one for one.
    """

    def __init__(self, names: tuple[str, ...] = ()) -> None:
        self.names = names

    def __sklearn_tags__(self) -> Tags:
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = True
        tags.input_tags.string = True
        tags.requires_fit = False
        return tags

    def fit(self, X: Any, y: Any = None) -> WasMissing:
        _as_frame(self, X, reset=True)
        return self

    def transform(self, X: Any) -> np.ndarray:
        check_is_fitted(self, "n_features_in_")
        return _as_frame(self, X, reset=False).isna().to_numpy(dtype=float)

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        inputs = [str(n) for n in _names_out(self, input_features)]
        return _override_names(was_missing_names(inputs), tuple(self.names))


# ---------------------------------------------------------------- ToLabels


class ToLabels(TransformerMixin, BaseEstimator):
    """Categories as text labels (2.0 -> "2"), so encoders never see a mix of numbers
    and the "Missing" fill. Missing stays NaN. Stateless.

    Needed because a numeric column can profile as categorical, like a quantity
    left with only 1 and 2 once its -1 codes are blanked, or a train split that
    only sees two values.
    """

    def __sklearn_tags__(self) -> Tags:
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = True
        tags.input_tags.string = True
        tags.input_tags.categorical = True
        tags.requires_fit = False
        return tags

    def fit(self, X: Any, y: Any = None) -> ToLabels:
        _as_frame(self, X, reset=True)
        return self

    def transform(self, X: Any) -> np.ndarray:
        check_is_fitted(self, "n_features_in_")
        frame = _as_frame(self, X, reset=False)
        cols = [to_labels(frame[j]).to_numpy(dtype=object) for j in frame.columns]
        return np.column_stack(cols) if cols else np.empty((len(frame), 0), dtype=object)

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        return _names_out(self, input_features)


# ---------------------------------------------------------------- FrequencyEncoder


def _shares(column: pd.Series) -> dict[Any, float]:
    n = len(column)
    if n == 0:
        return {}
    return {k: float(v) / n for k, v in column.value_counts(dropna=True).items()}


class FrequencyEncoder(TransformerMixin, BaseEstimator):
    """Replace each category with its share of training rows. Unseen values get 0.0.

    NaN also gets 0.0 when used alone. In build_preprocessor an imputer runs first,
    so blanks become the category "Missing" and get their own learned share.
    """

    def __sklearn_tags__(self) -> Tags:
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = True
        tags.input_tags.categorical = True
        tags.input_tags.string = True
        return tags

    def fit(self, X: Any, y: Any = None) -> FrequencyEncoder:
        frame = pd.DataFrame(_validate(self, X, numeric=False, reset=True))
        self.frequencies_ = [_shares(frame[j]) for j in frame.columns]
        return self

    def transform(self, X: Any) -> np.ndarray:
        check_is_fitted(self, "frequencies_")
        frame = pd.DataFrame(_validate(self, X, numeric=False, reset=False))
        cols = [
            frame[j].map(self.frequencies_[j]).astype(float).fillna(0.0).to_numpy()
            for j in frame.columns
        ]
        return np.column_stack(cols) if cols else np.empty((len(frame), 0))

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        return _names_out(self, input_features)


# ---------------------------------------------------------------- build_preprocessor


def _without_target(profile: DatasetProfile, target: str | None) -> list[ColumnProfile]:
    return [c for c in profile.columns if c.name != target]


def _route_type(col: ColumnProfile, df: pd.DataFrame | None) -> ColumnType:
    """The profile type, except numeric/boolean columns still stored as strings.

    That happens when the user switches off to_numeric or to_bool in clean().
    The profiler still reads "$79.03" as a number, but the values are text, so
    they go to the category encoders instead of crashing the numeric imputer.
    """
    if (
        df is not None
        and col.dtype in ("numeric", "boolean")
        and col.name in df.columns
        and not pd.api.types.is_numeric_dtype(df[col.name])
    ):
        return "categorical"
    return col.dtype


def _date_kind(col: ColumnProfile, route: ColumnType) -> DateKind:
    if route == "time":
        return "time"
    return "datetime" if col.has_time else "date"


def _category_pipeline(encoder: BaseEstimator) -> Pipeline:
    return Pipeline(
        [
            ("labels", ToLabels()),
            ("impute", SimpleImputer(strategy="constant", fill_value=MISSING_FILL)),
            ("encode", encoder),
        ]
    )


class _KNNImputer(KNNImputer):
    """KNNImputer whose names stay one-to-one when it keeps an all-blank column.

    With keep_empty_features=True the stock one keeps the column in its output
    but drops it from get_feature_names_out, so names and values stop lining up.
    """

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        if not self.keep_empty_features:
            return super().get_feature_names_out(input_features)
        return _names_out(self, input_features)


def _numeric_imputer(choice: ImputerChoice) -> BaseEstimator:
    # keep_empty_features so an all-blank train column doesn't shift the feature names
    if choice == "knn":
        return _KNNImputer(keep_empty_features=True)
    if choice == "iterative":
        return IterativeImputer(random_state=0, keep_empty_features=True)
    return SkewAwareImputer()


def _numeric_pipeline(
    choices: tuple[str, ...], family: ModelFamily, imputer: ImputerChoice
) -> Pipeline:
    """Outliers first (on raw values), then scale, then fill gaps on the scaled values."""
    return Pipeline(
        [
            ("outliers", OutlierHandler(choices=choices, family=family)),
            ("scale", StandardScaler()),
            ("impute", _numeric_imputer(imputer)),
        ]
    )


def _dates_pipeline(kinds: tuple[str, ...], names: tuple[str, ...]) -> Pipeline:
    return Pipeline(
        [
            ("features", DateFeatures(kinds=kinds, names=names)),
            ("scale", StandardScaler()),
            ("impute", SkewAwareImputer()),
        ]
    )


def _free_names(wanted: list[str], taken: set[str]) -> list[str]:
    """wanted, with name_2, name_3... wherever a name is already taken. Updates taken."""
    out: list[str] = []
    for name in wanted:
        free, i = name, 2
        while free in taken:
            free, i = f"{name}_{i}", i + 1
        taken.add(free)
        out.append(free)
    return out


def _renamed(wanted: list[str], taken: set[str]) -> tuple[str, ...]:
    """Names to pass as an override, or () when none of the defaults clash."""
    free = _free_names(wanted, taken)
    return tuple(free) if free != wanted else ()


def _input_names(profile: DatasetProfile, df: pd.DataFrame | None) -> set[str]:
    names = {c.name for c in profile.columns}
    return names | {str(c) for c in df.columns} if df is not None else names


def _check_no_target(ct: ColumnTransformer, target: str | None) -> None:
    if target is None:
        return
    for name, _, cols in ct.transformers:
        if target in list(cols):
            raise AssertionError(f"The target '{target}' ended up in the {name} group.")


def build_preprocessor(
    profile: DatasetProfile,
    target: str | None = None,
    df: pd.DataFrame | None = None,
    *,
    family: ModelFamily = "linear",
    options: PreprocessOptions | None = None,
) -> ColumnTransformer:
    """Unfitted ColumnTransformer for the cleaned frame described by profile.

    Pass the cleaned frame (or the training split) as df so columns whose type
    conversion was switched off are routed by their real dtype. family picks
    how outliers are handled (tree models get raw values) and options carries
    the user's outlier and imputer choices. The target never becomes a feature.
    Generated names (date parts, was_missing flags) that clash with a real
    column or each other get a free name like visit_year_2.
    Raises NoUsableFeaturesError (a ValueError) when no column is usable.
    """
    options = options or PreprocessOptions()
    if df is not None and target is not None and target in df.columns:
        df = df.drop(columns=target)
    cols = [(c, _route_type(c, df)) for c in _without_target(profile, target)]
    cols = [(c, t) for c, t in cols if t not in EXCLUDED_TYPES]

    numeric = [c.name for c, t in cols if t == "numeric"]
    choices = tuple(options.outliers.get(n, "auto") for n in numeric)
    dated = [(c, t) for c, t in cols if t in ("datetime", "time")]
    flagged = [
        c.name
        for c, t in cols
        if t in ("numeric", "boolean", "datetime", "time") and c.missing_pct > WAS_MISSING_PCT
    ]
    kinds = [_date_kind(c, t) for c, t in dated]
    taken = _input_names(profile, df)
    date_names = _renamed(date_feature_names([c.name for c, _ in dated], kinds), taken)
    flag_names = _renamed(was_missing_names(flagged), taken)
    groups: list[tuple[str, Any, list[str]]] = [
        ("numeric", _numeric_pipeline(choices, family, options.numeric_imputer), numeric),
        ("dates", _dates_pipeline(tuple(kinds), date_names), [c.name for c, _ in dated]),
        ("was_missing", WasMissing(names=flag_names), flagged),
        (
            "boolean",
            SimpleImputer(strategy="most_frequent"),
            [c.name for c, t in cols if t == "boolean"],
        ),
        (
            "onehot",
            _category_pipeline(OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            [c.name for c, t in cols if t == "categorical" and c.n_unique <= ONEHOT_MAX_UNIQUE],
        ),
        (
            "frequency",
            _category_pipeline(FrequencyEncoder()),
            [c.name for c, t in cols if t == "categorical" and c.n_unique > ONEHOT_MAX_UNIQUE],
        ),
    ]
    used = [g for g in groups if g[2]]
    if not used:
        raise NoUsableFeaturesError(
            "No usable feature columns: everything is the target, an ID, text, etc."
        )
    ct = ColumnTransformer(
        used, remainder="drop", sparse_threshold=0.0, verbose_feature_names_out=False
    )
    _check_no_target(ct, target)
    return ct
