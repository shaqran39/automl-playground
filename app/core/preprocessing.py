"""Phase 2: the learned part of preprocessing, built as an unfitted ColumnTransformer.

Everything here is fitted on the training rows only, inside the model
Pipeline, so nothing leaks from the test set. build_preprocessor just reads a
profile of the cleaned frame and decides which columns go where.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import MissingIndicator, SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils import Tags
from sklearn.utils.validation import check_is_fitted, validate_data

from app.core.profiling import ColumnProfile, ColumnType, DatasetProfile

ONEHOT_MAX_UNIQUE = 10
SKEW_THRESHOLD = 1.0
MISSING_FILL = "missing"
EXCLUDED_TYPES = frozenset({"id", "constant", "text", "datetime"})


# ---------------------------------------------------------------- shared helpers


def _validate(est: BaseEstimator, X: Any, numeric: bool, reset: bool) -> np.ndarray:
    """sklearn's input checks: 2D, width and column names match fit. NaN is allowed."""
    if numeric:
        return validate_data(est, X, dtype=float, ensure_all_finite="allow-nan", reset=reset)
    return validate_data(est, X, dtype=None, ensure_all_finite=False, reset=reset)


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


# ---------------------------------------------------------------- FrequencyEncoder


def _shares(column: pd.Series) -> dict[Any, float]:
    n = len(column)
    if n == 0:
        return {}
    return {k: float(v) / n for k, v in column.value_counts(dropna=True).items()}


class FrequencyEncoder(TransformerMixin, BaseEstimator):
    """Replace each category with its share of training rows. Unseen values get 0.0.

    NaN also gets 0.0 when used alone. In build_preprocessor an imputer runs first,
    so blanks become the category "missing" and get their own learned share.
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


def _usable(profile: DatasetProfile, target: str | None) -> list[ColumnProfile]:
    return [c for c in profile.columns if c.name != target and c.dtype not in EXCLUDED_TYPES]


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


def _category_pipeline(encoder: BaseEstimator) -> Pipeline:
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="constant", fill_value=MISSING_FILL)),
            ("encode", encoder),
        ]
    )


def build_preprocessor(
    profile: DatasetProfile, target: str | None = None, df: pd.DataFrame | None = None
) -> ColumnTransformer:
    """Unfitted ColumnTransformer for the cleaned frame described by profile.

    Pass the cleaned frame (or the training split) as df so columns whose type
    conversion was switched off are routed by their real dtype. Raises
    ValueError when no column is usable as a feature.
    """
    cols = [(c, _route_type(c, df)) for c in _usable(profile, target)]
    numeric = [c.name for c, t in cols if t == "numeric"]
    groups: list[tuple[str, Any, list[str]]] = [
        (
            "numeric",
            Pipeline([("impute", SkewAwareImputer()), ("scale", StandardScaler())]),
            numeric,
        ),
        (
            "missing_flags",
            MissingIndicator(features="all"),
            [c.name for c, t in cols if t == "numeric" and c.missing_pct > 0],
        ),
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
        raise ValueError("No usable feature columns: everything is the target, an ID, text, etc.")
    return ColumnTransformer(used, remainder="drop", sparse_threshold=0.0)
