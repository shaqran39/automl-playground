"""Phase 1: dataset profiling. Detects column types and per-column stats.

Profiling is read-only: nothing here is fitted or learned, and the input frame
is never mutated. Every type decision comes with plain-English reasons.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

ColumnType = Literal["numeric", "categorical", "boolean", "datetime", "text", "id", "constant"]

MISSING_TOKENS: frozenset[str] = frozenset({"", "na", "n/a", "nan", "null", "none", "-", "?"})
BOOLEAN_PAIRS: frozenset[frozenset[str]] = frozenset(
    frozenset(p)
    for p in [("yes", "no"), ("y", "n"), ("true", "false"), ("t", "f"), ("1", "0"), ("on", "off")]
)
CURRENCY_SYMBOLS = "$€£¥"

PARSE_THRESHOLD = 0.95
PRESENCE_MISSING_PCT = 5.0
ID_UNIQUE_RATIO = 0.95
ID_MIN_VALUES = 10
TEXT_MIN_WORDS = 5.0
TEXT_MIN_UNIQUE = 50
TEXT_UNIQUE_RATIO = 0.5
MAX_SAMPLES = 5
MAX_TOP_VALUES = 10

_NUMBER_RE = re.compile(r"^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$")
_THOUSANDS_RE = r"^[+-]?\d{1,3}(,\d{3})+(\.\d*)?$"
_ISO_DATE_RE = r"^\d{4}-\d{1,2}-\d{1,2}$"
_ISO_DATETIME_RE = r"^\d{4}-\d{2}-\d{2}[ T]\d{1,2}:\d{2}"
_TIME = r"(?:([ T])(\d{1,2}:\d{2}(?::\d{2})?))?"
_YMD_SLASH_RE = r"^\d{4}/\d{1,2}/\d{1,2}" + _TIME + "$"
_DMY_RE = r"^(\d{1,2})([/.-])(\d{1,2})\2(\d{4}|\d{2})" + _TIME + "$"
_MONTH_NAME_RE = (
    r"(?i)\b(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?"
    r"|sept?(?:ember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\.?\b"
)
_YEAR_RE = r"\b\d{4}\b"
_INT_RE = r"[+-]?\d+(\.0+)?"
_ID_NAME_RE = re.compile(r"(^|[_\s.-])(id|key|uuid|guid)$|^(id|key|uuid|guid)[_\s.-]", re.I)
_CAMEL_ID_RE = re.compile(r"[a-z0-9](Id|ID|Key|Uuid)$")
YEAR_RANGE = (1900, 2100)
SEQ_ID_MIN_VALUES = 50


@dataclass
class ColumnProfile:
    name: str
    dtype: ColumnType
    missing_pct: float
    n_unique: int
    parse_rate: float = 1.0
    reasons: list[str] = field(default_factory=list)
    sample_values: list[str] = field(default_factory=list)
    stats: dict[str, float | None] | None = None
    n_outliers: int = 0
    outlier_pct: float = 0.0
    outlier_bounds: tuple[float, float] | None = None
    disguise: list[str] = field(default_factory=list)
    top_values: dict[str, int] = field(default_factory=dict)
    case_variants: dict[str, list[str]] = field(default_factory=dict)
    date_format: str | None = None
    dayfirst: bool | None = None
    date_ambiguous: bool = False
    date_range: tuple[str, str] | None = None
    all_missing: bool = False


@dataclass
class DatasetProfile:
    n_rows: int
    n_cols: int
    n_duplicate_rows: int = 0
    columns: list[ColumnProfile] = field(default_factory=list)

    def column(self, name: str) -> ColumnProfile:
        """Return the first column profile with this name. Raises KeyError if absent."""
        for col in self.columns:
            if col.name == name:
                return col
        raise KeyError(name)


# ---------------------------------------------------------------- reading


def read_csv_raw(path: str | Path) -> pd.DataFrame:
    """Read a CSV with every cell kept as a raw string (blanks stay "")."""
    return pd.read_csv(path, dtype=str, keep_default_na=False)


# ---------------------------------------------------------------- normalising


def _is_missing_scalar(value: Any) -> bool:
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def to_clean_strings(values: pd.Series) -> pd.Series:
    """Stripped string version of a column, with every kind of missing as None."""
    out: list[str | None] = []
    for v in values:
        if v is None or _is_missing_scalar(v):
            out.append(None)
            continue
        text = str(v).strip()
        out.append(None if text.lower() in MISSING_TOKENS else text)
    return pd.Series(out, index=values.index, dtype=object)


# ---------------------------------------------------------------- numeric parsing


def parse_numeric(values: pd.Series) -> tuple[pd.Series, list[str]]:
    """Parse numbers hidden behind currency, %, thousands commas or (negatives).

    Returns float values (NaN where unparseable) and the disguises found on
    values that did parse.
    """
    na = values.isna()
    text = values.astype(object).where(~na, "").map(str).str.strip()

    parens = text.str.match(r"^\(.*\)$")
    text = text.where(~parens, text.str[1:-1]).str.strip()
    currency = text.str.contains(f"[{re.escape(CURRENCY_SYMBOLS)}]", regex=True)
    text = text.str.replace(f"[{re.escape(CURRENCY_SYMBOLS)}]", "", regex=True).str.strip()
    percent = text.str.endswith("%")
    text = text.where(~percent, text.str[:-1]).str.strip()
    thousands = text.str.match(_THOUSANDS_RE)
    text = text.where(~thousands, text.str.replace(",", "", regex=False))

    valid = text.map(lambda t: bool(_NUMBER_RE.match(t))) & ~na
    parsed = pd.to_numeric(text.where(valid, None), errors="coerce").astype(float)
    parsed = parsed.where(~parens, -parsed.abs())
    parsed.index = values.index

    ok = parsed.notna()
    flags = [
        ("currency", currency),
        ("percent", percent),
        ("thousands_sep", thousands),
        ("parens_negative", parens),
    ]
    found = [name for name, mask in flags if bool((mask & ok).any())]
    return parsed, found


# ---------------------------------------------------------------- date parsing


@dataclass
class _DateParse:
    parsed: pd.Series
    dayfirst: bool | None
    ambiguous: bool
    formats: list[str]
    conflict: bool = False


def _dmy_order(parts: pd.DataFrame) -> tuple[bool, bool, bool]:
    """Decide (dayfirst, ambiguous, conflict) from the first and second numeric parts.

    When some dates only work day first and others only month first, the bigger
    group wins and the column is flagged as ambiguous.
    """
    first_big = int((parts[0].astype(int) > 12).sum())
    second_big = int((parts[2].astype(int) > 12).sum())
    if first_big and second_big:
        return first_big >= second_big, True, True
    if first_big:
        return True, False, False
    if second_big:
        return False, False, False
    return True, True, False


def _time_formats(tsep: pd.Series, time: pd.Series) -> pd.Series:
    """Time format per row, like ' %H:%M'. Empty when there is no time part."""
    clock = time.str.count(":").map({1: "%H:%M", 2: "%H:%M:%S"})
    return (tsep + clock).fillna("")


def _dmy_formats(parts: pd.DataFrame, dayfirst: bool) -> pd.Series:
    """Format string per row for D/M/Y (or M/D/Y) parts from _DMY_RE."""
    sep = parts[1]
    day_month = ("%d" + sep + "%m") if dayfirst else ("%m" + sep + "%d")
    year = parts[3].str.len().map({4: "%Y", 2: "%y"})
    return day_month + sep + year + _time_formats(parts[4], parts[5])


def _to_datetime(values: pd.Series, fmt: str) -> pd.Series:
    """Parse with one format. Dates outside the ns range (like 9999-12-31) become NaT."""
    try:
        if fmt == "month name":
            parsed = pd.to_datetime(
                values, format="mixed", dayfirst=True, errors="coerce", utc=True
            )
        else:
            parsed = pd.to_datetime(values, format=fmt, errors="coerce", utc=True)
        parsed = parsed.dt.tz_localize(None)
        in_range = (parsed >= pd.Timestamp.min) & (parsed <= pd.Timestamp.max)
        return parsed.where(in_range).astype("datetime64[ns]")
    except (ValueError, TypeError, AttributeError, OverflowError):
        return pd.Series(pd.NaT, index=values.index, dtype="datetime64[ns]")


def _fill(out: pd.Series, text: pd.Series, mask: pd.Series, fmt: str) -> tuple[pd.Series, bool]:
    """Fill still-empty slots of out with text[mask] parsed by fmt. Returns (out, any parsed)."""
    if not bool(mask.any()):
        return out, False
    got = _to_datetime(text[mask], fmt)
    return out.fillna(got), bool(got.notna().any())


def _fill_by_format(
    out: pd.Series, text: pd.Series, fmts: pd.Series, formats: list[str]
) -> pd.Series:
    """Parse each group of rows sharing a format string. fmts is indexed like a subset of text."""
    for fmt in fmts.unique():
        mask = (fmts == fmt).reindex(text.index, fill_value=False)
        out, ok = _fill(out, text, mask, fmt)
        if ok and fmt not in formats:
            formats.append(fmt)
    return out


def _parse_dmy(out: pd.Series, text: pd.Series, formats: list[str]) -> _DateParse:
    parts = text.str.extract(_DMY_RE)
    # 2-digit years with dots look too much like version numbers ("1.2.30")
    dmy = parts[0].notna() & ~((parts[1] == ".") & (parts[3].str.len() == 2))
    if not bool(dmy.any()):
        return _DateParse(out, None, False, formats)
    parts = parts[dmy]
    dayfirst, ambiguous, conflict = _dmy_order(parts)
    out = _fill_by_format(out, text, _dmy_formats(parts, dayfirst), formats)
    if conflict:
        out = _fill_by_format(out, text, _dmy_formats(parts, not dayfirst), formats)
    return _DateParse(out, dayfirst, ambiguous, formats, conflict)


def _parse_dates_detail(values: pd.Series) -> _DateParse:
    text = values.astype(object).where(values.notna(), "").map(str).str.strip()
    text = text.reset_index(drop=True)
    out = pd.Series(pd.NaT, index=text.index, dtype="datetime64[ns]")
    formats: list[str] = []

    for mask, fmt in [
        (text.str.match(_ISO_DATE_RE), "%Y-%m-%d"),
        (text.str.match(_ISO_DATETIME_RE), "ISO8601"),
    ]:
        out, ok = _fill(out, text, mask, fmt)
        if ok:
            formats.append(fmt)
    ymd = text.str.extract(_YMD_SLASH_RE)
    ymd = ymd[text.str.match(_YMD_SLASH_RE)]
    out = _fill_by_format(out, text, "%Y/%m/%d" + _time_formats(ymd[0], ymd[1]), formats)

    result = _parse_dmy(out, text, formats)
    named = result.parsed.isna() & text.str.contains(_MONTH_NAME_RE) & text.str.contains(_YEAR_RE)
    out, ok = _fill(result.parsed, text, named, "month name")
    if ok:
        result.formats.append("month name")
    result.parsed = out.set_axis(values.index)
    return result


def parse_dates(values: pd.Series) -> tuple[pd.Series, bool | None, bool]:
    """Parse ISO and D/M/Y or M/D/Y dates with explicit formats.

    Returns (parsed datetimes with NaT where unparseable, dayfirst, ambiguous).
    dayfirst is None when only ISO style dates were seen.
    """
    result = _parse_dates_detail(values)
    return result.parsed, result.dayfirst, result.ambiguous


# ---------------------------------------------------------------- small helpers


def _pct(part: float, whole: float) -> float:
    return 100.0 * part / whole if whole else 0.0


def _fmt_pct(value: float) -> str:
    return f"{value:.0f}%"


def _quote_list(items: list[str], limit: int = 3) -> str:
    shown = ", ".join(f"'{i}'" for i in items[:limit])
    return shown + (" and more" if len(items) > limit else "")


def _boolean_key(value: str) -> str:
    low = value.lower()
    try:
        num = float(low)
    except ValueError:
        return low
    if num in (0.0, 1.0):
        return str(int(num))
    return low


def _is_boolean(present: pd.Series) -> bool:
    keys = frozenset(_boolean_key(v) for v in present.unique())
    return keys in BOOLEAN_PAIRS


def _case_variants(present: pd.Series) -> dict[str, list[str]]:
    groups: dict[str, set[str]] = {}
    for v in present.unique():
        groups.setdefault(v.lower(), set()).add(v)
    return {k: sorted(v) for k, v in groups.items() if len(v) > 1}


def _top_values(present: pd.Series) -> dict[str, int]:
    counts = present.value_counts().head(MAX_TOP_VALUES)
    return {str(k): int(v) for k, v in counts.items()}


def _sample_values(present: pd.Series) -> list[str]:
    return [str(v) for v in present.unique()[:MAX_SAMPLES]]


def _avg_words(present: pd.Series) -> float:
    return float(present.str.split().str.len().mean()) if len(present) else 0.0


def _float_or_none(value: Any) -> float | None:
    f = float(value)
    return None if np.isnan(f) else f


def numeric_stats(values: pd.Series) -> dict[str, float | None]:
    """Summary stats on parsed, non-missing numbers."""
    return {
        "min": _float_or_none(values.min()),
        "max": _float_or_none(values.max()),
        "mean": _float_or_none(values.mean()),
        "median": _float_or_none(values.median()),
        "std": _float_or_none(values.std()),
        "skew": _float_or_none(values.skew()),
    }


def iqr_outliers(values: pd.Series) -> tuple[int, float, tuple[float, float]]:
    """Count values outside the 1.5 * IQR fences. Returns (count, pct, bounds)."""
    q1, q3 = float(values.quantile(0.25)), float(values.quantile(0.75))
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    n = int(((values < lower) | (values > upper)).sum())
    return n, _pct(n, len(values)), (lower, upper)


def _is_sequential_ints(nums: pd.Series) -> bool:
    if len(nums) < ID_MIN_VALUES or not bool((nums == np.floor(nums)).all()):
        return False
    n_unique = nums.nunique()
    if n_unique / len(nums) < ID_UNIQUE_RATIO:
        return False
    span = float(nums.max() - nums.min()) + 1
    return span <= 1.1 * n_unique


def _has_id_name(name: str) -> bool:
    return bool(_ID_NAME_RE.search(name) or _CAMEL_ID_RE.search(name))


def _all_unique_ints(present: pd.Series, n_unique: int) -> bool:
    if len(present) < ID_MIN_VALUES or n_unique / len(present) < ID_UNIQUE_RATIO:
        return False
    return bool(present.str.fullmatch(_INT_RE).all())


def _looks_like_years(nums: pd.Series) -> bool:
    return bool(nums.between(*YEAR_RANGE).all())


def _int_id_reason(name: str, present: pd.Series, nums: pd.Series, n_unique: int) -> str | None:
    """Reason to call an all-integer column an ID, or None. Year-like columns never are."""
    if not _all_unique_ints(present, n_unique) or _looks_like_years(nums):
        return None
    if _has_id_name(name):
        return f"The name '{name}' plus all-unique whole numbers says this is an ID."
    if _is_sequential_ints(nums) and (nums.min() in (0, 1) or n_unique >= SEQ_ID_MIN_VALUES):
        return "Values are unique whole numbers counting up, so this looks like a row number or ID."
    return None


def _disguise_phrase(disguise: list[str]) -> str:
    parts = {
        "currency": "currency symbols",
        "thousands_sep": "thousands commas",
        "percent": "% signs",
        "parens_negative": "brackets around negatives",
    }
    removed = [parts[d] for d in disguise]
    return " after removing " + " and ".join(removed) if removed else ""


# ---------------------------------------------------------------- per-type builders


def _constant_or_presence(base: ColumnProfile, present: pd.Series, n_rows: int) -> ColumnProfile:
    if len(present) == 0:
        base.dtype = "constant"
        base.all_missing = True
        base.reasons = (
            ["There are no rows, so there's nothing to profile."]
            if n_rows == 0
            else ["Every value is missing, so this column carries no information."]
        )
        return base
    value = str(present.iloc[0])
    if base.missing_pct < PRESENCE_MISSING_PCT:
        base.dtype = "constant"
        base.reasons = [f"Only one value ('{value}') shows up, so it can't help a model."]
        return base
    base.dtype = "categorical"
    base.top_values = _top_values(present)
    base.reasons = [
        f"Only one value ('{value}') shows up, but {_fmt_pct(base.missing_pct)} of rows are "
        "missing, so present vs missing is the signal."
    ]
    return base


def _boolean(base: ColumnProfile, present: pd.Series) -> ColumnProfile:
    base.dtype = "boolean"
    base.top_values = _top_values(present)
    base.case_variants = _case_variants(present)
    pair = " / ".join(sorted({v.lower() for v in present.unique()}, reverse=True))
    base.reasons = [f"Has exactly two values ({pair}), which looks like a yes/no flag."]
    return base


def _numeric(
    base: ColumnProfile, nums: pd.Series, rate: float, disguise: list[str]
) -> ColumnProfile:
    base.dtype = "numeric"
    base.parse_rate = rate
    base.disguise = disguise
    base.stats = numeric_stats(nums)
    base.n_outliers, base.outlier_pct, base.outlier_bounds = iqr_outliers(nums)
    kind = "money" if "currency" in disguise else "numbers"
    base.reasons = [
        f"{_fmt_pct(100 * rate)} of values parse as {kind}{_disguise_phrase(disguise)}."
    ]
    if base.n_outliers:
        base.reasons.append(
            f"{base.n_outliers} values ({base.outlier_pct:.1f}%) sit outside the usual range "
            "(1.5x IQR rule)."
        )
    return base


def _datetime(base: ColumnProfile, dates: _DateParse, rate: float) -> ColumnProfile:
    base.dtype = "datetime"
    base.parse_rate = rate
    base.dayfirst = dates.dayfirst
    base.date_ambiguous = dates.ambiguous
    base.date_format = "mixed" if len(dates.formats) > 1 else dates.formats[0]
    valid = dates.parsed.dropna()
    base.date_range = (valid.min().date().isoformat(), valid.max().date().isoformat())
    base.reasons = [f"{_fmt_pct(100 * rate)} of values parse as dates."]
    if base.date_format == "mixed":
        base.reasons.append("Dates are written in more than one format.")
    if dates.conflict:
        order = "day first" if dates.dayfirst else "month first"
        base.reasons.append(
            "Some dates only make sense day first and others only month first, so both "
            f"styles are mixed in here. Went with {order} where it was unclear. "
            "Worth double checking."
        )
    elif dates.ambiguous:
        base.reasons.append(
            "No part of any date goes above 12, so day/month order is a guess "
            "(went with day first). Worth double checking."
        )
    elif dates.dayfirst is True:
        base.reasons.append("Some dates have a day above 12 up front, so they read as day/month.")
    elif dates.dayfirst is False:
        base.reasons.append(
            "Some dates have a day above 12 in the middle, so they read as month/day."
        )
    return base


def _id(base: ColumnProfile, reason: str) -> ColumnProfile:
    base.dtype = "id"
    base.reasons = [reason]
    return base


def _text(base: ColumnProfile, reason: str) -> ColumnProfile:
    base.dtype = "text"
    base.reasons = [reason]
    return base


def _categorical(base: ColumnProfile, present: pd.Series, num_rate: float) -> ColumnProfile:
    base.dtype = "categorical"
    base.top_values = _top_values(present)
    base.case_variants = _case_variants(present)
    base.reasons = [f"{base.n_unique} distinct values, so it's treated as categories."]
    if 0 < num_rate < PARSE_THRESHOLD:
        base.reasons.append(
            f"Only {_fmt_pct(100 * num_rate)} of values parse as numbers, so it stays categorical."
        )
    return base


def _add_case_reason(col: ColumnProfile) -> None:
    for spellings in col.case_variants.values():
        col.reasons.append(
            f"Some values differ only by case ({_quote_list(spellings)}). "
            "They're probably the same thing."
        )


# ---------------------------------------------------------------- column profile


def _string_id_reason(present: pd.Series, unique_ratio: float, avg_words: float) -> str | None:
    if len(present) < ID_MIN_VALUES or unique_ratio < ID_UNIQUE_RATIO or avg_words >= 2:
        return None
    if bool(present.str.contains(r"\s", regex=True).any()):
        return None
    return (
        f"{_fmt_pct(100 * unique_ratio)} of values are unique short codes like "
        f"'{present.iloc[0]}', so this looks like an ID."
    )


def _detect(base: ColumnProfile, present: pd.Series, n_rows: int) -> ColumnProfile:
    n_present = len(present)
    if base.n_unique <= 1:
        return _constant_or_presence(base, present, n_rows)
    if _is_boolean(present):
        return _boolean(base, present)

    nums, disguise = parse_numeric(present)
    num_rate = float(nums.notna().mean())
    if num_rate >= PARSE_THRESHOLD:
        valid = nums.dropna()
        id_reason = _int_id_reason(base.name, present, valid, base.n_unique)
        if id_reason:
            return _id(base, id_reason)
        col = _numeric(base, valid, num_rate, disguise)
        if _all_unique_ints(present, base.n_unique) and not _looks_like_years(valid):
            col.reasons.append("Every value is a different whole number, so it could be an ID.")
        return col

    dates = _parse_dates_detail(present)
    date_rate = float(dates.parsed.notna().mean())
    if date_rate >= PARSE_THRESHOLD:
        return _datetime(base, dates, date_rate)

    unique_ratio = base.n_unique / n_present
    avg_words = _avg_words(present)
    id_reason = _string_id_reason(present, unique_ratio, avg_words)
    if id_reason:
        return _id(base, id_reason)
    if avg_words >= TEXT_MIN_WORDS:
        return _text(base, f"Values average {avg_words:.1f} words, so this looks like free text.")
    if base.n_unique > TEXT_MIN_UNIQUE and unique_ratio > TEXT_UNIQUE_RATIO:
        return _text(
            base,
            f"{base.n_unique} different values ({_fmt_pct(100 * unique_ratio)} unique), "
            "too many to treat as categories.",
        )
    return _categorical(base, present, num_rate)


def profile_column(name: str, values: pd.Series) -> ColumnProfile:
    """Profile one column. Read-only on the input."""
    n_rows = len(values)
    if isinstance(values.dtype, pd.DatetimeTZDtype):
        values = values.dt.tz_localize(None)  # keep local wall time; returns a new series
    clean = to_clean_strings(values)
    present = clean.dropna().astype(object)
    base = ColumnProfile(
        name=name,
        dtype="categorical",
        missing_pct=_pct(n_rows - len(present), n_rows),
        n_unique=int(present.nunique()),
        sample_values=_sample_values(present),
    )
    col = _detect(base, present.astype(str) if len(present) else present, n_rows)
    _add_case_reason(col)
    if col.missing_pct > 0 and not col.all_missing:
        col.reasons.append(f"{col.missing_pct:.1f}% of values are missing.")
    return col


def _count_duplicate_rows(df: pd.DataFrame) -> int:
    if df.shape[0] == 0 or df.shape[1] == 0:
        return 0
    try:
        return int(df.set_axis(range(df.shape[1]), axis=1).duplicated().sum())
    except TypeError:  # unhashable cells such as lists
        return int(df.astype(str).set_axis(range(df.shape[1]), axis=1).duplicated().sum())


def profile(df: pd.DataFrame) -> DatasetProfile:
    """Profile a DataFrame: column types, missingness, cardinality, outliers, disguised types."""
    columns = [profile_column(str(df.columns[i]), df.iloc[:, i]) for i in range(df.shape[1])]
    return DatasetProfile(
        n_rows=int(df.shape[0]),
        n_cols=int(df.shape[1]),
        n_duplicate_rows=_count_duplicate_rows(df),
        columns=columns,
    )
