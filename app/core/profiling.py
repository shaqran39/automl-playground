"""Phase 1: dataset profiling. Detects column types and per-column stats.

Profiling is read-only: nothing here is fitted or learned, and the input frame
is never mutated. Every type decision comes with plain-English reasons.
"""

from __future__ import annotations

import copy
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import IO, Any, Literal

import numpy as np
import pandas as pd

from app.core import loading

ColumnType = Literal[
    "numeric", "categorical", "boolean", "datetime", "time", "text", "id", "constant"
]

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
_LEADING_ZERO_RE = r"0\d{2,}"  # 3+ chars, so padded hours/months like "05" stay numeric
LEADING_ZERO_MIN_COUNT = 2
_ID_NAME_RE = re.compile(r"(^|[_\s.-])(id|key|uuid|guid)$|^(id|key|uuid|guid)[_\s.-]", re.I)
_CAMEL_ID_RE = re.compile(r"[a-z0-9](Id|ID|Key|Uuid)$")
_TIME_RE = r"^(\d{1,2}):(\d{2})(?::(\d{2}))?\s*([AaPp][Mm])?$"
YEAR_RANGE = (1900, 2100)
SEQ_ID_MIN_VALUES = 50
TYPE_SAMPLE_ROWS = 5000  # rows looked at before trying the number, time and date parsers
GATE_SHARE = 0.5
SAMPLE_SEED = 0


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
    has_time: bool = False
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


def read_csv_raw(path: str | Path | IO[str] | IO[bytes]) -> pd.DataFrame:
    """Read a CSV with every cell kept as a raw string (blanks stay "").

    Separator, encoding, junk lines and broken rows are handled in app.core.loading.
    """
    return loading.read_csv_raw(path)


# ---------------------------------------------------------------- normalising


def _is_missing_scalar(value: Any) -> bool:
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _clean_value(value: Any) -> str | None:
    if value is None or _is_missing_scalar(value):
        return None
    text = str(value).strip()
    return None if text.lower() in MISSING_TOKENS else text


# ---------------------------------------------------------------- distinct values
#
# Big files repeat the same few values a lot, so every per-value check runs
# once per distinct value and is mapped back to the rows. Results are exactly
# what a row-by-row loop would give, row counts are used as weights where a
# rule counts rows.


def _all_strings(values: pd.Series) -> bool:
    if isinstance(values.dtype, pd.StringDtype):
        return True
    return values.dtype == object and pd.api.types.infer_dtype(values, skipna=True) in (
        "string",
        "empty",
    )


def factorize(values: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """(code per row, distinct values in first-seen order). Missing gets code -1.

    Floats are told apart by their bits, so -0.0 and 0.0 stay different values.
    Mixed object columns (1, 1.0 and True hash the same) aren't merged at all.
    """
    if values.dtype == np.float64:
        missing = values.isna().to_numpy()
        bits = np.ascontiguousarray(values.to_numpy()[~missing]).view(np.int64)
        present_codes, bit_uniques = pd.factorize(bits)
        codes = np.full(len(values), -1, dtype=np.int64)
        codes[~missing] = present_codes
        return codes, np.asarray(bit_uniques, dtype=np.int64).view(np.float64).astype(object)
    if values.dtype == object and not _all_strings(values):
        arr = values.to_numpy(dtype=object)
        missing = np.array([_is_missing_scalar(v) for v in arr], dtype=bool)
        return np.where(missing, -1, np.arange(len(arr))), arr
    codes, uniques = pd.factorize(values, use_na_sentinel=True)
    return codes, np.asarray(uniques, dtype=object)


_MAX_TOKEN_LEN = max(len(t) for t in MISSING_TOKENS)


def _clean_str(value: str) -> str | None:
    text = value.strip()
    return None if len(text) <= _MAX_TOKEN_LEN and text.lower() in MISSING_TOKENS else text


def _clean_uniques(uniques: np.ndarray) -> np.ndarray:
    """_clean_value on each distinct value. Strings take a shortcut (same result)."""
    if all(type(u) is str for u in uniques):
        return np.array([_clean_str(u) for u in uniques], dtype=object)
    return np.array([_clean_value(u) for u in uniques], dtype=object)


@dataclass
class Distinct:
    """A column's present values as distinct cleaned strings plus their row counts."""

    index: pd.Index  # the column's index
    codes: np.ndarray  # per row, position in uniques, -1 when missing
    uniques: pd.Series  # distinct cleaned strings in first-seen order (object, RangeIndex)
    counts: np.ndarray  # rows per distinct value
    raw_codes: np.ndarray | None = None  # the same, before trimming and null tokens
    raw_uniques: np.ndarray | None = None
    raw_cleaned: np.ndarray | None = None  # _clean_value of each raw distinct value

    @cached_property
    def objects(self) -> np.ndarray:
        """uniques as a plain object array. Looping over it is much faster than the Series."""
        return self.uniques.to_numpy(dtype=object)

    def at(self, positions: np.ndarray, index: pd.Index) -> Distinct:
        """The same column cut down to some rows (by position), counts redone."""
        codes = self.codes[positions]
        counts = np.bincount(codes[codes >= 0], minlength=self.n_unique)
        raw = None if self.raw_codes is None else self.raw_codes[positions]
        cut = Distinct(index, codes, self.uniques, counts, raw, self.raw_uniques, self.raw_cleaned)
        cut.__dict__["objects"] = self.objects  # same uniques, so share the cached array
        return cut

    def raw_counts(self) -> np.ndarray:
        assert self.raw_codes is not None and self.raw_uniques is not None
        present = self.raw_codes[self.raw_codes >= 0]
        return np.bincount(present, minlength=len(self.raw_uniques))

    def raw_to_rows(self, per_value: Any, missing: Any, dtype: Any = object) -> pd.Series:
        """Per raw distinct value results spread over the rows."""
        assert self.raw_codes is not None
        return Distinct(self.index, self.raw_codes, self.uniques, self.counts).to_rows(
            per_value, missing, dtype
        )

    @property
    def n_present(self) -> int:
        return int(self.counts.sum())

    @property
    def n_unique(self) -> int:
        return len(self.uniques)

    def share(self, mask: Any) -> float:
        """Share of present rows where the per-value mask is True."""
        n = self.n_present
        return float((self.counts * np.asarray(mask, dtype=bool)).sum()) / n if n else 0.0

    def total(self, mask: Any) -> int:
        return int((self.counts * np.asarray(mask, dtype=bool)).sum())

    def to_rows(self, per_value: Any, missing: Any, dtype: Any = object) -> pd.Series:
        """Per-value results spread back over every row. Missing rows get missing."""
        arr = np.asarray(per_value, dtype=dtype)
        out = np.empty(len(self.codes), dtype=arr.dtype) if len(arr) else None
        if out is None:
            return pd.Series([missing] * len(self.codes), index=self.index, dtype=dtype)
        ok = self.codes >= 0
        out[ok] = arr[self.codes[ok]]
        out[~ok] = missing
        return pd.Series(out, index=self.index, dtype=dtype)

    def present_rows(self, per_value: Any, dtype: Any = object) -> pd.Series:
        """Per-value results for the present rows only, in row order."""
        ok = self.codes >= 0
        arr = np.asarray(per_value, dtype=dtype)
        return pd.Series(arr[self.codes[ok]], index=self.index[ok], dtype=dtype)

    def sample_weights(self, n: int) -> np.ndarray:
        """Row counts per distinct value in a fixed random sample of at most n present rows."""
        if self.n_present <= n:
            return self.counts
        rng = np.random.default_rng(SAMPLE_SEED)
        picked = rng.choice(self.codes[self.codes >= 0], size=n, replace=False)
        return np.bincount(picked, minlength=self.n_unique)


def _never_needs_trimming(values: pd.Series) -> bool:
    """Numbers, bools and tz-naive datetimes. Missing ones are already left out by factorize."""
    dtype = values.dtype
    return dtype in (np.float64, np.int64, np.bool_) or str(dtype) == "datetime64[ns]"


def distinct(values: pd.Series) -> Distinct:
    """Factorize once, clean each distinct value once, merge values that clean the same."""
    raw_codes, raw_uniques = factorize(values)
    if _never_needs_trimming(values):  # str() of a number or date has no spaces or tokens
        cleaned = np.array([str(u) for u in raw_uniques], dtype=object)
    else:
        cleaned = _clean_uniques(raw_uniques)
    merged, uniques = pd.factorize(pd.Series(cleaned, dtype=object), use_na_sentinel=True)
    if len(merged) == 0:  # every value missing
        codes = np.full(len(raw_codes), -1, dtype=np.int64)
    else:
        codes = np.where(raw_codes >= 0, merged[np.maximum(raw_codes, 0)], -1)
    counts = np.bincount(codes[codes >= 0], minlength=len(uniques))
    uniques = pd.Series(np.asarray(uniques, dtype=object))
    return Distinct(values.index, codes, uniques, counts, raw_codes, raw_uniques, cleaned)


def to_clean_strings(values: pd.Series) -> pd.Series:
    """Stripped string version of a column, with every kind of missing as None."""
    d = distinct(values)
    return d.to_rows(d.uniques.to_numpy(dtype=object), None)


# ---------------------------------------------------------------- numeric parsing


def parse_numeric(values: pd.Series) -> tuple[pd.Series, list[str]]:
    """Parse numbers hidden behind currency, %, thousands commas or (negatives).

    Returns float values (NaN where unparseable) and the disguises found on
    values that did parse.
    """
    codes, uniques = factorize(values)
    parsed, found = _parse_numeric_rows(pd.Series(uniques, dtype=object))
    return _spread(values.index, codes, parsed.to_numpy(), np.nan, np.float64), found


def _spread(
    index: pd.Index, codes: np.ndarray, per_value: Any, missing: Any, dtype: Any
) -> pd.Series:
    """Per-distinct-value results spread back over the rows."""
    return Distinct(index, codes, pd.Series(dtype=object), np.zeros(0)).to_rows(
        per_value, missing, dtype
    )


def _parse_numeric_rows(values: pd.Series) -> tuple[pd.Series, list[str]]:
    """parse_numeric one row at a time. Run it on distinct values, not on every row."""
    parsed, found, _ = _parse_numeric_detail(values)
    return parsed, found


def _parse_numeric_detail(
    values: pd.Series, trimmed: np.ndarray | None = None
) -> tuple[pd.Series, list[str], np.ndarray]:
    """(numbers, disguises found, which values were already plain numbers).

    Plain numbers like "12.5" skip the disguise steps (they'd leave them alone
    anyway), but every value still goes through one pd.to_numeric call, exactly
    as before, so results match to the last bit. trimmed, when given, is values
    already as trimmed strings with nothing missing (distinct cleaned values).
    """
    if trimmed is not None:
        na = np.zeros(len(trimmed), dtype=bool)
        text = trimmed
    else:
        na = values.isna().to_numpy()
        items = zip(values.to_numpy(dtype=object), na, strict=True)
        text = np.array(["" if m else str(v).strip() for v, m in items], dtype=object)
    plain = np.array([bool(_NUMBER_RE.match(t)) for t in text], dtype=bool) & ~na
    final = np.where(plain, text, None)
    flags = {k: np.zeros(len(text), dtype=bool) for k in ("cur", "pct", "thou", "par")}
    rest = ~plain & ~na
    if rest.any():
        sub = _strip_disguises(pd.Series(text[rest], dtype=object))
        final[rest] = sub["text"].where(sub["valid"], None).to_numpy(dtype=object)
        for k in flags:
            flags[k][rest] = sub[k].to_numpy(dtype=bool)
    parsed = pd.to_numeric(pd.Series(final, dtype=object), errors="coerce").astype(float)
    parsed = parsed.where(~flags["par"], -parsed.abs())
    parsed.index = values.index
    ok = parsed.notna().to_numpy()
    names = [("currency", "cur"), ("percent", "pct"), ("thousands_sep", "thou"),
             ("parens_negative", "par")]  # fmt: skip
    return parsed, [name for name, k in names if bool((flags[k] & ok).any())], plain


def _strip_disguises(text: pd.Series) -> pd.DataFrame:
    """Peel currency, %, thousands commas and (negatives) off already-trimmed text."""
    parens = text.str.match(r"^\(.*\)$")
    text = text.where(~parens, text.str[1:-1]).str.strip()
    currency = text.str.contains(f"[{re.escape(CURRENCY_SYMBOLS)}]", regex=True)
    text = text.str.replace(f"[{re.escape(CURRENCY_SYMBOLS)}]", "", regex=True).str.strip()
    percent = text.str.endswith("%")
    text = text.where(~percent, text.str[:-1]).str.strip()
    thousands = text.str.match(_THOUSANDS_RE)
    text = text.where(~thousands, text.str.replace(",", "", regex=False))
    valid = text.map(lambda t: bool(_NUMBER_RE.match(t)))
    return pd.DataFrame(
        {
            "text": text,
            "valid": valid,
            "cur": currency,
            "pct": percent,
            "thou": thousands,
            "par": parens,
        }  # fmt: skip
    )


# ---------------------------------------------------------------- date parsing


@dataclass
class _DateParse:
    parsed: pd.Series
    dayfirst: bool | None
    ambiguous: bool
    formats: list[str]
    conflict: bool = False


def _dmy_order(parts: pd.DataFrame, weights: pd.Series) -> tuple[bool, bool, bool]:
    """Decide (dayfirst, ambiguous, conflict) from the first and second numeric parts.

    When some dates only work day first and others only month first, the bigger
    group wins and the column is flagged as ambiguous. weights is rows per value.
    """
    first_big = int(weights[parts[0].astype(int) > 12].sum())
    second_big = int(weights[parts[2].astype(int) > 12].sum())
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


def _parse_dmy(
    out: pd.Series, text: pd.Series, formats: list[str], weights: pd.Series
) -> _DateParse:
    parts = text.str.extract(_DMY_RE)
    # 2-digit years with dots look too much like version numbers ("1.2.30")
    dmy = parts[0].notna() & ~((parts[1] == ".") & (parts[3].str.len() == 2))
    if not bool(dmy.any()):
        return _DateParse(out, None, False, formats)
    parts = parts[dmy]
    dayfirst, ambiguous, conflict = _dmy_order(parts, weights[parts.index])
    out = _fill_by_format(out, text, _dmy_formats(parts, dayfirst), formats)
    if conflict:
        out = _fill_by_format(out, text, _dmy_formats(parts, not dayfirst), formats)
    return _DateParse(out, dayfirst, ambiguous, formats, conflict)


def _parse_dates_detail(values: pd.Series, weights: Any = None) -> _DateParse:
    """Parse a column of date strings. weights is rows per value, when values are distinct."""
    text = values.astype(object).where(values.notna(), "").map(str).str.strip()
    text = text.reset_index(drop=True)
    w = pd.Series(1 if weights is None else np.asarray(weights), index=text.index)
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

    result = _parse_dmy(out, text, formats, w)
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
    d = distinct(values)
    result = _parse_dates_detail(d.uniques, d.counts)
    parsed = d.to_rows(result.parsed.to_numpy(), np.datetime64("NaT"), "datetime64[ns]")
    return parsed, result.dayfirst, result.ambiguous


# ---------------------------------------------------------------- time parsing


def _clock(hour: int, minute: int, second: int, ampm: str) -> str | None:
    """ "HH:MM:SS" in 24h time, or None if the parts aren't a real time of day."""
    if minute > 59 or second > 59:
        return None
    if ampm:
        if not 1 <= hour <= 12:
            return None
        hour = hour % 12 + (12 if ampm == "pm" else 0)
    elif hour > 23:
        return None
    return f"{hour:02d}:{minute:02d}:{second:02d}"


def _parse_time(value: Any) -> str | None:
    if value is None or _is_missing_scalar(value):
        return None
    match = re.match(_TIME_RE, str(value).strip())
    if not match:
        return None
    hour, minute, second, ampm = match.groups()
    return _clock(int(hour), int(minute), int(second or 0), (ampm or "").lower())


def parse_times(values: pd.Series) -> pd.Series:
    """Times of day like "14:30", "9:05:10" or "2:30 PM" as "HH:MM:SS" (24h).

    None where the value is missing or isn't a real time. Never mutates the input.
    """
    codes, uniques = factorize(values)
    return _spread(values.index, codes, [_parse_time(v) for v in uniques], None, object)


class ParsedColumn:
    """One raw column, cleaned and parsed at most once, shared by profile() and clean().

    Numbers, times and dates are parsed on the distinct values the first time
    they're asked for. clean() reuses them when it sees exactly the same raw values.
    """

    def __init__(self, values: pd.Series) -> None:
        self.raw = values
        self.distinct = distinct(to_utc_naive(values))
        self._numbers: tuple[pd.Series, list[str]] | None = None
        self._plain: np.ndarray | None = None
        self._times: pd.Series | None = None
        self._dates: _DateParse | None = None

    def matches(self, values: pd.Series) -> bool:
        """True when values are the same raw values this was built from."""
        return self.raw.index.equals(values.index) and self.raw.equals(values)

    def numbers(self) -> tuple[pd.Series, list[str]]:
        """(number per distinct value, disguises found)."""
        if self._numbers is None:
            d = self.distinct
            parsed, found, self._plain = _parse_numeric_detail(d.uniques, d.objects)
            self._numbers = parsed, found
        return self._numbers

    def plain(self) -> np.ndarray:
        """Per distinct value, True when it's already a plain number like "12.5"."""
        self.numbers()
        assert self._plain is not None
        return self._plain

    def times(self) -> pd.Series:
        """ "HH:MM:SS" per distinct value, None where it isn't a time."""
        if self._times is None:
            self._times = pd.Series([_parse_time(v) for v in self.distinct.objects], dtype=object)
        return self._times

    def dates(self) -> _DateParse:
        """Date parse of the distinct values, with day/month order decided over all rows."""
        if self._dates is None:
            self._dates = _parse_dates_detail(self.distinct.uniques, self.distinct.counts)
        return self._dates

    def number_rows(self) -> pd.Series:
        return self.distinct.to_rows(self.numbers()[0].to_numpy(), np.nan, np.float64)

    def time_rows(self) -> pd.Series:
        return self.distinct.to_rows(self.times().to_numpy(dtype=object), None)

    def date_rows(self) -> pd.Series:
        parsed = self.dates().parsed.to_numpy()
        return self.distinct.to_rows(parsed, np.datetime64("NaT"), "datetime64[ns]")

    def clean_rows(self) -> pd.Series:
        return self.distinct.to_rows(self.distinct.uniques.to_numpy(dtype=object), None)


# ---------------------------------------------------------------- sample gates
#
# Parsing every distinct value of a free-text column as a number, time and date
# is the slow part on big files. A fixed sample decides whether a parser is
# worth trying at all. The gates are loose (half the sample) and accept more
# than the parsers do, so a column that would pass on the full data is never
# skipped. Every count and rate that ends up in the profile uses all rows.

_DATE_LIKE_RE = r"\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}"


def _looks_numeric(values: pd.Series) -> pd.Series:
    return _parse_numeric_rows(values)[0].notna()


def _looks_like_time(values: pd.Series) -> pd.Series:
    return values.str.match(_TIME_RE)


def _looks_like_date(values: pd.Series) -> pd.Series:
    named = values.str.contains(_MONTH_NAME_RE) & values.str.contains(_YEAR_RE)
    return values.str.contains(_DATE_LIKE_RE) | named


def _worth_trying(d: Distinct, check: Callable[[pd.Series], pd.Series]) -> bool:
    """True when at least GATE_SHARE of a sample of rows passes a cheap check."""
    weights = d.sample_weights(TYPE_SAMPLE_ROWS)
    seen = np.flatnonzero(weights)
    if seen.size == 0:
        return False
    passed = check(d.uniques.iloc[seen].reset_index(drop=True)).to_numpy(dtype=bool)
    return float((weights[seen] * passed).sum()) / float(weights[seen].sum()) >= GATE_SHARE


def to_utc_naive(values: pd.Series) -> pd.Series:
    """Timezone-aware datetimes moved to UTC with the zone dropped. Others pass through."""
    if isinstance(values.dtype, pd.DatetimeTZDtype):
        return values.dt.tz_convert("UTC").dt.tz_localize(None)
    return values


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


def _is_boolean(d: Distinct) -> bool:
    keys: set[str] = set()
    for v in d.objects:  # stops at the third key, so free text doesn't cost a full pass
        keys.add(_boolean_key(v))
        if len(keys) > 2:
            return False
    return frozenset(keys) in BOOLEAN_PAIRS


def _case_variants(d: Distinct) -> dict[str, list[str]]:
    groups: dict[str, set[str]] = {}
    for v in d.objects:
        groups.setdefault(v.lower(), set()).add(v)
    return {k: sorted(v) for k, v in groups.items() if len(v) > 1}


def _top_values(d: Distinct) -> dict[str, int]:
    """Same as value_counts(): stable sort by count over first-seen order."""
    counts = pd.Series(d.counts, index=pd.Index(d.uniques.to_numpy(dtype=object)))
    top = counts.sort_values(ascending=False, kind="stable").head(MAX_TOP_VALUES)
    return {str(k): int(v) for k, v in top.items()}


def _sample_values(d: Distinct) -> list[str]:
    return [str(v) for v in d.uniques.iloc[:MAX_SAMPLES]]


def _avg_words(d: Distinct) -> float:
    if not d.n_present:
        return 0.0
    words = np.array([len(v.split()) for v in d.objects], dtype=np.int64)
    return float((words * d.counts).sum()) / d.n_present


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


def _all_unique_ints(d: Distinct) -> bool:
    n = d.n_present
    if n < ID_MIN_VALUES or d.n_unique / n < ID_UNIQUE_RATIO:
        return False
    return bool(d.uniques.str.fullmatch(_INT_RE).all())


def _looks_like_years(nums: pd.Series) -> bool:
    return bool(nums.between(*YEAR_RANGE).all())


def _int_id_reason(name: str, d: Distinct, nums: pd.Series) -> str | None:
    """Reason to call an all-integer column an ID, or None. Year-like columns never are."""
    if not _all_unique_ints(d) or _looks_like_years(nums):
        return None
    if _has_id_name(name):
        return f"The name '{name}' plus all-unique whole numbers says this is an ID."
    if _is_sequential_ints(nums) and (nums.min() in (0, 1) or d.n_unique >= SEQ_ID_MIN_VALUES):
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


def _constant_or_presence(base: ColumnProfile, d: Distinct, n_rows: int) -> ColumnProfile:
    if d.n_present == 0:
        base.dtype = "constant"
        base.all_missing = True
        base.reasons = (
            ["There are no rows, so there's nothing to profile."]
            if n_rows == 0
            else ["Every value is missing, so this column carries no information."]
        )
        return base
    value = str(d.uniques.iloc[0])
    if base.missing_pct < PRESENCE_MISSING_PCT:
        base.dtype = "constant"
        base.reasons = [f"Only one value ('{value}') shows up, so it can't help a model."]
        return base
    base.dtype = "categorical"
    base.top_values = _top_values(d)
    base.reasons = [
        f"Only one value ('{value}') shows up, but {_fmt_pct(base.missing_pct)} of rows are "
        "missing, so present vs missing is the signal."
    ]
    return base


def _boolean(base: ColumnProfile, d: Distinct) -> ColumnProfile:
    base.dtype = "boolean"
    base.top_values = _top_values(d)
    base.case_variants = _case_variants(d)
    pair = " / ".join(sorted({v.lower() for v in d.objects}, reverse=True))
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


def _two_values(base: ColumnProfile, d: Distinct, nums: pd.Series) -> ColumnProfile:
    """Numbers with exactly two values that aren't 0/1, like sex coded 1 / 2."""
    col = _categorical(base, d, 1.0)
    pair = " / ".join(f"{v:g}" for v in sorted(nums.unique()))
    col.reasons = [f"Only two values ({pair}), so these are labels, not amounts."]
    return col


def _time(base: ColumnProfile, rate: float) -> ColumnProfile:
    base.dtype = "time"
    base.parse_rate = rate
    base.reasons = [f"{_fmt_pct(100 * rate)} of values parse as times of day (like '14:30')."]
    return base


def _datetime(base: ColumnProfile, dates: _DateParse, rate: float) -> ColumnProfile:
    base.dtype = "datetime"
    base.parse_rate = rate
    base.dayfirst = dates.dayfirst
    base.date_ambiguous = dates.ambiguous
    base.date_format = "mixed" if len(dates.formats) > 1 else dates.formats[0]
    valid = dates.parsed.dropna()
    base.date_range = (valid.min().date().isoformat(), valid.max().date().isoformat())
    base.has_time = bool((valid != valid.dt.normalize()).any())
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


def _categorical(base: ColumnProfile, d: Distinct, num_rate: float) -> ColumnProfile:
    base.dtype = "categorical"
    base.top_values = _top_values(d)
    base.case_variants = _case_variants(d)
    base.reasons = [f"{base.n_unique} distinct values, so it's treated as categories."]
    if 0 < num_rate < PARSE_THRESHOLD:
        base.reasons.append(
            f"Only {_fmt_pct(100 * num_rate)} of values parse as numbers, so it stays categorical."
        )
    return base


def _has_leading_zero_codes(d: Distinct) -> bool:
    """True for digit codes like '0800' or '02000'.

    Nearly every value must be a plain digit string, and at least two must start
    with 0, so one stray "007" in a price column doesn't flip it.
    """
    digits = d.uniques.str.fullmatch(r"\d+")
    leading_zero = d.uniques.str.fullmatch(_LEADING_ZERO_RE)
    return d.share(digits) >= PARSE_THRESHOLD and d.total(leading_zero) >= LEADING_ZERO_MIN_COUNT


def _code(base: ColumnProfile, d: Distinct) -> ColumnProfile:
    """Digit codes with leading zeros (postcodes, phone numbers). Never numeric."""
    n_present = d.n_present
    unique_ratio = base.n_unique / n_present
    if _has_id_name(base.name) and n_present >= ID_MIN_VALUES and unique_ratio >= ID_UNIQUE_RATIO:
        return _id(base, f"The name '{base.name}' plus all-unique codes says this is an ID.")
    id_reason = _string_id_reason(d, unique_ratio, _avg_words(d))
    if id_reason:
        return _id(base, id_reason)
    col = _categorical(base, d, 0.0)
    example = d.uniques[d.uniques.str.fullmatch(_LEADING_ZERO_RE)].iloc[0]
    col.reasons.insert(
        0,
        f"Some values start with 0 (like '{example}'), so these are codes, not amounts. "
        "Treating them as numbers would drop the zero.",
    )
    return col


def _add_case_reason(col: ColumnProfile) -> None:
    for spellings in col.case_variants.values():
        col.reasons.append(
            f"Some values differ only by case ({_quote_list(spellings)}). "
            "They're probably the same thing."
        )


# ---------------------------------------------------------------- column profile


def _string_id_reason(d: Distinct, unique_ratio: float, avg_words: float) -> str | None:
    if d.n_present < ID_MIN_VALUES or unique_ratio < ID_UNIQUE_RATIO or avg_words >= 2:
        return None
    if bool(d.uniques.str.contains(r"\s", regex=True).any()):
        return None
    return (
        f"{_fmt_pct(100 * unique_ratio)} of values are unique short codes like "
        f"'{d.uniques.iloc[0]}', so this looks like an ID."
    )


def _numeric_or_id(base: ColumnProfile, p: ParsedColumn, rate: float) -> ColumnProfile:
    d = p.distinct
    nums, disguise = p.numbers()
    valid = d.present_rows(nums.to_numpy(), np.float64).dropna()
    if valid.nunique() == 2:
        return _two_values(base, d, valid)
    id_reason = _int_id_reason(base.name, d, valid)
    if id_reason:
        return _id(base, id_reason)
    col = _numeric(base, valid, rate, disguise)
    if _all_unique_ints(d) and not _looks_like_years(valid):
        col.reasons.append("Every value is a different whole number, so it could be an ID.")
    return col


def _text_or_categorical(base: ColumnProfile, p: ParsedColumn) -> ColumnProfile:
    d = p.distinct
    unique_ratio = base.n_unique / d.n_present
    avg_words = _avg_words(d)
    id_reason = _string_id_reason(d, unique_ratio, avg_words)
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
    num_rate = d.share(p.numbers()[0].notna())  # exact, for the "only X% parse" reason
    return _categorical(base, d, num_rate)


def _detect(base: ColumnProfile, p: ParsedColumn, n_rows: int) -> ColumnProfile:
    d = p.distinct
    if base.n_unique <= 1:
        return _constant_or_presence(base, d, n_rows)
    if _is_boolean(d):
        return _boolean(base, d)
    if _has_leading_zero_codes(d):
        return _code(base, d)
    if _worth_trying(d, _looks_numeric):
        rate = d.share(p.numbers()[0].notna())
        if rate >= PARSE_THRESHOLD:
            return _numeric_or_id(base, p, rate)
    if _worth_trying(d, _looks_like_time):
        rate = d.share(p.times().notna())
        if rate >= PARSE_THRESHOLD:
            return _time(base, rate)
    if _worth_trying(d, _looks_like_date):
        rate = d.share(p.dates().parsed.notna())
        if rate >= PARSE_THRESHOLD:
            return _datetime(base, p.dates(), rate)
    return _text_or_categorical(base, p)


def _profile_parsed(name: str, p: ParsedColumn) -> ColumnProfile:
    d = p.distinct
    n_rows = len(d.codes)
    base = ColumnProfile(
        name=name,
        dtype="categorical",
        missing_pct=_pct(n_rows - d.n_present, n_rows),
        n_unique=d.n_unique,
        sample_values=_sample_values(d),
    )
    col = _detect(base, p, n_rows)
    _add_case_reason(col)
    if col.missing_pct > 0 and not col.all_missing:
        col.reasons.append(f"{col.missing_pct:.1f}% of values are missing.")
    return col


def profile_column(name: str, values: pd.Series) -> ColumnProfile:
    """Profile one column. Read-only on the input."""
    return _profile_parsed(name, ParsedColumn(values))


def _count_duplicate_rows(df: pd.DataFrame) -> int:
    if df.shape[0] == 0 or df.shape[1] == 0:
        return 0
    try:
        return int(df.set_axis(range(df.shape[1]), axis=1).duplicated().sum())
    except TypeError:  # unhashable cells such as lists
        return int(df.astype(str).set_axis(range(df.shape[1]), axis=1).duplicated().sum())


Progress = Callable[[int, int, str], None]


def _reusable(earlier: DatasetProfile | None) -> dict[str, tuple[ParsedColumn, ColumnProfile]]:
    cached = parsed_columns(earlier) if earlier is not None else None
    if earlier is None or cached is None:
        return {}
    return {c.name: (p, c) for c, p in zip(earlier.columns, cached, strict=True)}


def profile(
    df: pd.DataFrame, progress: Progress | None = None, reuse: DatasetProfile | None = None
) -> DatasetProfile:
    """Profile a DataFrame: column types, missingness, cardinality, outliers, disguised types.

    progress, if given, is called with (columns done, total, column name) as it goes.
    The parsed columns are kept on the result so clean() doesn't parse them again.
    reuse is an earlier profile: a column with exactly the same name, index and
    values (like a text column cleaning didn't touch) is copied over, not redone.
    """
    earlier = _reusable(reuse)
    columns: list[ColumnProfile] = []
    parsed: list[ParsedColumn] = []
    for i in range(df.shape[1]):
        name, values = str(df.columns[i]), df.iloc[:, i]
        if progress is not None:
            progress(i, df.shape[1], name)
        old = earlier.get(name)
        if old is not None and old[0].matches(values):
            parsed.append(old[0])
            columns.append(copy.deepcopy(old[1]))
            continue
        parsed.append(ParsedColumn(values))
        columns.append(_profile_parsed(name, parsed[-1]))
    prof = DatasetProfile(
        n_rows=int(df.shape[0]),
        n_cols=int(df.shape[1]),
        n_duplicate_rows=_count_duplicate_rows(df),
        columns=columns,
    )
    prof.parsed = parsed  # type: ignore[attr-defined]  # a cache, not a dataclass field
    return prof


def parsed_columns(prof: DatasetProfile) -> list[ParsedColumn] | None:
    """The parsed columns profile() kept, one per column, or None."""
    return getattr(prof, "parsed", None)
