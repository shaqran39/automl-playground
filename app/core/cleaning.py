"""Phase 2: contextual cleaning with a reversible cleaning log.

clean() runs before the train/test split, so it only does stateless, row-local
fixes: trimming, null tokens, type conversion, case merging, dedupe, blanking
values that can't be real (placeholder codes, impossible values, bad dates and
times) and dropping id/constant columns. Nothing here learns from the data (no
imputing, scaling, encoding, clipping or outlier fences). That all lives in the
model Pipeline.

Every step that changes something is logged as (column, action, reason). Plain
parses ("5.1" -> 5.1, "2024-01-05" -> a date, "9:05" -> "09:05:00") are always
applied and never logged. Values a parse can't keep (junk dates, 9999-12-31) are
missing either way, so they get a flag, never a step that pretends it can undo it.
The log is always built as if every step was on, so
switching steps off never changes what's listed, only whether each fix gets
applied.
"""

from __future__ import annotations

import datetime as dt
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd

from app.core.profiling import (
    MISSING_TOKENS,
    ColumnProfile,
    DatasetProfile,
    Distinct,
    ParsedColumn,
    _boolean_key,
    _clean_uniques,
    _disguise_phrase,
    _fmt_pct,
    _is_missing_scalar,
    _quote_list,
    factorize,
    parsed_columns,
    to_clean_strings,
    to_utc_naive,
)

StepKind = Literal["fix", "flag"]
Transform = Callable[[pd.DataFrame], pd.DataFrame]
ReferenceDate = pd.Timestamp | dt.date | str | None

HIGH_MISSING_PCT = 50.0
TRUE_KEYS: frozenset[str] = frozenset({"yes", "y", "true", "t", "1", "on"})
LOG_COLUMNS = ["step_id", "column", "action", "reason", "kind", "enabled", "rows_affected"]
TABLE = "*"

PLACEHOLDER_CODES: tuple[float, ...] = (-1, 99, 999, 9999, -99, -999, -9999)
PLACEHOLDER_MIN_COUNT = 2
IMPOSSIBLE_MAX_SHARE = 0.10  # a rule only runs when at most 10% of values break it
OLDEST_DATE = pd.Timestamp("1900-01-01")
PLACEHOLDER_DATES = (pd.Timestamp("1900-01-01"), pd.Timestamp("1970-01-01"))
FUTURE_OK_TOKENS: frozenset[str] = frozenset(
    {"due", "deadline", "scheduled", "appointment", "end", "expected", "next", "renewal", "start"}
)


@dataclass
class CleaningStep:
    step_id: str
    column: str
    action: str
    reason: str
    kind: StepKind = "fix"
    enabled: bool = True
    rows_affected: int = 0


@dataclass
class CleaningLog:
    steps: list[CleaningStep] = field(default_factory=list)
    row_ids: pd.Series | None = None

    def step(self, step_id: str) -> CleaningStep:
        """Return the step with this id. Raises KeyError if absent."""
        for s in self.steps:
            if s.step_id == step_id:
                return s
        raise KeyError(step_id)

    def to_frame(self) -> pd.DataFrame:
        """One row per step, for showing in the UI."""
        rows = [[getattr(s, c) for c in LOG_COLUMNS] for s in self.steps]
        return pd.DataFrame(rows, columns=LOG_COLUMNS)


@dataclass
class _Planned:
    """A logged step plus the change it makes.

    apply is None for flag steps. step is None for silent parses, which are
    always applied and never logged. Steps that blank values set blank instead
    of apply, so clean() can stretch the rows to cover duplicate copies.
    """

    step: CleaningStep | None
    apply: Transform | None = None
    blank: _Blank | None = None


def _step(column: str, action: str, reason: str, rows: int, kind: StepKind = "fix") -> CleaningStep:
    return CleaningStep(
        step_id=f"{action}:{column}",
        column=column,
        action=action,
        reason=reason,
        kind=kind,
        rows_affected=int(rows),
    )


# ---------------------------------------------------------------- small helpers


def _plural(n: int, word: str) -> str:
    return f"{n} {word}" if n == 1 else f"{n} {word}s"


def _is_text_column(values: pd.Series) -> bool:
    return values.dtype == object or isinstance(values.dtype, pd.StringDtype)


def _is_numeric(values: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(values) and not pd.api.types.is_bool_dtype(values)


def _is_datetime(values: pd.Series) -> bool:
    return pd.api.types.is_datetime64_any_dtype(values)


def _any_dtype(values: pd.Series) -> bool:
    return True


def _present_count(values: pd.Series) -> int:
    return int(to_clean_strings(values).notna().sum())


def _free_name(frame: pd.DataFrame, name: str) -> str:
    """name, or name_2, name_3... if a column already uses it."""
    if name not in frame.columns:
        return name
    i = 2
    while f"{name}_{i}" in frame.columns:
        i += 1
    return f"{name}_{i}"


def _replace_column(frame: pd.DataFrame, col: str, new: dict[str, pd.Series]) -> pd.DataFrame:
    """Swap col for the new columns, keeping them in col's spot."""
    pos = frame.columns.get_loc(col)
    middle = pd.DataFrame(new, index=frame.index)
    return pd.concat([frame.iloc[:, :pos], middle, frame.iloc[:, pos + 1 :]], axis=1)


def _drop(col: str) -> Transform:
    return lambda frame: frame.drop(columns=col)


def _set_column(col: str, fn: Callable[[pd.Series], pd.Series]) -> Transform:
    def apply(frame: pd.DataFrame) -> pd.DataFrame:
        return frame.assign(**{col: fn(frame[col])})

    return apply


@dataclass(frozen=True)
class _Blank:
    """Rows (index labels on the planned frame) of one column to set to missing.

    accepts checks the column still has the right type, so numbers left as
    text because to_numeric is off never get blanked.
    """

    column: str
    labels: pd.Index
    accepts: Callable[[pd.Series], bool] = _any_dtype


def _null_rows(blank: _Blank, labels: pd.Index) -> Transform:
    """Set exactly these rows of the column to missing.

    The rows are fixed when the step is planned, so a step never catches values
    that an earlier, switched-off step would have handled (a 999 age code stays
    999, it isn't picked up as age > 120).
    """

    def apply(frame: pd.DataFrame) -> pd.DataFrame:
        col = blank.column
        if col not in frame.columns or not blank.accepts(frame[col]):
            return frame
        return frame.assign(**{col: frame[col].mask(frame.index.isin(labels))})

    return apply


def duplicate_groups(frame: pd.DataFrame, codes: np.ndarray | None = None) -> pd.Series:
    """Group number per row, shared by rows that exactly repeat each other.

    Uses the same notion of "exact copy" as dedupe, so copies kept with dedupe
    off still get the same fixes as the row they repeat. codes, when given, is
    one integer per cell standing for its raw text, which groups much faster.
    """
    if frame.shape[1] == 0:
        return pd.Series(np.arange(len(frame)), index=frame.index)
    if codes is not None:
        ids = pd.DataFrame(codes).groupby(list(range(codes.shape[1])), sort=False).ngroup()
        return pd.Series(ids.to_numpy(), index=frame.index)
    positional = frame.set_axis(range(frame.shape[1]), axis=1)
    keys = list(positional.columns)
    try:
        groups = positional.groupby(keys, dropna=False, sort=False).ngroup()
    except TypeError:  # unhashable cells such as lists
        groups = positional.astype(str).groupby(keys, dropna=False, sort=False).ngroup()
    return pd.Series(groups.to_numpy(), index=frame.index)


def _with_copies(labels: pd.Index, groups: pd.Series) -> pd.Index:
    """labels plus every row that's an exact copy of one of them."""
    wanted = groups[groups.index.isin(labels)].unique()
    return groups.index[groups.isin(wanted)]


_CAMEL_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")


def name_tokens(name: str) -> list[str]:
    """Lowercase words in a column name: "patientAge" -> [patient, age], "usage" -> [usage]."""
    spaced = _CAMEL_RE.sub(" ", str(name))
    return [t.lower() for t in re.split(r"[^A-Za-z0-9]+", spaced) if t]


def _is_presence_case(col: ColumnProfile) -> bool:
    return col.dtype == "categorical" and col.n_unique == 1 and not col.all_missing


def _duplicate_mask(frame: pd.DataFrame) -> np.ndarray:
    """True for rows that exactly repeat an earlier row. Matches the profiler's count."""
    if frame.shape[0] == 0 or frame.shape[1] == 0:
        return np.zeros(frame.shape[0], dtype=bool)
    positional = frame.set_axis(range(frame.shape[1]), axis=1)
    try:
        return positional.duplicated().to_numpy()
    except TypeError:  # unhashable cells such as lists
        return positional.astype(str).duplicated().to_numpy()


class _Parsed:
    """The raw columns of one clean() call, parsed once.

    Taken straight from the profile when it saw exactly these raw values, so a
    column is never parsed twice. A view is only handed out while the column
    still holds its raw text (maybe trimmed, maybe with duplicate rows gone),
    which is true up to and including its own type conversion.
    """

    def __init__(self, work: pd.DataFrame, prof: DatasetProfile) -> None:
        self._work = work
        cached = parsed_columns(prof) or []
        self._cols: dict[str, ParsedColumn] = {
            col.name: p
            for col, p in zip(prof.columns, cached, strict=False)
            if p.matches(work[col.name])
        }

    @property
    def reused(self) -> int:
        return len(self._cols)

    def row_codes(self) -> np.ndarray | None:
        """One integer per cell for its raw text, when every column is text.

        Two rows are exact copies when all their codes match. Only for text,
        since numbers compare by value (-0.0 equals 0.0) but code by bits.
        """
        cols = list(self._work.columns)
        if not cols or not all(_is_text_column(self._work[c]) for c in cols):
            return None
        out = np.empty((len(self._work), len(cols)), dtype=np.int64)
        for j, name in enumerate(cols):
            p = self._cols.get(name) or ParsedColumn(self._work[name])
            self._cols[name] = p
            assert p.distinct.raw_codes is not None
            out[:, j] = p.distinct.raw_codes
        return out

    def view(self, name: str, values: pd.Series) -> tuple[ParsedColumn, Distinct] | None:
        if not _is_text_column(values) or name not in self._work.columns:
            return None
        pos = self._work.index.get_indexer(values.index)
        if (pos < 0).any():
            return None
        p = self._cols.get(name)
        if p is None:
            p = self._cols[name] = ParsedColumn(self._work[name])
        if len(pos) == len(self._work) and bool((pos == np.arange(len(pos))).all()):
            return p, p.distinct
        return p, p.distinct.at(pos, values.index)


def _view(ctx: _Parsed | None, name: str, values: pd.Series) -> tuple[ParsedColumn, Distinct]:
    """The parsed column for values: shared when possible, else parsed on the spot."""
    got = None if ctx is None else ctx.view(name, values)
    if got is not None:
        return got
    p = ParsedColumn(values)
    return p, p.distinct


def _reference_day(reference_date: ReferenceDate) -> pd.Timestamp:
    """The cutoff for future dates, at midnight UTC. None means today in UTC."""
    ref = pd.Timestamp.now(tz="UTC") if reference_date is None else pd.Timestamp(reference_date)
    if ref.tzinfo is not None:
        ref = ref.tz_convert("UTC").tz_localize(None)
    return ref.normalize()


# ---------------------------------------------------------------- value fixes


def _normalize_value(value: Any) -> Any:
    if isinstance(value, str):
        text = value.strip()
        return np.nan if text.lower() in MISSING_TOKENS else text
    return np.nan if _is_missing_scalar(value) else value


def _raw_distinct(ctx: _Parsed | None, name: str, values: pd.Series) -> Distinct:
    """Distinct raw values (before trimming), from the shared parse when possible."""
    got = None if ctx is None else ctx.view(name, values)
    if got is not None and got[1].raw_codes is not None:
        return got[1]
    codes, uniques = factorize(values)
    return Distinct(values.index, codes, pd.Series(dtype=object), np.zeros(0), codes, uniques)


def _raw_cleaned(d: Distinct) -> np.ndarray:
    return d.raw_cleaned if d.raw_cleaned is not None else _clean_uniques(d.raw_uniques)


def _normalized(ctx: _Parsed | None, name: str, values: pd.Series) -> pd.Series:
    """normalize_values, worked out once per distinct raw value."""
    d = _raw_distinct(ctx, name, values)
    cleaned = _raw_cleaned(d)
    if _is_str(d.raw_uniques).all():
        per_value = np.where(pd.isna(cleaned), np.nan, cleaned)
    else:
        per_value = [
            (np.nan if c is None else c) if isinstance(u, str) else _normalize_value(u)
            for u, c in zip(d.raw_uniques, cleaned, strict=True)
        ]
    out = d.raw_to_rows(per_value, np.nan)
    return pd.Series(out.to_numpy(), index=values.index, dtype=values.dtype, name=values.name)


def normalize_values(values: pd.Series) -> pd.Series:
    """Trim strings and turn null tokens (and None) into NaN. Non-strings are left alone."""
    return _normalized(None, str(values.name), values)


def _is_str(values: np.ndarray) -> np.ndarray:
    if pd.api.types.infer_dtype(values, skipna=False) == "string":
        return np.ones(len(values), dtype=bool)
    return np.fromiter((isinstance(u, str) for u in values), dtype=bool, count=len(values))


def _normalize_counts(d: Distinct) -> tuple[int, int]:
    """(values trimmed, values turned into missing) among the strings."""
    raw, cleaned = d.raw_uniques, _raw_cleaned(d)
    is_str = _is_str(raw)
    nulled = is_str & pd.isna(cleaned)
    trimmed = is_str & ~nulled & (cleaned != raw)
    counts = d.raw_counts()
    return int(counts[trimmed].sum()), int(counts[nulled].sum())


def _numbers(ctx: _Parsed | None, name: str, values: pd.Series) -> pd.Series:
    p, d = _view(ctx, name, values)
    out = d.to_rows(p.numbers()[0].to_numpy(), np.nan, np.float64)
    return out.rename(values.name)


def to_numbers(values: pd.Series) -> pd.Series:
    """float64 numbers with disguises removed. Leftovers become NaN."""
    return _numbers(None, str(values.name), values)


def _bool_value(value: str | None) -> float:
    if value is None:
        return np.nan
    return 1.0 if _boolean_key(value) in TRUE_KEYS else 0.0


def _bools(ctx: _Parsed | None, name: str, values: pd.Series) -> pd.Series:
    _, d = _view(ctx, name, values)
    out = d.to_rows([_bool_value(v) for v in d.objects], np.nan, np.float64)
    return out.rename(values.name)


def to_bools(values: pd.Series) -> pd.Series:
    """1.0 for the yes side, 0.0 for the no side, NaN when missing."""
    return _bools(None, str(values.name), values)


def _datetimes(ctx: _Parsed | None, name: str, values: pd.Series) -> pd.Series:
    values = to_utc_naive(values)
    if _is_datetime(values):
        fits = (values >= pd.Timestamp.min) & (values <= pd.Timestamp.max)
        return values.where(fits).astype("datetime64[ns]")
    p, d = _view(ctx, name, values)
    out = d.to_rows(p.dates().parsed.to_numpy(), np.datetime64("NaT"), "datetime64[ns]")
    return out.rename(values.name)


def to_datetimes(values: pd.Series) -> pd.Series:
    """datetime64[ns], tz-naive in UTC. NaT where a value didn't parse or won't fit in ns."""
    return _datetimes(None, str(values.name), values)


def _clean_text(ctx: _Parsed | None, name: str, values: pd.Series) -> pd.Series:
    _, d = _view(ctx, name, values)
    return d.to_rows(d.uniques.to_numpy(dtype=object), None)


def _time_rows(ctx: _Parsed | None, name: str, values: pd.Series) -> pd.Series:
    p, d = _view(ctx, name, values)
    return d.to_rows(p.times().to_numpy(dtype=object), None)


def _times(ctx: _Parsed | None, name: str, values: pd.Series) -> pd.Series:
    parsed = _time_rows(ctx, name, values)
    keep_raw = parsed.isna() & _clean_text(ctx, name, values).notna()
    return parsed.where(~keep_raw, values.astype(object)).rename(values.name)


def to_times(values: pd.Series) -> pd.Series:
    """ "HH:MM:SS" strings. Values that aren't real times keep their raw text."""
    return _times(None, str(values.name), values)


def _case_winner(spellings: list[str], counts: pd.Series) -> str:
    """All-uppercase spelling if there is one, else the most common, ties by sorted()."""
    ordered = sorted(spellings)
    upper = [s for s in ordered if s == s.upper() and s != s.lower()]
    if upper:
        return upper[0]
    return max(ordered, key=lambda s: (int(counts.get(s, 0)), -ordered.index(s)))


def case_mapping(col: ColumnProfile, values: pd.Series) -> dict[str, str]:
    """Spelling -> winning spelling, only for spellings that change."""
    counts = values.value_counts()
    mapping: dict[str, str] = {}
    for spellings in col.case_variants.values():
        winner = _case_winner(spellings, counts)
        mapping.update({s: winner for s in spellings if s != winner})
    return mapping


def apply_mapping(values: pd.Series, mapping: dict[str, str]) -> pd.Series:
    hit = values.isin(list(mapping))
    return values.where(~hit, values.map(mapping))


# ---------------------------------------------------------------- stages
#
# Each stage looks at the planned frame (every earlier step applied) and
# returns the steps that apply, in profile column order.


def _dedupe_stage(frame: pd.DataFrame, prof: DatasetProfile) -> list[_Planned]:
    if prof.n_duplicate_rows <= 0:
        return []
    mask = _duplicate_mask(frame)
    n = int(mask.sum())
    if n == 0:
        return []
    reason = (
        f"{_plural(n, 'row')} exactly repeated an earlier row, so the copies were dropped. "
        "The first one of each stays."
    )
    return [_Planned(_step(TABLE, "dedupe", reason, n), lambda f: f[~_duplicate_mask(f)])]


def _normalize_stage(frame: pd.DataFrame, prof: DatasetProfile, ctx: _Parsed) -> list[_Planned]:
    out: list[_Planned] = []
    for col in prof.columns:
        values = frame[col.name]
        if not _is_text_column(values):
            continue
        trimmed, nulled = _normalize_counts(_raw_distinct(ctx, col.name, values))
        if trimmed + nulled == 0:
            continue
        bits = []
        if trimmed:
            bits.append(f"trimmed extra spaces on {_plural(trimmed, 'value')}")
        if nulled:
            bits.append(
                f"turned {_plural(nulled, 'blank or placeholder value')} (like '' or 'N/A') "
                "into proper missing values"
            )
        text = " and ".join(bits)
        reason = text[0].upper() + text[1:] + "."
        apply = _set_column(col.name, lambda v, n=col.name: _normalized(ctx, n, v))
        out.append(_Planned(_step(col.name, "normalize", reason, trimmed + nulled), apply))
    return out


def _to_numeric_reason(disguised: int, leftover: int, disguise: list[str]) -> str:
    bits = []
    if disguised:
        bits.append(
            f"Stored {_plural(disguised, 'value')} as plain numbers{_disguise_phrase(disguise)}."
        )
    if leftover:
        bits.append(
            f"{_plural(leftover, 'value')} didn't parse as a number and "
            f"{'is' if leftover == 1 else 'are'} now missing."
        )
    return " ".join(bits)


def _to_numeric_stage(frame: pd.DataFrame, prof: DatasetProfile, ctx: _Parsed) -> list[_Planned]:
    """Numbers stored as text. Logged only when a disguise is removed or a value is lost."""
    out: list[_Planned] = []
    for col in prof.columns:
        values = frame[col.name]
        if col.dtype != "numeric" or pd.api.types.is_numeric_dtype(values):
            continue
        apply = _set_column(col.name, lambda v, n=col.name: _numbers(ctx, n, v))
        p, d = _view(ctx, col.name, values)
        parsed, disguise = p.numbers()  # one per distinct value
        plain = p.plain()
        disguised = d.total(parsed.notna().to_numpy() & ~plain)
        leftover = d.total(parsed.isna().to_numpy())
        if disguised + leftover == 0:
            out.append(_Planned(None, apply))
            continue
        reason = _to_numeric_reason(disguised, leftover, disguise)
        out.append(_Planned(_step(col.name, "to_numeric", reason, disguised + leftover), apply))
    return out


def _to_bool_stage(frame: pd.DataFrame, prof: DatasetProfile, ctx: _Parsed) -> list[_Planned]:
    """Yes/no columns. Logged only when words (yes, true, on...) become 1/0."""
    out: list[_Planned] = []
    for col in prof.columns:
        values = frame[col.name]
        if col.dtype != "boolean" or _is_numeric(values):
            continue
        apply = _set_column(col.name, lambda v, n=col.name: _bools(ctx, n, v))
        _, d = _view(ctx, col.name, values)
        present = [v for v, n in zip(d.objects, d.counts, strict=True) if n]
        n_words = d.total([_boolean_key(v) not in ("0", "1") for v in d.objects])
        if n_words == 0:
            out.append(_Planned(None, apply))
            continue
        yes = sorted({v for v in present if _boolean_key(v) in TRUE_KEYS})
        no = sorted({v for v in present if _boolean_key(v) not in TRUE_KEYS})
        reason = (
            f"It's a yes/no column, so {_quote_list(yes)} became 1 and "
            f"{_quote_list(no)} became 0. Missing stays missing."
        )
        out.append(_Planned(_step(col.name, "to_bool", reason, n_words), apply))
    return out


def _merge_case_stage(frame: pd.DataFrame, prof: DatasetProfile) -> list[_Planned]:
    out: list[_Planned] = []
    for col in prof.columns:
        if col.dtype != "categorical" or not col.case_variants:
            continue
        values = frame[col.name]
        mapping = case_mapping(col, values)
        n = int(values.isin(list(mapping)).sum())
        if not mapping or n == 0:
            continue
        counts = values.value_counts()
        merges = [
            f"{_quote_list(sorted(set(sp)))} became '{_case_winner(sp, counts)}'"
            for sp in col.case_variants.values()
        ]
        reason = (
            "Some values only differ by case, so they were merged: "
            + "; ".join(merges)
            + ". All-caps spellings win, otherwise the most common one does."
        )
        apply = _set_column(col.name, lambda v, m=mapping: apply_mapping(v, m))
        out.append(_Planned(_step(col.name, "merge_case", reason, n), apply))
    return out


def _presence_stage(frame: pd.DataFrame, prof: DatasetProfile) -> list[_Planned]:
    out: list[_Planned] = []
    for col in prof.columns:
        if not _is_presence_case(col):
            continue
        new = _free_name(frame, f"{col.name}_present")
        value = col.sample_values[0] if col.sample_values else "a value"
        reason = (
            f"Only one value ('{value}') shows up, so the useful part is whether it's there "
            f"or not. Swapped for {new} (1 = present, 0 = missing)."
        )

        def apply(f: pd.DataFrame, c: str = col.name, n: str = new) -> pd.DataFrame:
            flag = to_clean_strings(f[c]).notna().astype(np.int64)
            return _replace_column(f, c, {n: flag})

        out.append(_Planned(_step(col.name, "presence_flag", reason, len(frame)), apply))
    return out


def _label(value: Any) -> Any:
    if _is_missing_scalar(value):
        return np.nan
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return str(int(value))
    return str(value)


def to_labels(values: pd.Series) -> pd.Series:
    """Text labels for codes stored as numbers (2.0 -> "2"). Missing stays missing."""
    codes, uniques = factorize(values)
    d = Distinct(values.index, codes, pd.Series(dtype=object), np.zeros(0))
    return d.to_rows([_label(u) for u in uniques], np.nan).rename(values.name)


def _labels_stage(frame: pd.DataFrame, prof: DatasetProfile) -> list[_Planned]:
    """Categories stored as numbers (like sex 1 / 2 read by plain pd.read_csv) become text,
    so the category encoders see labels. Silent, like any plain parse."""
    out: list[_Planned] = []
    for col in prof.columns:
        if col.dtype != "categorical" or col.name not in frame.columns:
            continue
        if _is_numeric(frame[col.name]):
            out.append(_Planned(None, _set_column(col.name, to_labels)))
    return out


# ---------------------------------------------------------------- placeholder codes


def placeholder_codes(values: pd.Series) -> dict[float, int]:
    """Codes like 999 or -1 that stand in for missing, as code -> count.

    A code counts when it shows up at least twice and sits apart from the real
    values: positive codes above twice the biggest real value, negative codes
    only when no real value is negative.
    """
    present = values.dropna()
    others = present[~present.isin(PLACEHOLDER_CODES)]
    if others.empty:
        return {}
    found: dict[float, int] = {}
    for code in PLACEHOLDER_CODES:
        n = int((present == code).sum())
        apart = code > 2 * others.max() if code > 0 else others.min() >= 0
        if n >= PLACEHOLDER_MIN_COUNT and apart:
            found[code] = n
    return found


def _placeholder_stage(frame: pd.DataFrame, prof: DatasetProfile) -> list[_Planned]:
    out: list[_Planned] = []
    for col in prof.columns:
        if col.dtype != "numeric":
            continue
        values = frame[col.name]
        if not _is_numeric(values):
            continue
        codes = placeholder_codes(values)
        if not codes:
            continue
        hit = values.isin(list(codes))
        listed = ", ".join(f"{c:g} ({_plural(n, 'time')})" for c, n in codes.items())
        reason = (
            f"Found codes that look like stand-ins for missing: {listed}. They sit far away "
            "from the real values, so they're now missing."
        )
        blank = _Blank(col.name, values.index[hit], _is_numeric)
        step = _step(col.name, "placeholder", reason, int(hit.sum()))
        out.append(_Planned(step, blank=blank))
    return out


# ---------------------------------------------------------------- impossible values


@dataclass(frozen=True)
class _Rule:
    """One impossible-value rule. label says what's wrong, noun what the real values are."""

    label: str
    noun: str
    breaks: Callable[[pd.Series], pd.Series]


_NON_NEGATIVE = frozenset(
    {"price", "cost", "qty", "quantity", "count", "weight", "height", "salary", "income"}
)
_PCT = frozenset({"pct", "percent", "percentage"})
_NEVER_ZERO = frozenset({"bmi", "bp", "systolic", "diastolic", "glucose", "weight", "height"})
_NEVER_ZERO_PAIRS = (frozenset({"blood", "pressure"}), frozenset({"heart", "rate"}))


def rules_for(col: ColumnProfile) -> list[_Rule]:
    """Rules that fit this column, picked from whole words in its name."""
    tokens = set(name_tokens(col.name))
    rules: list[_Rule] = []
    if "age" in tokens:
        rules.append(_Rule("negative", "ages", lambda v: v < 0))
        rules.append(_Rule("over 120", "ages", lambda v: v > 120))
    if tokens & _NON_NEGATIVE:
        rules.append(_Rule("negative", "amounts", lambda v: v < 0))
    if tokens & _PCT or "percent" in col.disguise:
        rules.append(_Rule("negative", "percentages", lambda v: v < 0))
        rules.append(_Rule("over 100", "percentages", lambda v: v > 100))
    if tokens & _NEVER_ZERO or any(pair <= tokens for pair in _NEVER_ZERO_PAIRS):
        rules.append(_Rule("zero", "readings", lambda v: v == 0))
    return rules


@dataclass
class _RuleCheck:
    ran: dict[str, pd.Series]  # label -> rows, for rules that ran
    skipped: dict[str, tuple[str, pd.Series]]  # label -> (noun, rows), skipped by the guard

    def hit(self) -> pd.Series:
        """Rows the rules that ran will blank."""
        return _any_row(list(self.ran.values()))

    def n_skipped(self) -> int:
        return int(_any_row([rows for _, rows in self.skipped.values()]).sum())


def _any_row(masks: list[pd.Series]) -> pd.Series:
    out = masks[0].copy() if masks else pd.Series(dtype=bool)
    for mask in masks[1:]:
        out |= mask
    return out


def _merged_rules(values: pd.Series, rules: list[_Rule]) -> dict[str, tuple[str, pd.Series]]:
    """label -> (noun, rows breaking it). Rules sharing a label (like "negative" for an
    age_count column, from both the age and the count rule) merge, so no value counts twice."""
    present = values.notna()
    merged: dict[str, tuple[str, pd.Series]] = {}
    for rule in rules:
        breaks = rule.breaks(values) & present
        if rule.label in merged:
            noun, rows = merged[rule.label]
            merged[rule.label] = (noun, rows | breaks)
        else:
            merged[rule.label] = (rule.noun, breaks)
    return merged


def _check_rules(values: pd.Series, rules: list[_Rule]) -> _RuleCheck:
    """Run each rule, unless more than 10% of present values break it (then it's skipped)."""
    n_present = int(values.notna().sum())
    check = _RuleCheck({}, {})
    for label, (noun, rows) in _merged_rules(values, rules).items():
        n = int(rows.sum())
        if n == 0:
            continue
        if n <= IMPOSSIBLE_MAX_SHARE * n_present:
            check.ran[label] = rows
        else:
            check.skipped[label] = (noun, rows)
    return check


def _count_phrase(n: int, label: str) -> str:
    """ "3 zero values", "1 negative value", "2 values over 120"."""
    if label.startswith("over "):
        return f"{_plural(n, 'value')} {label}"
    return _plural(n, f"{label} value")


def _impossible_reason(name: str, ran: dict[str, pd.Series]) -> str:
    listed = ", ".join(_count_phrase(int(rows.sum()), label) for label, rows in ran.items())
    return f"Some values can't be real for {name}, so they're now missing: {listed}."


def _skipped_reason(name: str, skipped: dict[str, tuple[str, pd.Series]], n_present: int) -> str:
    return " ".join(
        f"{name} has {_fmt_pct(100 * int(rows.sum()) / n_present)} "
        f"{'values ' + label if label.startswith('over ') else label + ' values'}, "
        f"so it looks standardised or coded some other way, not real {noun}. "
        "Rule skipped, values kept."
        for label, (noun, rows) in skipped.items()
    )


def _impossible_stage(frame: pd.DataFrame, prof: DatasetProfile) -> list[_Planned]:
    """Values that can't be real, then flags for rules the 10% guard switched off.

    Both come from one look at the frame, so the guard decides once per rule.
    """
    fixes: list[_Planned] = []
    flags: list[_Planned] = []
    for col in prof.columns:
        if col.dtype != "numeric":
            continue
        values = frame[col.name]
        if not _is_numeric(values):
            continue
        check = _check_rules(values, rules_for(col))
        if check.ran:
            hit = check.hit()
            reason = _impossible_reason(col.name, check.ran)
            step = _step(col.name, "impossible", reason, int(hit.sum()))
            fixes.append(_Planned(step, blank=_Blank(col.name, values.index[hit], _is_numeric)))
        if check.skipped:
            reason = _skipped_reason(col.name, check.skipped, int(values.notna().sum()))
            step = _step(col.name, "impossible_skipped", reason, check.n_skipped(), kind="flag")
            flags.append(_Planned(step))
    return fixes + flags


# ---------------------------------------------------------------- dates and times


def _future_allowed(name: str) -> bool:
    """Due dates, end dates, renewals and the like can sit in the future."""
    return any(t in FUTURE_OK_TOKENS or t.startswith("expir") for t in name_tokens(name))


_YEAR_9999_RE = r"(?<!\d)9999(?!\d)"


def bad_dates(parsed: pd.Series, name: str, ref: pd.Timestamp) -> tuple[pd.Series, list[str]]:
    """(rows to blank, one phrase per reason) for parsed dates that can't be right.

    Only covers dates that parsed, so switching the step off really brings them
    back. Each row is counted under one reason only, in the order listed.
    """
    placeholder = parsed.dt.normalize().isin(PLACEHOLDER_DATES)
    too_old = (parsed < OLDEST_DATE) & ~placeholder
    future = (parsed.dt.normalize() > ref) & (not _future_allowed(name))
    reasons = [
        (placeholder, "placeholder date", " (like 1900-01-01 or 1970-01-01)"),
        (too_old, "date", " from before 1900"),
        (future, "date", f" in the future (after {ref.date().isoformat()})"),
    ]
    phrases = [_plural(int(m.sum()), word) + tail for m, word, tail in reasons if m.any()]
    return placeholder | too_old | future, phrases


def _unparseable_reason(text: pd.Series, rows: pd.Series) -> str:
    """text is the trimmed text of the column, rows the values that aren't dates."""
    text = text[rows]
    year_9999 = text[text.str.contains(_YEAR_9999_RE, regex=True)]
    example = (year_9999 if len(year_9999) else text).iloc[0]
    n = int(rows.sum())
    if n == 1:
        return f"1 value isn't a real date (like '{example}'), so it's missing."
    return f"{n} values aren't real dates (like '{example}'), so they're missing."


def _dates_stage(
    frame: pd.DataFrame, prof: DatasetProfile, ref: pd.Timestamp, ctx: _Parsed
) -> list[_Planned]:
    """Parse dates (silent, always on), then blank the ones that can't be right.

    Values that don't parse (or won't fit, like 9999-12-31) are missing either
    way, so they get a flag instead of a step you could switch off.
    """
    fixes: list[_Planned] = []
    flags: list[_Planned] = []
    for col in prof.columns:
        if col.dtype != "datetime":
            continue
        values = frame[col.name]
        apply = _set_column(col.name, lambda v, n=col.name: _datetimes(ctx, n, v))
        fixes.append(_Planned(None, apply))
        parsed = _datetimes(ctx, col.name, values)
        bad, phrases = bad_dates(parsed, col.name, ref)
        if phrases:
            reason = f"Some dates can't be right, so they're now missing: {', '.join(phrases)}."
            step = _step(col.name, "invalid_dates", reason, int(bad.sum()))
            fixes.append(_Planned(step, blank=_Blank(col.name, values.index[bad], _is_datetime)))
        text = _clean_text(ctx, col.name, values)
        junk = text.notna() & parsed.isna()
        if junk.any():
            reason = _unparseable_reason(text, junk)
            step = _step(col.name, "unparseable_dates", reason, int(junk.sum()), kind="flag")
            flags.append(_Planned(step))
    return fixes + flags


def _times_stage(frame: pd.DataFrame, prof: DatasetProfile, ctx: _Parsed) -> list[_Planned]:
    """Times of day become "HH:MM:SS" (silent), then ones that aren't real times are blanked."""
    out: list[_Planned] = []
    for col in prof.columns:
        if col.dtype != "time":
            continue
        values = frame[col.name]
        out.append(_Planned(None, _set_column(col.name, lambda v, n=col.name: _times(ctx, n, v))))
        bad = _clean_text(ctx, col.name, values).notna() & _time_rows(ctx, col.name, values).isna()
        n = int(bad.sum())
        if n == 0:
            continue
        examples = _quote_list(sorted({str(v).strip() for v in values[bad]}))
        reason = (
            f"{_plural(n, 'value')} ({examples}) can't be a time of day, so they're now missing."
            if n > 1
            else f"1 value ({examples}) can't be a time of day, so it's now missing."
        )
        step = _step(col.name, "invalid_times", reason, n)
        out.append(_Planned(step, blank=_Blank(col.name, values.index[bad])))
    return out


# ---------------------------------------------------------------- dropping and flags


def _drop_id_stage(frame: pd.DataFrame, prof: DatasetProfile) -> list[_Planned]:
    out: list[_Planned] = []
    for col in prof.columns:
        if col.dtype != "id":
            continue
        why = col.reasons[0] if col.reasons else "This looks like an ID."
        reason = (
            f"{why} IDs don't carry over to new rows, so it's left out of the features "
            "(it's still kept to label rows)."
        )
        out.append(_Planned(_step(col.name, "drop_id", reason, len(frame)), _drop(col.name)))
    return out


def _drop_constant_stage(frame: pd.DataFrame, prof: DatasetProfile) -> list[_Planned]:
    out: list[_Planned] = []
    for col in prof.columns:
        if col.dtype != "constant":
            continue
        if col.all_missing:
            reason = "Every value is missing, so this column carries no information."
        else:
            value = col.sample_values[0] if col.sample_values else ""
            reason = f"Only one value ('{value}') shows up, so it can't help a model."
        out.append(_Planned(_step(col.name, "drop_constant", reason, len(frame)), _drop(col.name)))
    return out


HIGH_MISSING_TYPES = frozenset({"numeric", "boolean", "categorical", "datetime", "time"})


def _high_missing_stage(frame: pd.DataFrame, prof: DatasetProfile) -> list[_Planned]:
    """Flag kept feature columns with lots of gaps. ids, text and constants never reach a model."""
    out: list[_Planned] = []
    for col in prof.columns:
        if col.missing_pct <= HIGH_MISSING_PCT or col.dtype not in HIGH_MISSING_TYPES:
            continue
        if _is_presence_case(col) or col.name not in frame.columns:
            continue
        n_missing = len(frame) - _present_count(frame[col.name])
        reason = (
            f"{col.missing_pct:.0f}% of values are missing. It's kept, and gaps get filled in "
            "during training, but think about dropping it."
        )
        out.append(_Planned(_step(col.name, "high_missing", reason, n_missing, kind="flag")))
    return out


Stage = Callable[[pd.DataFrame, DatasetProfile], list[_Planned]]


def _stages(ref: pd.Timestamp, ctx: _Parsed) -> list[Stage]:
    """Every stage in log order. Placeholder codes go before impossible values on purpose,
    so a 999 age is reported as a code, not as an age over 120."""
    return [
        _dedupe_stage,
        lambda frame, prof: _normalize_stage(frame, prof, ctx),
        lambda frame, prof: _to_numeric_stage(frame, prof, ctx),
        lambda frame, prof: _to_bool_stage(frame, prof, ctx),
        _merge_case_stage,
        _presence_stage,
        _labels_stage,
        _placeholder_stage,
        _impossible_stage,
        lambda frame, prof: _dates_stage(frame, prof, ref, ctx),
        lambda frame, prof: _times_stage(frame, prof, ctx),
        _drop_id_stage,
        _drop_constant_stage,
        _high_missing_stage,
    ]


# ---------------------------------------------------------------- entry point


def _working_copy(df: pd.DataFrame, prof: DatasetProfile) -> pd.DataFrame:
    """Deep copy with columns named exactly as in the profile."""
    names = [c.name for c in prof.columns]
    if len(names) != df.shape[1]:
        raise ValueError("The profile doesn't match this frame (different number of columns).")
    if len(set(names)) != len(names):
        raise ValueError("Column names must be unique to clean the frame.")
    if not df.index.is_unique:
        raise ValueError(
            "Row labels (the index) must be unique to clean the frame, since fixes are "
            "tracked per row. Try df.reset_index(drop=True) first."
        )
    return df.copy(deep=True).set_axis(names, axis=1)


def _row_ids(
    work: pd.DataFrame, prof: DatasetProfile, deduped: bool, ctx: _Parsed
) -> pd.Series | None:
    id_col = next((c.name for c in prof.columns if c.dtype == "id"), None)
    if id_col is None:
        return None
    ids = _clean_text(ctx, id_col, work[id_col]).rename(id_col)
    return ids[~_duplicate_mask(work)] if deduped else ids


def clean(
    df: pd.DataFrame,
    profile: DatasetProfile,
    disabled: set[str] | None = None,
    reference_date: ReferenceDate = None,
) -> tuple[pd.DataFrame, CleaningLog]:
    """Apply stateless fixes and log each one. Steps listed in disabled are logged but skipped.

    reference_date is "today" when spotting future dates; None means the real today.
    Never mutates df or profile. The original index is kept.
    """
    disabled = disabled or set()
    work = _working_copy(df, profile)
    ctx = _Parsed(work, profile)
    groups = duplicate_groups(work, ctx.row_codes())
    planned, actual = work, work
    log = CleaningLog()
    for stage in _stages(_reference_day(reference_date), ctx):
        for item in stage(planned, profile):
            step, apply = item.step, item.apply
            if item.blank is not None:
                # with dedupe off, the copies of a fixed row get the same fix
                apply = _null_rows(item.blank, _with_copies(item.blank.labels, groups))
            if step is None:  # silent parse: always on, never logged
                if apply is not None:
                    planned, actual = apply(planned), apply(actual)
                continue
            if apply is not None:
                step.enabled = step.step_id not in disabled
                planned = apply(planned)
                if step.enabled:
                    actual = apply(actual)
            log.steps.append(step)
    deduped = any(s.action == "dedupe" and s.enabled for s in log.steps)
    log.row_ids = _row_ids(work, profile, deduped, ctx)
    return actual.copy(), log
