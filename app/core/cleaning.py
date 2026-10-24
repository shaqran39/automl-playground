"""Phase 2: contextual cleaning with a reversible cleaning log.

clean() runs before the train/test split, so it only does stateless, row-local
fixes: trimming, null tokens, type conversion, case merging, dedupe and
dropping id/constant columns. Nothing here learns from the data (no imputing,
scaling, encoding or clipping). That all lives in the model Pipeline.

Every step is logged as (column, action, reason). The log is always built as
if every step was on, so switching steps off never changes what's listed, only
whether each fix gets applied.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd

from app.core.profiling import (
    MISSING_TOKENS,
    ColumnProfile,
    DatasetProfile,
    _boolean_key,
    _disguise_phrase,
    _is_missing_scalar,
    _quote_list,
    parse_dates,
    parse_numeric,
    to_clean_strings,
)

StepKind = Literal["fix", "flag"]
Transform = Callable[[pd.DataFrame], pd.DataFrame]

HIGH_MISSING_PCT = 50.0
TRUE_KEYS: frozenset[str] = frozenset({"yes", "y", "true", "t", "1", "on"})
LOG_COLUMNS = ["step_id", "column", "action", "reason", "kind", "enabled", "rows_affected"]
TABLE = "*"


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
    """A logged step plus the change it makes. apply is None for flag steps."""

    step: CleaningStep
    apply: Transform | None = None


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


# ---------------------------------------------------------------- value fixes


def _normalize_value(value: Any) -> Any:
    if isinstance(value, str):
        text = value.strip()
        return np.nan if text.lower() in MISSING_TOKENS else text
    return np.nan if _is_missing_scalar(value) else value


def normalize_values(values: pd.Series) -> pd.Series:
    """Trim strings and turn null tokens (and None) into NaN. Non-strings are left alone."""
    out = [_normalize_value(v) for v in values]
    return pd.Series(out, index=values.index, dtype=values.dtype, name=values.name)


def _normalize_counts(values: pd.Series) -> tuple[int, int]:
    """(values trimmed, values turned into missing) among the strings."""
    trimmed = nulled = 0
    for v in values:
        if not isinstance(v, str):
            continue
        text = v.strip()
        if text.lower() in MISSING_TOKENS:
            nulled += 1
        elif text != v:
            trimmed += 1
    return trimmed, nulled


def to_numbers(values: pd.Series) -> pd.Series:
    """float64 numbers with disguises removed. Leftovers become NaN."""
    parsed, _ = parse_numeric(to_clean_strings(values))
    return parsed.astype(np.float64).rename(values.name)


def _bool_value(value: str | None) -> float:
    if value is None:
        return np.nan
    return 1.0 if _boolean_key(value) in TRUE_KEYS else 0.0


def to_bools(values: pd.Series) -> pd.Series:
    """1.0 for the yes side, 0.0 for the no side, NaN when missing."""
    out = [_bool_value(v) for v in to_clean_strings(values)]
    return pd.Series(out, index=values.index, dtype=np.float64, name=values.name)


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


def _parse_dates(values: pd.Series) -> pd.Series:
    if isinstance(values.dtype, pd.DatetimeTZDtype):
        return values.dt.tz_localize(None)
    if pd.api.types.is_datetime64_any_dtype(values):
        return values
    parsed, _, _ = parse_dates(to_clean_strings(values))
    return parsed


def date_part_columns(values: pd.Series, names: tuple[str, str, str]) -> dict[str, pd.Series]:
    """year, month and day of week (Mon=0) as float64, NaN where the date didn't parse."""
    parsed = _parse_dates(values)
    parts = [parsed.dt.year, parsed.dt.month, parsed.dt.day_of_week]
    return {n: p.astype(np.float64) for n, p in zip(names, parts, strict=True)}


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


def _normalize_stage(frame: pd.DataFrame, prof: DatasetProfile) -> list[_Planned]:
    out: list[_Planned] = []
    for col in prof.columns:
        values = frame[col.name]
        if not _is_text_column(values):
            continue
        trimmed, nulled = _normalize_counts(values)
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
        apply = _set_column(col.name, normalize_values)
        out.append(_Planned(_step(col.name, "normalize", reason, trimmed + nulled), apply))
    return out


def _to_numeric_stage(frame: pd.DataFrame, prof: DatasetProfile) -> list[_Planned]:
    out: list[_Planned] = []
    for col in prof.columns:
        values = frame[col.name]
        if col.dtype != "numeric" or pd.api.types.is_numeric_dtype(values):
            continue
        clean = to_clean_strings(values)
        parsed, disguise = parse_numeric(clean)
        n_present = int(clean.notna().sum())
        leftover = n_present - int(parsed.notna().sum())
        reason = f"Values read as numbers, so stored as numbers{_disguise_phrase(disguise)}."
        if leftover:
            reason += (
                f" {_plural(leftover, 'value')} didn't parse as a number and "
                f"{'is' if leftover == 1 else 'are'} now missing."
            )
        apply = _set_column(col.name, to_numbers)
        out.append(_Planned(_step(col.name, "to_numeric", reason, n_present), apply))
    return out


def _to_bool_stage(frame: pd.DataFrame, prof: DatasetProfile) -> list[_Planned]:
    out: list[_Planned] = []
    for col in prof.columns:
        values = frame[col.name]
        if col.dtype != "boolean":
            continue
        if pd.api.types.is_numeric_dtype(values) and not pd.api.types.is_bool_dtype(values):
            continue
        present = to_clean_strings(values).dropna()
        yes = sorted({v for v in present if _boolean_key(v) in TRUE_KEYS})
        no = sorted({v for v in present if _boolean_key(v) not in TRUE_KEYS})
        reason = (
            f"It's a yes/no column, so {_quote_list(yes)} became 1 and "
            f"{_quote_list(no)} became 0. Missing stays missing."
        )
        apply = _set_column(col.name, to_bools)
        out.append(_Planned(_step(col.name, "to_bool", reason, len(present)), apply))
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
        merges = [
            f"{_quote_list(sorted(set(sp)))} became '{_case_winner(sp, values.value_counts())}'"
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


def _date_parts_stage(frame: pd.DataFrame, prof: DatasetProfile) -> list[_Planned]:
    out: list[_Planned] = []
    for col in prof.columns:
        if col.dtype != "datetime":
            continue
        names = tuple(_free_name(frame, f"{col.name}_{p}") for p in ("year", "month", "dayofweek"))
        values = frame[col.name]
        n_present = _present_count(values)
        n_parsed = int(_parse_dates(values).notna().sum())
        reason = (
            "Models can't use raw dates, so this was split into year, month and day of week "
            "(Monday = 0)."
        )
        if n_present > n_parsed:
            reason += f" {_plural(n_present - n_parsed, 'value')} didn't parse and became missing."

        def apply(f: pd.DataFrame, c: str = col.name, n: tuple[str, ...] = names) -> pd.DataFrame:
            return _replace_column(f, c, date_part_columns(f[c], n))  # type: ignore[arg-type]

        out.append(_Planned(_step(col.name, "date_parts", reason, n_parsed), apply))
    return out


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


HIGH_MISSING_TYPES = frozenset({"numeric", "boolean", "categorical", "datetime"})


def _high_missing_stage(frame: pd.DataFrame, prof: DatasetProfile) -> list[_Planned]:
    """Flag kept feature columns with lots of gaps. ids, text and constants never reach a model."""
    out: list[_Planned] = []
    for col in prof.columns:
        if col.missing_pct <= HIGH_MISSING_PCT or col.dtype not in HIGH_MISSING_TYPES:
            continue
        if _is_presence_case(col):
            continue
        if col.name in frame:
            n_missing = len(frame) - _present_count(frame[col.name])
        else:  # dates were already swapped for date parts, so estimate from the profile
            n_missing = round(col.missing_pct / 100 * len(frame))
        reason = (
            f"{col.missing_pct:.0f}% of values are missing. It's kept, and gaps get filled in "
            "during training, but think about dropping it."
        )
        out.append(_Planned(_step(col.name, "high_missing", reason, n_missing, kind="flag")))
    return out


STAGES: list[Callable[[pd.DataFrame, DatasetProfile], list[_Planned]]] = [
    _dedupe_stage,
    _normalize_stage,
    _to_numeric_stage,
    _to_bool_stage,
    _merge_case_stage,
    _presence_stage,
    _date_parts_stage,
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
    return df.copy(deep=True).set_axis(names, axis=1)


def _row_ids(work: pd.DataFrame, prof: DatasetProfile, deduped: bool) -> pd.Series | None:
    id_col = next((c.name for c in prof.columns if c.dtype == "id"), None)
    if id_col is None:
        return None
    ids = to_clean_strings(work[id_col]).rename(id_col)
    return ids[~_duplicate_mask(work)] if deduped else ids


def clean(
    df: pd.DataFrame, profile: DatasetProfile, disabled: set[str] | None = None
) -> tuple[pd.DataFrame, CleaningLog]:
    """Apply stateless fixes and log each one. Steps listed in disabled are logged but skipped.

    Never mutates df or profile. The original index is kept.
    """
    disabled = disabled or set()
    work = _working_copy(df, profile)
    planned, actual = work, work
    log = CleaningLog()
    for stage in STAGES:
        for item in stage(planned, profile):
            step = item.step
            if item.apply is not None:
                step.enabled = step.step_id not in disabled
                planned = item.apply(planned)
                if step.enabled:
                    actual = item.apply(actual)
            log.steps.append(step)
    deduped = any(s.action == "dedupe" and s.enabled for s in log.steps)
    log.row_ids = _row_ids(work, profile, deduped)
    return actual.copy(), log
