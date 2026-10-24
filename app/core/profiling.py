"""Phase 1: dataset profiling. Detects column types and per-column stats."""

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class ColumnProfile:
    name: str
    dtype: str  # numeric | categorical | datetime | text | id | constant
    missing_pct: float
    n_unique: int


@dataclass
class DatasetProfile:
    n_rows: int
    n_cols: int
    columns: list[ColumnProfile] = field(default_factory=list)


def profile(df: pd.DataFrame) -> DatasetProfile:
    """Profile a DataFrame. TODO (Phase 1): full type detection, outliers, disguised types."""
    raise NotImplementedError
