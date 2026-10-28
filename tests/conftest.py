from pathlib import Path

import pandas as pd
import pytest

DATA = Path(__file__).parent.parent / "sample_data"
SAMPLE_NAMES: list[str] = sorted(p.name for p in DATA.glob("*.csv"))
MESSY = DATA / "messy.csv"
MESSY_CLINIC = DATA / "messy_clinic.csv"
# fixed "today" for invalid-date checks, so tests never depend on the clock
REFERENCE_DATE = "2025-01-01"


@pytest.fixture(params=SAMPLE_NAMES)
def sample_name(request: pytest.FixtureRequest) -> str:
    return request.param


@pytest.fixture
def sample_df(sample_name: str) -> pd.DataFrame:
    return pd.read_csv(DATA / sample_name)


@pytest.fixture
def messy_raw() -> pd.DataFrame:
    """messy.csv with every cell as a raw string (no NaN conversion)."""
    return pd.read_csv(MESSY, dtype=str, keep_default_na=False)


@pytest.fixture
def kitchen_sink_raw(messy_raw: pd.DataFrame) -> pd.DataFrame:
    """messy.csv plus a >50% missing numeric column and one exact duplicate row.

    Every cleaning step applies to this frame at least once, so it's handy for
    disabling steps one at a time.
    """
    df = messy_raw.copy()
    df["sparse"] = [f"{(i % 7) * 1.5}" if i % 5 in (0, 1) else "" for i in range(len(df))]
    return pd.concat([df, df.iloc[[0]]], ignore_index=True)


@pytest.fixture(params=["read_csv_raw", "pd.read_csv"])
def messy_df(request: pytest.FixtureRequest) -> pd.DataFrame:
    """messy.csv loaded both via the app's raw reader and plain pandas."""
    if request.param == "read_csv_raw":
        from app.core import profiling

        return profiling.read_csv_raw(MESSY)
    return pd.read_csv(MESSY)


@pytest.fixture
def clinic_raw() -> pd.DataFrame:
    """messy_clinic.csv as raw strings, via the app's reader.

    Every Phase 2 rework rule fires on it at least once (placeholders, impossible
    values, invalid dates and times, 2-value codes, skewed income, blanks).
    """
    from app.core import profiling

    return profiling.read_csv_raw(MESSY_CLINIC)
