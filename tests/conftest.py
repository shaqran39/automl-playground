from pathlib import Path

import pandas as pd
import pytest

DATA = Path(__file__).parent.parent / "sample_data"
SAMPLE_NAMES: list[str] = sorted(p.name for p in DATA.glob("*.csv"))
MESSY = DATA / "messy.csv"


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


@pytest.fixture(params=["read_csv_raw", "pd.read_csv"])
def messy_df(request: pytest.FixtureRequest) -> pd.DataFrame:
    """messy.csv loaded both via the app's raw reader and plain pandas."""
    if request.param == "read_csv_raw":
        from app.core import profiling

        return profiling.read_csv_raw(MESSY)
    return pd.read_csv(MESSY)
