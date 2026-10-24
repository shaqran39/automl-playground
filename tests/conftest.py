from pathlib import Path

import pandas as pd
import pytest

DATA = Path(__file__).parent.parent / "sample_data"


@pytest.fixture(params=sorted(p.name for p in DATA.glob("*.csv")))
def sample_df(request) -> pd.DataFrame:
    return pd.read_csv(DATA / request.param)
