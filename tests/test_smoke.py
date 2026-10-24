import pandas as pd


def test_samples_load(sample_df: pd.DataFrame) -> None:
    assert len(sample_df) > 0
