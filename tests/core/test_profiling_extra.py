"""Extra profiling edge cases beyond the Phase 1 contract tests."""

import pandas as pd
import pytest

from app.core import profiling as P


def _col(values: list[object] | pd.Series, name: str = "col") -> P.ColumnProfile:
    return P.profile(pd.DataFrame({name: values})).column(name)


def test_real_datetime_dtype_column() -> None:
    col = _col(pd.date_range("2024-01-01", periods=20))
    assert col.dtype == "datetime"
    assert col.date_range == ("2024-01-01", "2024-01-20")


def test_tz_aware_datetimes_keep_local_dates() -> None:
    col = _col(pd.date_range("2024-01-01", periods=20, tz="Australia/Sydney"))
    assert col.dtype == "datetime"
    assert col.date_range == ("2024-01-01", "2024-01-20")


def test_month_first_dates() -> None:
    col = _col(["01/13/2024", "02/14/2024", "03/01/2024"] * 4)
    assert col.dtype == "datetime"
    assert col.dayfirst is False
    assert col.date_range == ("2024-01-13", "2024-03-01")


def test_ambiguous_dates_flagged_in_reasons() -> None:
    col = _col(["01/02/2024", "03/04/2024", "05/06/2024"] * 4)
    assert col.date_ambiguous is True
    assert any("guess" in r for r in col.reasons)


def test_negative_currency_both_orders() -> None:
    parsed, found = P.parse_numeric(pd.Series(["-$5", "$-6", "($7)"]))
    assert parsed.tolist() == [-5.0, -6.0, -7.0]
    assert set(found) == {"currency", "parens_negative"}


def test_numeric_parse_rejects_inf_and_words() -> None:
    parsed, _ = P.parse_numeric(pd.Series(["inf", "Infinity", "1e3"]))
    assert parsed.isna().tolist() == [True, True, False]


def test_unhashable_cells_do_not_crash() -> None:
    df = pd.DataFrame({"lists": [[1], [1], [2]], "x": [1, 1, 2]})
    prof = P.profile(df)
    assert prof.n_duplicate_rows == 1
    assert prof.column("lists").dtype in {"categorical", "constant"}


def test_reasons_have_no_em_dashes(messy_df: pd.DataFrame) -> None:
    for col in P.profile(messy_df).columns:
        assert all("—" not in r for r in col.reasons)


def test_short_unique_codes_with_spaces_are_not_ids() -> None:
    values = [f"order {i}" for i in range(30)]
    assert _col(values).dtype != "id"


@pytest.mark.parametrize(
    "outlier",
    ["9999-12-31", "0001-01-01", "1600-01-01", "31/12/9999", "9999-12-31 00:00"],
)
def test_out_of_range_date_does_not_crash(outlier: str) -> None:
    df = pd.DataFrame({"d": ["2024-01-05"] * 30 + [outlier]})
    col = P.profile(df).column("d")
    assert col.dtype == "datetime"
    assert col.date_range == ("2024-01-05", "2024-01-05")


def test_out_of_range_date_in_text_column_does_not_crash() -> None:
    df = pd.DataFrame({"c": ["red", "blue", "green"] * 10 + ["9999-12-31"]})
    assert P.profile(df).column("c").dtype == "categorical"


def test_far_future_datetime64_column_does_not_crash() -> None:
    values = pd.Series(pd.to_datetime(["2024-01-05"] * 30 + ["3000-01-01"]).astype("datetime64[s]"))
    P.profile(pd.DataFrame({"d": values}))
