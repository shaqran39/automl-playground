"""Extra profiling edge cases beyond the Phase 1 contract tests."""

import numpy as np
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


# ---------------------------------------------------------------- review fixes


def _col(values: list, name: str = "c") -> P.ColumnProfile:
    return P.profile(pd.DataFrame({name: values})).column(name)


@pytest.mark.parametrize(
    ("values", "dayfirst"),
    [
        (["13/01/2024"] + [f"01/{d}/2024" for d in range(13, 31)], False),
        (["01/13/2024"] + [f"{d}/01/2024" for d in range(13, 31)], True),
    ],
)
def test_conflicting_day_month_order_keeps_majority_and_flags(
    values: list[str], dayfirst: bool
) -> None:
    col = _col(values)
    assert col.dtype == "datetime"
    assert col.dayfirst is dayfirst
    assert col.date_ambiguous is True
    assert col.parse_rate == pytest.approx(1.0)
    assert any("both" in r for r in col.reasons)


def test_parse_dates_conflict_parses_every_row() -> None:
    parsed, dayfirst, ambiguous = P.parse_dates(
        pd.Series(["13/01/2024", "01/14/2024", "01/15/2024"])
    )
    assert parsed.notna().all()
    assert parsed.dt.strftime("%Y-%m-%d").tolist() == ["2024-01-13", "2024-01-14", "2024-01-15"]
    assert dayfirst is False
    assert ambiguous is True


@pytest.mark.parametrize(
    "template",
    [
        "{d:02d}/03/2024 10:30",
        "{d:02d}-03-2024 10:30:15",
        "{d:02d}/03/24",
        "{d}-3-24",
        "2024/03/{d:02d} 10:00",
        "{d} Mar 2024",
        "Mar {d}, 2024",
        "{d} March 2024",
    ],
)
def test_common_date_formats_are_datetime(template: str) -> None:
    col = _col([template.format(d=d) for d in range(1, 29)])
    assert col.dtype == "datetime"
    assert col.date_range == ("2024-03-01", "2024-03-28")


def test_two_digit_year_parses() -> None:
    parsed, dayfirst, ambiguous = P.parse_dates(pd.Series(["05/03/24"]))
    assert parsed.iloc[0] == pd.Timestamp("2024-03-05")
    assert dayfirst is True
    assert ambiguous is True


def test_dotted_version_strings_are_not_dates() -> None:
    col = _col([f"1.{i}.{j}" for i in range(1, 6) for j in range(10, 16)])
    assert col.dtype != "datetime"


def test_month_word_without_year_is_not_date() -> None:
    col = _col(["may call back", "march order", "dec promo"] * 10)
    assert col.dtype != "datetime"


@pytest.mark.parametrize(
    ("values", "name"),
    [
        (list(range(1990, 2021)), "year"),
        (list(range(1990, 2021)), "x"),
        (list(range(20, 41)), "age"),
    ],
)
def test_unique_years_and_small_ranges_stay_numeric(values: list[int], name: str) -> None:
    assert _col(values, name).dtype == "numeric"


@pytest.mark.parametrize("name", ["id", "ID", "customer_id", "userId", "uuid", "order key"])
def test_random_unique_ints_with_id_name_are_id(name: str) -> None:
    rng = np.random.default_rng(0)
    values = rng.choice(10**6, 200, replace=False).tolist()
    assert _col(values, name).dtype == "id"


@pytest.mark.parametrize("name", ["amount", "paid", "valid", "acid"])
def test_random_unique_ints_without_id_name_stay_numeric_with_note(name: str) -> None:
    rng = np.random.default_rng(0)
    col = _col(rng.choice(10**6, 200, replace=False).tolist(), name)
    assert col.dtype == "numeric"
    assert any("could be an ID" in r for r in col.reasons)


def test_huge_int_ids_with_id_name_are_id() -> None:
    assert _col([str(10**18 + i) for i in range(50)], "account_id").dtype == "id"


def test_sequential_ints_from_1000_are_id() -> None:
    assert _col(list(range(1000, 1300)), "x").dtype == "id"


def test_id_named_floats_stay_numeric() -> None:
    rng = np.random.default_rng(1)
    assert _col(rng.normal(0, 1, 100).tolist(), "score_id").dtype == "numeric"


# ---------------------------------------------------------------- leading-zero codes


@pytest.mark.parametrize(
    "values",
    [
        ["0800", "0870", "2000", "3000", "4000", "6000"] * 5,
        ["02000", "02134", "10001", "90210"] * 5,
        ["0800"] * 10 + ["2000"] * 10,
    ],
)
def test_leading_zero_codes_are_categorical(values: list[str]) -> None:
    col = _col(values, "postcode")
    assert col.dtype == "categorical"
    assert col.stats is None
    assert any("start with 0" in r for r in col.reasons)


def test_many_distinct_leading_zero_codes_are_categorical_not_text() -> None:
    codes = [f"{i:05d}" for i in range(0, 1200, 10)]  # 120 codes like 00010
    values = (codes * 2)[:200]
    assert _col(values, "zip").dtype == "categorical"


def test_unique_leading_zero_codes_with_id_name_are_id() -> None:
    assert _col([f"{i:06d}" for i in range(100)], "account_id").dtype == "id"


@pytest.mark.parametrize(
    "values",
    [
        ["0", "10", "20", "30"] * 10,
        ["0.5", "0.25", "1.5", "0"] * 10,
        ["-0.5", "0.0", "1", "2"] * 10,
    ],
)
def test_plain_zero_and_decimals_stay_numeric(values: list[str]) -> None:
    assert _col(values).dtype == "numeric"


def test_one_stray_leading_zero_does_not_flip_numeric_column() -> None:
    rng = np.random.default_rng(2)
    prices = [f"{v:.2f}" for v in rng.uniform(10, 200, 999)] + ["00"]
    assert _col(prices, "price").dtype == "numeric"
    amounts = ["007"] + [str(v) for v in rng.integers(1, 500, 199)]
    assert _col(amounts, "amount").dtype == "numeric"


def test_one_leading_zero_value_does_not_flip_text_column() -> None:
    comments = [f"customer number {i} said the app was slow and buggy today" for i in range(99)]
    assert _col(comments + ["01"], "comment").dtype == "text"


@pytest.mark.parametrize(
    ("values", "name"),
    [
        ([f"{h:02d}" for h in range(24)] * 3, "hour"),
        ([f"{m:02d}" for m in range(1, 13)] * 5, "month"),
    ],
)
def test_zero_padded_time_parts_stay_numeric(values: list[str], name: str) -> None:
    assert _col(values, name).dtype == "numeric"


def test_unique_leading_zero_codes_without_id_name_are_id() -> None:
    assert _col([f"0{i:05d}" for i in range(100)], "ref").dtype == "id"


def test_leading_zero_codes_with_missing_values() -> None:
    col = _col(["0800", "0870", "2000", "", None, "3000"] * 5, "postcode")
    assert col.dtype == "categorical"
    assert col.missing_pct == pytest.approx(100 * 2 / 6)


def test_zero_one_and_padded_one_is_boolean() -> None:
    assert _col(["0", "1", "01"] * 10).dtype == "boolean"


def test_postcodes_through_plain_read_csv_lose_zero_and_go_numeric(tmp_path) -> None:
    path = tmp_path / "pc.csv"
    path.write_text("postcode\n" + "\n".join(["0800", "0870", "2000", "3000"] * 5))
    assert P.profile(pd.read_csv(path)).column("postcode").dtype == "numeric"
    assert P.profile(P.read_csv_raw(path)).column("postcode").dtype == "categorical"


def test_id_name_needs_enough_rows() -> None:
    assert _col(["01", "02", "03"], "id").dtype != "id"
