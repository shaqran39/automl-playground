"""read_csv_report on small broken CSVs: separators, encodings, quotes, junk lines, bad rows."""

import codecs
import io
from pathlib import Path

import pandas as pd
import pytest

from app.core import loading as L
from app.core import profiling as P


def load(data: bytes | str) -> tuple[pd.DataFrame, L.CsvReport]:
    raw = data if isinstance(data, bytes) else data.encode("utf-8")
    return L.read_csv_report(io.BytesIO(raw))


# ---------------------------------------------------------------- separators


@pytest.mark.parametrize("sep", [",", ";", "\t", "|"])
def test_detects_separator(sep: str) -> None:
    text = "\n".join(sep.join(r) for r in [["a", "b", "c"], ["1", "x", "2.5"], ["3", "y", "4"]])
    df, rep = load(text + "\n")
    assert rep.sep == sep
    assert list(df.columns) == ["a", "b", "c"]
    assert df.shape == (2, 3)


def test_airbnb_shape_semicolons_with_commas_inside_values() -> None:
    """The real bug: ';' file whose values hold commas. Reading on commas crashed on line 10."""
    header = "Room ID;Name;Coordinates;Location"
    rows = [f"{i};Flat {i};55.95, -3.18;United kingdom, Edinburgh, Old Town" for i in range(8)]
    rows.append(
        "9;Big flat;55.95, -3.18;United kingdom, Edinburgh, Old Town, Princes Street, Leith"
    )
    df, rep = load("\r\n".join([header, *rows]) + "\r\n")
    assert rep.sep == ";"
    assert df.shape == (9, 4)
    assert df.loc[8, "Location"] == "United kingdom, Edinburgh, Old Town, Princes Street, Leith"
    assert rep.skipped_rows == 0


def test_single_column_file() -> None:
    df, rep = load("name\nann\nbob\n")
    assert list(df.columns) == ["name"]
    assert df["name"].tolist() == ["ann", "bob"]


def test_trailing_separator_on_every_line() -> None:
    df, rep = load("a,b,c\n1,2,3,\n4,5,6,\n")
    assert rep.sep == ","
    assert list(df.columns) == ["a", "b", "c"]
    assert df["a"].tolist() == ["1", "4"]


def test_separator_note_only_when_not_comma() -> None:
    assert load("a,b\n1,2\n")[1].notes() == []
    assert "semicolons" in load("a;b\n1;2\n")[1].notes()[0]


# ---------------------------------------------------------------- encodings


def test_utf8_bom_is_stripped_from_first_column() -> None:
    df, rep = load(codecs.BOM_UTF8 + b"Room ID,price\n1,2\n")
    assert rep.encoding == "utf-8-sig"
    assert df.columns[0] == "Room ID"


def test_plain_utf8() -> None:
    df, rep = load("city,n\nSödermalm,1\n浦东新区,2\n")
    assert rep.encoding == "utf-8"
    assert df["city"].tolist() == ["Södermalm", "浦东新区"]


def test_latin1_bytes() -> None:
    df, rep = load("city,n\ncafé,1\nZürich,2\n".encode("latin-1"))
    assert rep.encoding in ("cp1252", "latin-1")
    assert df["city"].tolist() == ["café", "Zürich"]
    assert "isn't UTF-8" in rep.notes()[0]


def test_cp1252_euro_sign() -> None:
    df, rep = load("price,n\n€5,1\n€7,2\n".encode("cp1252"))
    assert df["price"].tolist() == ["€5", "€7"]


def test_latin1_fallback_when_cp1252_fails() -> None:
    df, rep = load(b"a,b\n\x81x,1\ny,2\n")  # 0x81 is undefined in cp1252
    assert rep.encoding == "latin-1"
    assert df.shape == (2, 2)


def test_utf16_with_bom() -> None:
    df, rep = load("a\tb\n1\t2\n".encode("utf-16"))
    assert rep.encoding == "utf-16"
    assert rep.sep == "\t"
    assert df.shape == (1, 2)


# ---------------------------------------------------------------- quotes


def test_quoted_fields_with_separators_and_line_breaks() -> None:
    text = 'id,note,price\n1,"big, bright\nand sunny",10\n2,"says ""hi""",20\n3,plain,30\n'
    df, rep = load(text)
    assert rep.sep == ","
    assert df.shape == (3, 3)
    assert df.loc[0, "note"] == "big, bright\nand sunny"
    assert df.loc[1, "note"] == 'says "hi"'
    assert rep.skipped_rows == 0


def test_quoted_semicolon_file_with_commas() -> None:
    df, rep = load('a;b\n"x, y";1\n"z; w";2\n')
    assert rep.sep == ";"
    assert df["a"].tolist() == ["x, y", "z; w"]


def test_unclosed_quote_does_not_crash() -> None:
    """Either it reads (pandas swallows to the end) or a friendly CsvLoadError."""
    try:
        df, _ = load('a,b\n1,"oops\n2,3\n4,5\n')
    except L.CsvLoadError as e:
        assert "quote" in str(e)
    else:
        assert list(df.columns) == ["a", "b"]


# ---------------------------------------------------------------- junk above the header


def test_skips_title_lines_above_header() -> None:
    text = "Sales report\nGenerated 2024-01-01\n\nid,name,amount\n1,a,3\n2,b,4\n3,c,5\n"
    df, rep = load(text)
    assert list(df.columns) == ["id", "name", "amount"]
    assert df.shape == (3, 3)
    assert rep.junk_lines == 2
    assert "Skipped 2 lines" in " ".join(rep.notes())


def test_junk_line_with_a_separator_is_still_skipped() -> None:
    text = "Report, Q1\nid;name;amount\n1;a;3\n2;b;4\n3;c;5\n"
    df, rep = load(text)
    assert rep.sep == ";"
    assert list(df.columns) == ["id", "name", "amount"]


def test_leading_blank_lines_are_not_junk() -> None:
    df, rep = load("\n\na,b\n1,2\n")
    assert list(df.columns) == ["a", "b"]
    assert rep.junk_lines == 0


# ---------------------------------------------------------------- broken rows


def test_rows_with_too_many_fields_are_skipped_and_counted() -> None:
    text = "a,b,c\n1,2,3\n4,5,6,7,8\n9,10,11\n12,13,14,15\n16,17,18\n"
    df, rep = load(text)
    assert df["a"].tolist() == ["1", "9", "16"]
    assert rep.skipped_rows == 2
    assert "2 rows" in rep.warnings()[0]
    assert "they were left out" in rep.warnings()[0]


def test_one_skipped_row_reads_naturally() -> None:
    _, rep = load("a,b\n1,2\n3,4,5,6\n7,8\n")
    assert rep.warnings() == [
        "1 row had more values than the header has columns, so it was left out."
    ]


def test_short_rows_are_padded_and_counted() -> None:
    df, rep = load("a,b,c\n1,2,3\n4\n5,6,7\n")
    assert df.loc[1].tolist() == ["4", "", ""]
    assert rep.padded_rows == 1
    assert "1 row" in rep.warnings()[0]


@pytest.mark.parametrize(
    "text",
    [
        "a,b,c\n1,2,3\n4\n5,6,7\n",
        'a,b,c\n1,"x\ny",3\n4,5\n"q",\n\n  \n7,8,9\n',
        'a,b,c\n"multi\nline, with sep",2,3\n9\n',
    ],
)
def test_short_rows_fast_count_matches_the_csv_scan(text: str) -> None:
    """pyarrow counts short rows; the csv module is the reference (and the fallback)."""
    import csv

    reader = csv.reader(io.StringIO(text))
    next(reader)
    expected = sum(1 for r in reader if 0 < len(r) < 3 and any(c.strip() for c in r))
    assert L._short_rows_arrow(text, ",", 3) == expected
    assert L._short_rows(text, ",", 3) == expected


def test_clean_file_has_no_warnings() -> None:
    _, rep = load("a,b\n1,2\n3,4\n")
    assert rep.warnings() == []
    assert (rep.skipped_rows, rep.padded_rows, rep.junk_lines) == (0, 0, 0)


# ---------------------------------------------------------------- files that can't be read


@pytest.mark.parametrize("data", [b"", b"   \n\n", codecs.BOM_UTF8])
def test_empty_file_is_a_friendly_error(data: bytes) -> None:
    with pytest.raises(L.CsvLoadError, match="empty"):
        load(data)


def test_excel_file_is_a_friendly_error() -> None:
    with pytest.raises(L.CsvLoadError, match="Excel"):
        load(b"PK\x03\x04" + b"\x00" * 50)


def test_binary_file_is_a_friendly_error() -> None:
    with pytest.raises(L.CsvLoadError, match="text file"):
        load(b"\x89PNG\r\n\x1a\n\x00\x00\x00")


def test_missing_path_is_a_friendly_error(tmp_path: Path) -> None:
    with pytest.raises(L.CsvLoadError, match="Couldn't open"):
        L.read_csv_report(tmp_path / "nope.csv")


def test_csv_load_error_is_a_value_error() -> None:
    assert issubclass(L.CsvLoadError, ValueError)


def test_header_only_gives_empty_frame() -> None:
    df, _ = load("a;b;c\n")
    assert list(df.columns) == ["a", "b", "c"]
    assert len(df) == 0


# ---------------------------------------------------------------- inputs and contract


def test_reads_path_str_and_text_streams(tmp_path: Path) -> None:
    path = tmp_path / "x.csv"
    path.write_bytes(b"a;b\n1;2\n")
    for source in (path, str(path), io.StringIO("a;b\n1;2\n")):
        df, rep = L.read_csv_report(source)
        assert df.shape == (1, 2)
        assert rep.sep == ";"


def test_same_stream_can_be_read_twice() -> None:
    """Streamlit hands back the same upload object on every rerun."""
    stream = io.BytesIO(b"a,b\n1,2\n")
    assert L.read_csv_report(stream)[0].shape == L.read_csv_report(stream)[0].shape == (1, 2)


def test_cells_stay_raw_strings() -> None:
    df, _ = load("zip,price,blank\n0800,$5.00,\n02000,7,\n")
    assert df["zip"].tolist() == ["0800", "02000"]
    assert df["blank"].tolist() == ["", ""]
    assert not df.isna().any().any()


def test_header_names_are_trimmed() -> None:
    df, _ = load(" a , b\n1,2\n")
    assert list(df.columns) == ["a", "b"]


def test_trimming_never_makes_duplicate_names() -> None:
    df, _ = load("a,a ,b\n1,2,3\n")
    assert list(df.columns) == ["a", "a ", "b"]


def test_profiling_read_csv_raw_uses_the_robust_loader() -> None:
    df = P.read_csv_raw(io.BytesIO(b"Title line\na;b\n1;2\n3;4\n"))
    assert list(df.columns) == ["a", "b"]
    assert df.shape == (2, 2)
