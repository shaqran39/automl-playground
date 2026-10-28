"""Read any real-world CSV into an all-string frame without crashing.

Real files come with semicolons, tabs or pipes, Excel's latin-1, a BOM, title
lines above the header and the odd broken row. read_csv_report sniffs the
encoding, separator and header row from a sample, parses once, skips rows that
don't fit, and says what it did. Only files that can't be read at all raise
CsvLoadError, with a message that's safe to show in the UI.
"""

from __future__ import annotations

import codecs
import csv
import io
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import IO

import pandas as pd

SEPARATORS = (",", ";", "\t", "|")
SEPARATOR_NAMES = {",": "commas", ";": "semicolons", "\t": "tabs", "|": "pipes"}
SAMPLE_CHARS = 200_000
MAX_JUNK_LINES = 30
CHECK_RECORDS = 50
CONSISTENT_SHARE = 0.8

Source = str | Path | IO[str] | IO[bytes]


class CsvLoadError(ValueError):
    """The file can't be read as a table. The message is plain English for the UI."""


@dataclass
class CsvReport:
    encoding: str
    sep: str
    junk_lines: int = 0  # lines above the real header that were skipped
    skipped_rows: int = 0  # rows with too many fields, dropped
    padded_rows: int = 0  # rows with too few fields, filled with blanks

    def notes(self) -> list[str]:
        """Things worth knowing that didn't lose any data."""
        out = []
        if self.sep != ",":
            out.append(f"Columns are split by {SEPARATOR_NAMES[self.sep]}, not commas.")
        if self.encoding not in ("utf-8", "utf-8-sig", "text"):
            out.append(f"The file isn't UTF-8, so it was read as {self.encoding}.")
        if self.junk_lines:
            out.append(f"Skipped {_lines(self.junk_lines)} above the header.")
        return out

    def warnings(self) -> list[str]:
        """Rows that didn't fit the header."""
        out = []
        if self.skipped_rows:
            it = "it was" if self.skipped_rows == 1 else "they were"
            out.append(
                f"{_rows(self.skipped_rows)} had more values than the header has columns, "
                f"so {it} left out."
            )
        if self.padded_rows:
            out.append(
                f"{_rows(self.padded_rows)} had fewer values than the header, "
                "so the missing ones are blank."
            )
        return out


def _rows(n: int) -> str:
    return "1 row" if n == 1 else f"{n:,} rows"


def _lines(n: int) -> str:
    return "1 line" if n == 1 else f"{n} lines"


# ---------------------------------------------------------------- bytes -> text


def _read_source(source: Source) -> bytes | str:
    if isinstance(source, str | Path):
        try:
            return Path(source).read_bytes()
        except OSError as e:
            raise CsvLoadError("Couldn't open that file. Check the path and try again.") from e
    if hasattr(source, "seek"):
        source.seek(0)  # streamlit hands back the same upload object on every rerun
    return source.read()


def _decode(data: bytes) -> tuple[str, str]:
    """(text, encoding). Tries UTF-16 by BOM, then utf-8-sig, utf-8, cp1252, latin-1."""
    if data.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)):
        return data.decode("utf-16"), "utf-16"
    if data.startswith(b"PK\x03\x04"):
        raise CsvLoadError(
            "This looks like an Excel file, not a CSV. Open it in Excel and use "
            "Save As > CSV, then upload that."
        )
    if b"\x00" in data[:8192]:
        raise CsvLoadError(
            "This doesn't look like a text file. Make sure it's a CSV, not a binary file."
        )
    if data.startswith(codecs.BOM_UTF8):
        return data.decode("utf-8-sig"), "utf-8-sig"
    for encoding in ("utf-8", "cp1252"):
        try:
            return data.decode(encoding), encoding
        except UnicodeDecodeError:
            continue
    return data.decode("latin-1"), "latin-1"  # never fails: every byte is a character


# ---------------------------------------------------------------- sniffing


@dataclass
class _Record:
    line: int  # 0-based physical line the record starts on
    width: int
    last_empty: bool


@dataclass
class _Guess:
    sep: str
    header: int  # index into records
    share: float  # share of the following records that match the header width
    width: int
    matches: int  # how many of the following records match


def _sample(text: str) -> str:
    if len(text) <= SAMPLE_CHARS:
        return text
    cut = text.rfind("\n", 0, SAMPLE_CHARS)
    return text[: cut + 1] if cut > 0 else text[:SAMPLE_CHARS]


def _records(sample: str, sep: str) -> list[_Record]:
    """Quote-aware records in the sample. Blank lines are left out."""
    reader = csv.reader(io.StringIO(sample), delimiter=sep, quotechar='"')
    out: list[_Record] = []
    start = 0
    try:
        for row in reader:
            if any(cell.strip() for cell in row):
                out.append(_Record(start, len(row), row[-1].strip() == ""))
            start = reader.line_num
    except csv.Error:  # a broken quote in the sample: judge on what came before it
        pass
    return out


def _fits(rec: _Record, width: int) -> bool:
    """Same width, or one extra empty field from a separator at the end of the line."""
    return rec.width == width or (rec.width == width + 1 and rec.last_empty)


def _candidate(records: list[_Record], i: int, sep: str) -> _Guess:
    """Header guess at record i. A lone last line only counts when it's the whole file."""
    rest = records[i + 1 : i + 1 + CHECK_RECORDS]
    matches = sum(_fits(r, records[i].width) for r in rest)
    if rest:
        share = matches / len(rest)
    else:
        share = 1.0 if i == 0 else 0.0
    return _Guess(sep, i, share, records[i].width, matches)


def _guess(records: list[_Record], sep: str) -> _Guess | None:
    """First record in the top lines that the rows below agree with.

    If none is consistent enough (lots of broken rows), the one most rows agree with.
    """
    candidates = [
        _candidate(records, i, sep)
        for i, r in enumerate(records)
        if r.line < MAX_JUNK_LINES and r.width >= 2
    ]
    if not candidates:
        return None
    good = [g for g in candidates if g.share >= CONSISTENT_SHARE]
    return good[0] if good else max(candidates, key=lambda g: (g.matches, -g.header))


def _sniff(sample: str) -> tuple[str, int]:
    """(separator, physical line of the header).

    Most consistent rows win, then the header nearest the top, then more columns.
    """
    found: list[tuple[tuple[float, int, int], str, int]] = []
    for sep in SEPARATORS:
        records = _records(sample, sep)
        guess = _guess(records, sep)
        if guess is not None:
            line = records[guess.header].line
            found.append(((round(guess.share, 2), -line, guess.width), sep, line))
    if found:
        _, sep, line = max(found, key=lambda f: f[0])
        return sep, line
    records = _records(sample, ",")  # one column, or nothing but blank lines
    return ",", records[0].line if records else 0


def _line_offset(text: str, line: int) -> int:
    pos = 0
    for _ in range(line):
        nxt = text.find("\n", pos)
        if nxt < 0:
            return len(text)
        pos = nxt + 1
    return pos


# ---------------------------------------------------------------- parsing


def _parse(text: str, sep: str) -> tuple[pd.DataFrame, int]:
    """(frame, rows skipped for having too many fields)."""
    args = dict(sep=sep, dtype=str, keep_default_na=False, index_col=False, quotechar='"')
    try:
        with warnings.catch_warnings():  # a separator at the end of each line is fine
            warnings.simplefilter("ignore", pd.errors.ParserWarning)
            return pd.read_csv(io.StringIO(text), **args), 0
    except pd.errors.ParserError:
        pass  # some rows don't fit: redo it slowly, counting the ones we skip
    skipped: list[int] = []

    def on_bad(_fields: list[str]) -> None:
        skipped.append(1)

    try:
        frame = pd.read_csv(io.StringIO(text), engine="python", on_bad_lines=on_bad, **args)
    except (pd.errors.ParserError, csv.Error) as e:
        raise CsvLoadError(
            "Couldn't split this file into rows. Often a quote mark (\") is opened and "
            "never closed. Try re-saving it from Excel or Google Sheets as CSV."
        ) from e
    return frame, len(skipped)


def _short_rows(text: str, sep: str, width: int) -> int:
    """Rows with fewer fields than the header. pandas pads them with "" without saying so."""
    reader = csv.reader(io.StringIO(text), delimiter=sep, quotechar='"')
    try:
        next(reader, None)  # the header
        return sum(1 for row in reader if 0 < len(row) < width and any(c.strip() for c in row))
    except csv.Error:
        return 0  # a broken quote: pandas already coped, so just don't count


def _tidy(frame: pd.DataFrame) -> pd.DataFrame:
    """Trim spaces around header names, unless that would make two names the same."""
    raw = [str(c) for c in frame.columns]
    trimmed = [c.strip() for c in raw]
    frame.columns = trimmed if len(set(trimmed)) == len(trimmed) else raw
    return frame.fillna("")


def read_csv_report(source: Source) -> tuple[pd.DataFrame, CsvReport]:
    """Read a CSV with every cell as a raw string (blanks stay "") and report what was done.

    Raises CsvLoadError only when the file can't be read as a table at all.
    """
    data = _read_source(source)
    text, encoding = (data, "text") if isinstance(data, str) else _decode(data)
    text = text.removeprefix("﻿")
    if not text.strip():
        raise CsvLoadError("The file is empty.")
    sep, header_line = _sniff(_sample(text))
    offset = _line_offset(text, header_line)
    junk = sum(1 for line in text[:offset].splitlines() if line.strip())
    body = text[offset:]
    try:
        frame, skipped = _parse(body, sep)
    except pd.errors.EmptyDataError as e:
        raise CsvLoadError("The file has no columns to read.") from e
    padded = _short_rows(body, sep, frame.shape[1])
    return _tidy(frame), CsvReport(encoding, sep, junk, skipped, padded)


def read_csv_raw(source: Source) -> pd.DataFrame:
    """Read a CSV with every cell kept as a raw string (blanks stay "")."""
    return read_csv_report(source)[0]
