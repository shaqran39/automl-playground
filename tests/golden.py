"""Snapshots of profile() and clean() on every sample CSV, for the regression test.

The golden files in tests/golden/ were written before the performance pass, so
any change in a profile, cleaning log or cleaned frame shows up as a diff.
Regenerate them only on purpose: python -m tests.golden
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

from app.core.cleaning import CleaningLog, clean
from app.core.profiling import profile, read_csv_raw

DATA = Path(__file__).parent.parent / "sample_data"
GOLDEN = Path(__file__).parent / "golden"
REFERENCE_DATE = "2025-01-01"
REL_TOL = 1e-9


def _frame_digest(frame: pd.DataFrame) -> dict[str, Any]:
    """Columns, dtypes and a hash of each column's values (index included)."""
    hashes = {
        str(c): hashlib.sha1(
            pd.util.hash_pandas_object(frame.iloc[:, i].astype(str), index=True).values.tobytes()
        ).hexdigest()
        for i, c in enumerate(frame.columns)
    }
    return {
        "shape": list(frame.shape),
        "columns": [str(c) for c in frame.columns],
        "dtypes": [str(t) for t in frame.dtypes],
        "index": hashlib.sha1(frame.index.astype(str).str.cat(sep="|").encode()).hexdigest(),
        "values": hashes,
    }


def _log(log: CleaningLog) -> dict[str, Any]:
    ids = None if log.row_ids is None else _frame_digest(log.row_ids.to_frame())
    return {"steps": [dataclasses.asdict(s) for s in log.steps], "row_ids": ids}


def snapshot(name: str) -> dict[str, Any]:
    raw = read_csv_raw(DATA / name)
    prof = profile(raw)
    out: dict[str, Any] = {"profile": dataclasses.asdict(prof)}
    clean_df, log = clean(raw, prof, reference_date=REFERENCE_DATE)
    out["clean"] = {"log": _log(log), "frame": _frame_digest(clean_df)}
    out["clean_profile"] = dataclasses.asdict(profile(clean_df))
    every = {s.step_id for s in log.steps if s.kind == "fix"}
    off_df, off_log = clean(raw, prof, disabled=every, reference_date=REFERENCE_DATE)
    out["all_off"] = {"log": _log(off_log), "frame": _frame_digest(off_df)}
    return json.loads(json.dumps(out, default=str))  # tuples -> lists, like the stored file


def golden_path(name: str) -> Path:
    return GOLDEN / f"{Path(name).stem}.json"


def diff(old: Any, new: Any, path: str = "") -> list[str]:
    """Paths where two snapshots differ. Floats match within REL_TOL, NaN matches NaN."""
    if isinstance(old, dict) and isinstance(new, dict):
        out = [f"{path}.{k}: missing" for k in old.keys() - new.keys()]
        out += [f"{path}.{k}: unexpected" for k in new.keys() - old.keys()]
        for k in old.keys() & new.keys():
            out += diff(old[k], new[k], f"{path}.{k}")
        return out
    if isinstance(old, list) and isinstance(new, list):
        if len(old) != len(new):
            return [f"{path}: length {len(old)} != {len(new)}"]
        return [
            d
            for i, (a, b) in enumerate(zip(old, new, strict=True))
            for d in diff(a, b, f"{path}[{i}]")
        ]
    if isinstance(old, float) and isinstance(new, float):
        if math.isnan(old) and math.isnan(new):
            return []
        return (
            []
            if math.isclose(old, new, rel_tol=REL_TOL, abs_tol=1e-12)
            else [f"{path}: {old} != {new}"]
        )
    return [] if old == new else [f"{path}: {old!r} != {new!r}"]


def write_all() -> None:
    GOLDEN.mkdir(exist_ok=True)
    for path in sorted(DATA.glob("*.csv")):
        golden_path(path.name).write_text(json.dumps(snapshot(path.name), indent=1), "utf-8")
        print("wrote", golden_path(path.name))


if __name__ == "__main__":
    write_all()
