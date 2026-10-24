"""Smoke tests for the Streamlit screen, driven headless with AppTest."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from tests.conftest import SAMPLE_NAMES

APP = str(Path(__file__).parents[2] / "frontend" / "streamlit_app.py")


def _app() -> AppTest:
    return AppTest.from_file(APP, default_timeout=60).run()


def test_starts_empty() -> None:
    at = _app()
    assert not at.exception
    assert "raw_df" not in at.session_state


@pytest.mark.parametrize("name", SAMPLE_NAMES)
def test_each_sample_renders(name: str) -> None:
    at = _app()
    at.selectbox[0].select(name).run()
    assert not at.exception
    assert at.session_state["dataset_name"] == name
    assert len(at.dataframe) >= 3


def test_messy_keeps_row_ids_out_of_features() -> None:
    at = _app()
    at.selectbox[0].select("messy.csv").run()
    clean_df = at.session_state["clean_df"]
    log = at.session_state["cleaning_log"]
    assert "customer_id" not in clean_df.columns
    assert log.row_ids is not None
    assert log.row_ids.index.equals(clean_df.index)


def test_disabled_step_in_state_is_applied() -> None:
    at = _app()
    at.selectbox[0].select("messy.csv").run()
    at.session_state["disabled_steps"] = {"merge_case:state"}
    at.run()
    assert not at.exception
    assert "nsw" in set(at.session_state["clean_df"]["state"].dropna())


def test_clearing_the_source_drops_data_from_memory() -> None:
    at = _app()
    at.selectbox[0].select("messy.csv").run()
    at.selectbox[0].select("(none)").run()
    assert not at.exception
    assert "raw_df" not in at.session_state
    assert "clean_df" not in at.session_state
