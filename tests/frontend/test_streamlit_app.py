"""Smoke tests for the Streamlit screen, driven headless with AppTest."""

import ast
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


def test_error_shows_friendly_message_not_traceback() -> None:
    at = _app()
    at.selectbox[0].select("iris.csv").run()
    at.session_state["raw_profile"] = at.session_state["raw_profile"].__class__(n_rows=0, n_cols=1)
    at.run()  # clean() now raises: the profile doesn't match the frame
    assert not at.exception
    assert any("Something went wrong" in e.value for e in at.error)
    assert any("ValueError" in e.value for e in at.error)


def test_same_source_is_not_read_again() -> None:
    at = _app()
    at.selectbox[0].select("iris.csv").run()
    report = at.session_state["csv_report"]
    assert report.sep == ","
    at.session_state["raw_df"] = at.session_state["raw_df"].head(10)
    at.run()
    assert len(at.session_state["raw_df"]) == 10  # kept, not re-read from disk


def test_clearing_the_source_drops_data_from_memory() -> None:
    at = _app()
    at.selectbox[0].select("messy.csv").run()
    at.selectbox[0].select("(none)").run()
    assert not at.exception
    assert "raw_df" not in at.session_state
    assert "clean_df" not in at.session_state
    assert "csv_report" not in at.session_state


# ---------------------------------------------------------------- Phase 2 rework (spec E)


def _texts(at: AppTest) -> list[str]:
    found = [str(e.value) for e in [*at.header, *at.subheader, *at.markdown, *at.caption]]
    return found + [str(e.label) for e in at.expander]


def test_clinic_has_outlier_and_imputer_state() -> None:
    at = _app()
    at.selectbox[0].select("messy_clinic.csv").run()
    assert not at.exception
    assert isinstance(at.session_state["outlier_choices"], dict)
    assert at.session_state["numeric_imputer"] in ("auto", "knn", "iterative")


def test_outliers_section_and_tree_caption() -> None:
    at = _app()
    at.selectbox[0].select("messy_clinic.csv").run()
    texts = _texts(at)
    assert any("Outlier" in t for t in texts), texts
    caption = [str(c.value) for c in at.caption]
    assert any("tree" in c.lower() for c in caption), caption
    assert not any("—" in t for t in texts)  # no em dashes in UI text


@pytest.mark.parametrize("choice", ["auto", "keep", "cap", "log", "missing"])
def test_outlier_choice_in_state_is_used(choice: str) -> None:
    at = _app()
    at.selectbox[0].select("messy_clinic.csv").run()
    at.session_state["outlier_choices"] = {"income": choice}
    at.run()
    assert not at.exception


@pytest.mark.parametrize("imputer", ["auto", "knn", "iterative"])
def test_imputer_choice_in_state_is_used(imputer: str) -> None:
    at = _app()
    at.selectbox[0].select("messy_clinic.csv").run()
    at.session_state["numeric_imputer"] = imputer
    at.run()
    assert not at.exception
    assert at.session_state["numeric_imputer"] == imputer


def test_outlier_choices_reset_with_the_dataset() -> None:
    at = _app()
    at.selectbox[0].select("messy_clinic.csv").run()
    at.session_state["outlier_choices"] = {"income": "cap"}
    at.run()
    at.selectbox[0].select("messy.csv").run()
    assert not at.exception
    assert at.session_state["outlier_choices"].get("income", "auto") == "auto"


def test_clinic_dates_stay_in_clean_df() -> None:
    at = _app()
    at.selectbox[0].select("messy_clinic.csv").run()
    clean_df = at.session_state["clean_df"]
    assert "visit_date" in clean_df.columns
    assert "visit_date_year" not in clean_df.columns  # date features live in the Pipeline now


def test_group_labels_cover_every_group() -> None:
    tree = ast.parse(Path(APP).read_text(encoding="utf-8"))
    labels = next(
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.AnnAssign | ast.Assign)
        and "GROUP_LABELS"
        in ast.unparse(node.targets[0] if isinstance(node, ast.Assign) else node.target)
    )
    assert isinstance(labels, ast.Dict)
    keys = {k.value for k in labels.keys if isinstance(k, ast.Constant)}
    assert keys == {"numeric", "dates", "was_missing", "boolean", "onehot", "frequency"}
    assert all("—" not in ast.unparse(v) for v in labels.values)
