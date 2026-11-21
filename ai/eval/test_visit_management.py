"""Deleting saved services changes only this visit, never service data."""
import os
from pathlib import Path
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / "app.py"


def saved_app():
    with patch.dict(os.environ, {"BANGEOL_PIPELINE_MODE": "시연용 가짜"}):
        app = AppTest.from_file(str(APP), default_timeout=30).run()
    app.button(key="nav_explore").click().run()
    ids = [button.key.split("_", 2)[2] for button in app.button
           if button.key and button.key.startswith("open_")][:3]
    assert len(ids) == 3
    app.session_state["visit"] = ids
    app.session_state["page"] = "visit"
    app.run()
    assert not app.exception
    return app, ids


def test_delete_selected_preserves_other_saved_services_and_master():
    app, ids = saved_app()
    assert app.button(key="visit_remove_selected").disabled
    app.checkbox(key=f"visit_select_{ids[0]}").check().run()
    assert app.button(key="visit_remove_selected").label == "삭제 (1)"
    assert len([b for b in app.button if b.label.startswith("삭제")]) == 1
    app.button(key="visit_remove_selected").click().run()
    assert not app.exception
    assert app.session_state["visit"] == ids[1:]
    app.button(key="nav_explore").click().run()
    assert len([b for b in app.button if b.key and b.key.startswith("open_")]) == 5


def test_one_delete_button_handles_partial_and_all_selection():
    app, ids = saved_app()
    app.checkbox(key="visit_select_all").check().run()
    assert all(app.checkbox(key=f"visit_select_{sid}").value for sid in ids)
    app.checkbox(key=f"visit_select_{ids[0]}").uncheck().run()
    assert not app.checkbox(key="visit_select_all").value
    assert app.button(key="visit_remove_selected").label == "삭제 (2)"
    app.button(key="visit_remove_selected").click().run()
    assert app.session_state["visit"] == [ids[0]]
    app.checkbox(key="visit_select_all").check().run()
    app.button(key="visit_remove_selected").click().run()
    assert not app.exception
    assert app.session_state["visit"] == []
    assert any("아직 담은 서비스가 없어요" in m.value for m in app.markdown)
    app.button(key="visit_browse").click().run()
    assert app.session_state["page"] == "explore"
