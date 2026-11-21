"""Final runtime wiring, without model downloads, keys, or network calls."""
import json
import os
import runpy
import shutil
import tempfile
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from extract.postprocess import rule_constraints, access_clauses
from runtime_state import pipeline_inputs

AI = Path(__file__).resolve().parents[1]
PREP = runpy.run_path(str(AI / "scripts/prepare_runtime.py"))


def test_visit_constraint_fix_keeps_query_splitting():
    text = "전화는 괜찮지만 직접 찾아가서 상담받는 건 부담스러워요."
    assert [(v["channel"], v["level"]) for v in rule_constraints(text)] == [("방문", "부담")]
    assert "직접 찾아가서" in access_clauses(text)


def test_runtime_exports_rebuild_only_when_reviewed_inputs_change(tmp_path):
    master = tmp_path / "data/processed/services_master.csv"
    master.parent.mkdir(parents=True)
    shutil.copyfile(AI / "data/sample_services.csv", master)
    cfg_path = tmp_path / "ai/configs/config.yaml"
    cfg_path.parent.mkdir(parents=True)
    cfg_path.write_text(yaml.safe_dump({"models": {"llm": "kanana", "embedder": "bge"},
                                         "retrieval": {"dense_profile": "core"}}))
    source = tmp_path / "src/producer.py"
    source.parent.mkdir()
    source.write_text("version = 1")
    from core.data import load_services
    ids = list(load_services(master).servId)

    def export(command, **kwargs):
        kind = next(k for k, (module, _) in PREP["ADDENDA"].items() if command[-1] == module)
        path = tmp_path / PREP["ADDENDA"][kind][1]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"schema": f"{kind}-{'review' if kind == 'q1' else 'info'}-v1",
                                    "master_sha256": PREP["sha256"](master),
                                    "services": {sid: {} for sid in ids} if kind == "q1" else {}}))

    with patch("subprocess.run", side_effect=export) as run:
        first = PREP["prepare"](tmp_path)
        assert run.call_count == 2
        assert PREP["prepare"](tmp_path) == first
        assert run.call_count == 2
        source.write_text("version = 2")
        PREP["prepare"](tmp_path)
        assert run.call_count == 4
        (tmp_path / PREP["ADDENDA"]["q2"][1]).write_text("corrupt")
        PREP["prepare"](tmp_path)
        assert run.call_count == 5


def test_pipeline_cache_tracks_master_and_addendum_changes(tmp_path):
    master, q1 = tmp_path / "master.csv", tmp_path / "q1.json"
    master.write_text("first")
    q1.write_text("first")
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump({"paths": {"services": str(master), "q1_review": str(q1)}}))
    before = pipeline_inputs(config)[2]
    q1.write_text("updated reviewed data")
    assert pipeline_inputs(config)[2] != before


def test_final_runtime_refuses_missing_addenda_before_loading_models(tmp_path):
    from pipeline import Pipeline
    config = yaml.safe_load((AI / "configs/config.yaml").read_text(encoding="utf-8"))
    config["paths"].update(q1_review=None, q2_info=None)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf-8")
    with patch.dict(os.environ, {"BANGEOL_REQUIRE_REVIEW_DATA": "1"}), patch("pipeline.load_models") as load:
        with pytest.raises(FileNotFoundError, match="prepare_runtime"):
            Pipeline(config_path=path, mock=True)
        load.assert_not_called()


def test_ui_uses_reviewed_domains_and_source_comparison_without_search():
    from core.data import load_services
    from streamlit.testing.v1 import AppTest
    from core.review_data import Q2_CARD_LABELS
    services = load_services(AI / "data/sample_services.csv").set_index("servId", drop=False)
    # The raw labels intentionally lack the approved domain, as in the final
    # labor/future review. Browsing must use the connected review layer.
    services["q1_core_domains"] = [("미래설계",) if i == 0 else () for i in range(len(services))]
    sid = services.index[0]
    item = {"status": "supplement_possible", "status_label": Q2_CARD_LABELS["supplement_possible"],
            "ssis": {"source_fields": {"apply_text_raw": "복지로 원문 근거"}, "url": "https://www.bokjiro.go.kr/"},
            "gov24": {"source_fields": {"신청방법": "정부24 원문 근거"}, "url": "https://www.gov.kr/"}}
    review = SimpleNamespace(card_fields=lambda selected: {"q2_source_comparison": {
        "same_program_confirmed": True, "items": {"channel": item}}})
    backend = SimpleNamespace(services=services, review=review, terms={"domains": {}}, domains=["미래설계"], sgg_map={})
    with patch.dict(os.environ, {"BANGEOL_PIPELINE_MODE": "실제 흐름(가짜 모델)"}), patch("pipeline.Pipeline", return_value=backend):
        app = AppTest.from_file(str(AI / "app.py"), default_timeout=30).run()
        app.button(key="nav_explore").click().run()
        assert any(b.label == "진로 · 1개" for b in app.button)
        labels = [b.label for b in app.button]
        assert "일상 & 여가 · 0개" in labels and "근로 & 권익 · 0개" in labels   # 바뀐 둘러보기 라벨
        app.session_state["selected"] = sid
        app.session_state["page"] = "detail"
        app.run()
        assert not app.exception
        assert any("복지로 원문 근거" in m.value for m in app.markdown)
        assert any("정부24 원문 근거" in m.value for m in app.markdown)
