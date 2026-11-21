"""결과·상세 화면이 NaN·None·빈칸이 든 서비스 값에서도 끝까지 그려지는지 확인. 모델·GPU 필요 없음."""
import copy
import os
import unittest
from pathlib import Path
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from confirm.session import new_session
from extract.schema import empty_record

APP = Path(__file__).resolve().parents[1] / "app.py"
NAN = float("nan")
EMPTY_CARD = {"reason": NAN, "how_to_apply": NAN, "check_needed": NAN, "evidence": NAN, "fallback": False,
              "explanation_pending": True, "explanation_kind": "source", "checks": {}, "check_reasons": {},
              "servNm": NAN, "region_sido": NAN, "source": NAN, "summary_text": NAN}
# 카드마다 비는 칸을 달리한다 (실제 사례: WLF00001160은 how_to_apply가 NaN)
VARIANTS = [
    {**EMPTY_CARD},                                                            # 전부 NaN
    {"reason": "설명 문장이에요.", "how_to_apply": NAN, "check_needed": ["확인할 것", NAN, ""]},
    {"reason": None, "how_to_apply": "", "check_needed": None},
    {"reason": "설명 문장이에요.", "how_to_apply": "방문 신청", "check_needed": "문자열 하나"},
    {"reason": NAN, "how_to_apply": "방문 신청", "check_needed": []},
]


def ready_session():
    record = empty_record()
    record["needs"] = [{"domain": "경제", "level": "main", "status": "명시", "evidence": "생활비"}]
    s = new_session("생활비", {"crisis": False}, record)
    s["status"] = "ready"
    return s


def cards():
    out = []
    for i, v in enumerate(VARIANTS):
        base = {**copy.deepcopy(EMPTY_CARD), "explanation_pending": True, "explanation_kind": "source"}
        base.update(v)
        base["servId"] = f"MOCK{i:03d}"
        base.setdefault("servNm", f"[서비스 이름 {i + 1}]")
        if i:
            base["servNm"] = f"[서비스 이름 {i + 1}]"
        out.append(base)
    return out


class EmptyValueTests(unittest.TestCase):
    def app(self, **state):
        with patch.dict(os.environ, {"BANGEOL_PIPELINE_MODE": "시연용 가짜"}):
            app = AppTest.from_file(str(APP), default_timeout=30)
            for k, v in state.items():
                app.session_state[k] = v
            return app.run()

    def test_results_render_with_empty_values(self):
        out = {"status": "ready", "searched": True, "crisis": {"crisis": False}, "session": ready_session(),
               "cards": cards(), "excluded": [], "retrieval_status": "ok", "retrieval_errors": [], "ranks": {}}
        app = self.app(page="results", session=ready_session(), out=out, outcome="results")
        self.assertFalse(app.exception, [e.value for e in app.exception])
        self.assertEqual(len([b for b in app.button if b.label.startswith("자세히 보기")]), len(VARIANTS))
        text = " ".join(m.value for m in app.markdown)
        self.assertNotIn("nan", text.replace("finan", ""))          # 'nan' 글자가 그대로 보이면 안 됨
        # 결과 카드에는 신청 방법 줄이 없고(4번은 팀원 확인 전, 현재 화면 기준) 신청 방법은 상세 화면에서 확인한다

    def test_detail_renders_with_empty_values(self):
        for card in cards():
            out = {"status": "ready", "searched": True, "crisis": {"crisis": False}, "session": ready_session(),
                   "cards": [card], "excluded": [], "retrieval_status": "ok", "retrieval_errors": [], "ranks": {}}
            app = self.app(page="detail", session=ready_session(), out=out, outcome="results",
                           selected=card["servId"], detail_from="results")
            self.assertFalse(app.exception, (card["servId"], [e.value for e in app.exception]))
            self.assertTrue(any("신청 방법" in m.value for m in app.markdown), card["servId"])   # 상세 화면에 신청 방법이 있다


if __name__ == "__main__":
    unittest.main()
