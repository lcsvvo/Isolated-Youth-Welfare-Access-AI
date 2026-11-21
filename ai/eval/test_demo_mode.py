"""배포용 DEMO_MODE: 켜면 첫 화면부터 '실제 흐름(가짜 모델)'·개발자 메뉴 숨김, 끄면 기존과 같음. secrets 파일이 없어도 오류 없음."""
import os
import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock

from streamlit.testing.v1 import AppTest

AI = Path(__file__).resolve().parents[1]
APP = str(AI / "app.py")
DEPLOY = str(AI.parent / "deploy" / "streamlit_app.py")


def run(entry=APP, demo=None, query=None):
    env = {k: v for k, v in os.environ.items() if k not in ("DEMO_MODE", "BANGEOL_PIPELINE_MODE")}
    if demo is not None:
        env["DEMO_MODE"] = demo
    with mock.patch.dict(os.environ, env, clear=True):
        at = AppTest.from_file(entry, default_timeout=180)
        if query:
            at.query_params.update(query)
        return at.run()


def html(at):
    return " ".join(m.value for m in at.markdown)


class DemoModeTests(unittest.TestCase):
    def tearDown(self):
        import streamlit as st
        st.cache_resource.clear()          # 실제 흐름으로 만든 Pipeline이 다른 테스트로 새지 않게
    def test_off_is_unchanged(self):
        at = run()
        self.assertFalse(at.exception)
        self.assertEqual(at.session_state["pipe_choice"], "시연용 가짜")
        self.assertIn('href="?dev=1"', html(at))

    def test_off_dev_query_still_opens_dev_menu(self):
        at = run(query={"dev": "1"})
        self.assertTrue(any(b.key == "open_dev" or "개발자" in str(getattr(b, "label", "")) for b in at.button) or
                        "개발자 설정" in str(at.main))

    def test_on_starts_with_real_flow_and_hides_dev(self):
        for value in ("true", "1", "TRUE"):
            at = run(demo=value, query={"dev": "1"})
            self.assertFalse(at.exception, value)
            self.assertEqual(at.session_state["pipe_choice"], "실제 흐름(가짜 모델)")
            self.assertNotIn('href="?dev=1"', html(at))
            self.assertFalse([b for b in at.button if b.key == "open_dev"])

    def test_on_dev_page_redirects_home(self):
        at = run(demo="true")
        at.session_state["page"] = "dev"
        at.run()
        self.assertFalse(at.exception)
        self.assertNotIn("개발자 보기", html(at))

    def test_false_value_is_off(self):
        self.assertEqual(run(demo="false").session_state["pipe_choice"], "시연용 가짜")

    def test_deploy_entry_runs_without_secrets(self):
        at = run(entry=DEPLOY, demo="true")
        self.assertFalse(at.exception)
        self.assertEqual(at.session_state["pipe_choice"], "실제 흐름(가짜 모델)")

    def test_heavy_packages_not_imported_in_mock(self):
        code = ("import os,sys;os.environ['DEMO_MODE']='true';"
                "from streamlit.testing.v1 import AppTest;"
                f"AppTest.from_file(r'{DEPLOY}',default_timeout=180).run();"
                "print([m for m in ('torch','transformers','sentence_transformers') if m in sys.modules])")
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, encoding="utf-8").stdout
        self.assertIn("[]", out.strip().splitlines()[-1])

    def test_deploy_requirements_are_light(self):
        lines = (AI.parent / "deploy" / "requirements.txt").read_text(encoding="utf-8").splitlines()
        text = " ".join(l for l in lines if not l.lstrip().startswith("#"))
        for name in ("torch", "transformers", "sentence-transformers", "accelerate"):
            self.assertNotIn(name, text)
        self.assertIn("streamlit", text)


if __name__ == "__main__":
    unittest.main()
