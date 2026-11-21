"""AI 설명이 원문 안내로 바뀔 때 이유가 로그 한 줄로 남는지 확인. 가짜 모델만 쓰고 GPU·네트워크는 필요 없다."""
import logging
import os
import unittest
from pathlib import Path
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from core.models import LLM
from explain.explain import explain

SVC = {"servId": "WLF00000001", "servNm": "테스트 서비스", "target_text": "만 19세 이상 청년에게 지원합니다.",
       "criteria_text": "소득 기준 없음", "benefit_text": "월 20만원을 지원합니다.", "apply_text": "주민센터 방문 신청"}
SOURCE_SENTENCE = "만 19세 이상 청년에게 지원합니다."
USER_VIEW = {"needs": [{"domain": "경제", "evidence": "월세가 밀렸어요 비밀문장"}], "activity_range": "집 안"}
CHECKS = {"연령": "확인 필요", "지역": "통과"}
GOOD = {"reason": "월 20만원을 지원받을 수 있는 서비스예요.", "how_to_apply": "주민센터에 방문해 주세요.",
        "check_needed": ["나이"], "evidence": SOURCE_SENTENCE}


class FakeLLM:
    """chat_json 호출마다 answers를 차례로 돌려주거나, Exception이면 던진다."""

    def __init__(self, *answers):
        self.answers, self.calls = list(answers), 0

    def chat_json(self, system, user, task=None, retries=1):
        self.calls += 1
        a = self.answers[min(self.calls - 1, len(self.answers) - 1)]
        if isinstance(a, Exception):
            raise a
        return a


class RawTextModel:
    """진짜 LLM.chat_json을 가짜 출력으로 돌려 JSON 파싱 실패 경로를 만든다."""
    def __init__(self, raw):
        self.raw = raw

    def chat(self, system, user, max_new_tokens=512):
        return self.raw

    chat_json = LLM.chat_json


class ExplainLoggingTests(unittest.TestCase):
    def run_explain(self, llm):
        with self.assertLogs("explain", level="INFO") as cm:
            card = explain(dict(SVC), USER_VIEW, CHECKS, llm)
        return card, cm.output

    def reasons(self, lines):
        return " | ".join(lines)

    def test_exception_logs_type_only(self):
        card, lines = self.run_explain(FakeLLM(RuntimeError("개인 입력 비밀문장이 섞인 메시지")))
        self.assertTrue(card["fallback"])
        text = self.reasons(lines)
        self.assertIn("[explain] servId=WLF00000001 attempt=1/2 reason=exception:RuntimeError", text)
        self.assertIn("attempt=2/2 reason=exception:RuntimeError", text)
        self.assertIn("reason=fallback_used", text)
        self.assertNotIn("비밀문장", text)                      # 예외 메시지는 남기지 않음

    def test_timeout_like_error_is_named(self):
        _, lines = self.run_explain(FakeLLM(TimeoutError("x")))
        self.assertIn("reason=exception:TimeoutError", self.reasons(lines))

    def test_json_parse_failure(self):
        card, lines = self.run_explain(RawTextModel("죄송하지만 JSON을 만들지 못했어요"))
        self.assertTrue(card["fallback"])
        self.assertIn("reason=exception:ValueError", self.reasons(lines))

    def test_truncated_json_failure(self):
        _, lines = self.run_explain(RawTextModel('{"reason": "끊긴 출력", "evidence": "만 19'))
        self.assertRegex(self.reasons(lines), r"reason=exception:(JSONDecodeError|ValueError)")

    def test_validation_evidence_not_in_source(self):
        bad = {**GOOD, "evidence": "원문에 없는 문장 만 30세 이하"}
        card, lines = self.run_explain(FakeLLM(bad))
        self.assertTrue(card["fallback"])
        text = self.reasons(lines)
        self.assertIn("reason=validation: evidence_not_in_source", text)
        self.assertNotIn("원문에 없는 문장", text)                # 모델 출력 원문은 남기지 않음

    def test_validation_evidence_empty(self):
        _, lines = self.run_explain(FakeLLM({**GOOD, "evidence": "  "}))
        self.assertIn("reason=validation: evidence_empty", self.reasons(lines))

    def test_bad_format(self):
        _, lines = self.run_explain(FakeLLM(["목록을 반환"]))
        self.assertIn("reason=bad_format: not_an_object", self.reasons(lines))
        _, lines = self.run_explain(FakeLLM({**GOOD, "evidence": 123}))
        self.assertIn("reason=bad_format: evidence_not_text", self.reasons(lines))

    def test_retry_success_keeps_first_failure_log_and_no_fallback(self):
        llm = FakeLLM({**GOOD, "evidence": "없는 문장"}, GOOD)
        card, lines = self.run_explain(llm)
        self.assertFalse(card["fallback"])
        text = self.reasons(lines)
        self.assertIn("attempt=1/2 reason=validation: evidence_not_in_source", text)
        self.assertNotIn("fallback_used", text)
        self.assertEqual(llm.calls, 2)

    def test_success_logs_nothing(self):
        with self.assertNoLogs("explain", level="INFO"):
            card = explain(dict(SVC), USER_VIEW, CHECKS, FakeLLM(GOOD))
        self.assertFalse(card["fallback"])

    def test_logs_never_contain_user_text_source_or_output(self):
        _, lines = self.run_explain(FakeLLM({**GOOD, "reason": "모델이쓴설명XYZ", "evidence": "틀린근거문장"}))
        text = self.reasons(lines)
        for secret in ("비밀문장", "월세가 밀렸어요", "월 20만원", "주민센터", "모델이쓴설명XYZ", "틀린근거문장", "테스트 서비스"):
            self.assertNotIn(secret, text)
        self.assertIn("elapsed_ms=", text)


class AppLoggingSetupTests(unittest.TestCase):
    def test_app_sets_up_explain_logger_once(self):
        app_path = str(Path(__file__).resolve().parents[1] / "app.py")
        with patch.dict(os.environ, {"BANGEOL_PIPELINE_MODE": "시연용 가짜"}):
            for _ in range(2):          # 스크립트가 다시 돌아도 핸들러가 늘지 않음
                AppTest.from_file(app_path, default_timeout=30).run()
        lg = logging.getLogger("explain")
        self.assertEqual(len([h for h in lg.handlers if isinstance(h, logging.StreamHandler)]), 1)
        self.assertLessEqual(lg.level, logging.INFO)


if __name__ == "__main__":
    unittest.main()
