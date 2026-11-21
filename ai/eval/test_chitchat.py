"""일상 대화(Gemini) 연결 확인. 진짜 Gemini·키·네트워크는 쓰지 않고 가짜 클라이언트로 호출 횟수와 보낸 내용을 본다."""
import os
import unittest
from pathlib import Path
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

import chitchat

APP = Path(__file__).resolve().parents[1] / "app.py"


class FakeResponse:
    def __init__(self, text):
        self.text = text


class FakeModels:
    def __init__(self, owner):
        self.owner = owner

    def generate_content(self, model, contents, config):
        self.owner.calls.append({"model": model, "contents": contents, "config": config})
        if self.owner.error:
            raise self.owner.error
        return FakeResponse(self.owner.answer)


class FakeClient:
    def __init__(self, answer="안녕하세요. 반가워요.", error=None):
        self.answer, self.error, self.calls = answer, error, []
        self.models = FakeModels(self)


class RuleTests(unittest.TestCase):
    def test_greeting_and_thanks_are_chitchat(self):
        for t in ("안녕하세요", "안녕하세요!", "고마워요", "고마워요 ㅎㅎ", "감사합니다", "잘 가요", "하이", "뭐해?"):
            self.assertTrue(chitchat.looks_like_chitchat(t), t)

    def test_welfare_distress_and_long_text_are_not(self):
        for t in ("월세가 걱정돼요", "일자리 알아보고 싶어요", "요즘 너무 힘들어요", "죽고 싶어요", "고마워요 근데 월세가 밀렸어요",
                  "안녕하세요 일하고 싶은데 사람 만나는 게 부담스러워요 도와주세요", "뭘 해야 할지 모르겠어요", "", "   "):
            self.assertFalse(chitchat.looks_like_chitchat(t), t)

    def test_key_from_secrets_then_env(self):
        with patch.dict(os.environ, {"GEMINI_API_KEY": "env-key"}):
            self.assertEqual(chitchat.resolve_api_key({"GEMINI_API_KEY": "secret-key"}), "secret-key")
            self.assertEqual(chitchat.resolve_api_key({}), "env-key")        # secrets에 없으면 환경변수
            self.assertEqual(chitchat.resolve_api_key(None), "env-key")
        with patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(chitchat.resolve_api_key(None))

    def test_model_is_configurable(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(chitchat.resolve_model(None), "gemini-3.5-flash-lite")
            self.assertEqual(chitchat.resolve_model({"GEMINI_MODEL": "other-model"}), "other-model")


class ReplyTests(unittest.TestCase):
    def run_reply(self, client, key="test-key", text="안녕하세요"):
        with patch.object(chitchat, "resolve_api_key", return_value=key), \
                patch.object(chitchat, "make_client", return_value=client):
            return chitchat.reply(text)

    def test_success_sends_only_this_sentence(self):
        fake = FakeClient("안녕하세요. 만나서 반가워요.")
        r = self.run_reply(fake)
        self.assertEqual((r.source, r.text), ("gemini", "안녕하세요. 만나서 반가워요."))
        self.assertEqual(len(fake.calls), 1)
        self.assertEqual(fake.calls[0]["contents"], "안녕하세요")
        self.assertTrue(fake.calls[0]["config"].system_instruction)

    def test_failures_fall_back(self):
        for name, fake, key in (("키 없음", FakeClient(), None), ("오류", FakeClient(error=RuntimeError("boom")), "k"),
                                ("시간 초과", FakeClient(error=TimeoutError()), "k"), ("빈 응답", FakeClient(answer=""), "k"),
                                ("None 응답", FakeClient(answer=None), "k")):
            r = self.run_reply(fake, key=key)
            self.assertEqual((r.source, r.text), ("fallback", chitchat.FALLBACK_BY_STAGE[3]), name)   # 단계를 모르면 3
            self.assertTrue(r.error, name)
        self.assertEqual(self.run_reply(FakeClient(), key=None).error, "no_key")

    def test_error_text_does_not_leak(self):
        r = self.run_reply(FakeClient(error=RuntimeError("secret-token-123")), key="k")
        self.assertNotIn("secret-token-123", r.error + r.text)

    def test_long_reply_is_cut(self):
        r = self.run_reply(FakeClient("가" * 1000))
        self.assertLessEqual(len(r.text), chitchat.MAX_REPLY)


class AppTests(unittest.TestCase):
    def app(self, scenario, fake, key="test-key"):
        self.fake = fake
        patches = [patch.dict(os.environ, {"BANGEOL_PIPELINE_MODE": "시연용 가짜"}),
                   patch.object(chitchat, "resolve_api_key", return_value=key),
                   patch.object(chitchat, "make_client", return_value=fake)]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        app = AppTest.from_file(str(APP), default_timeout=30)
        app.session_state["scenario"] = scenario
        return app.run()

    def say(self, app, text):
        app.chat_input[0].set_value(text).run()
        self.assertFalse(app.exception, [e.value for e in app.exception])
        return app

    def texts(self, app):
        return " ".join([m.value for m in app.markdown] + [c.value for c in app.caption])

    def test_greeting_gets_gemini_answer_and_conversation_continues(self):
        fake = FakeClient("안녕하세요. 반가워요.")
        app = self.say(self.app("욕구 없음", fake), "안녕하세요")
        self.assertEqual(len(fake.calls), 1)
        self.assertEqual(fake.calls[0]["contents"], "안녕하세요")
        self.assertIsNone(app.session_state["session"])
        text = self.texts(app)
        self.assertIn("안녕하세요\\. 반가워요\\.", text)
        self.assertIn("AI가 만든 답이에요.", text)
        self.assertIn("일상 대화는 외부 AI(Google Gemini)로 처리돼요.", text)          # 입력창 안내
        self.assertTrue(app.chat_input[0].placeholder)                                 # 입력은 계속 가능
        self.say(app, "고마워요")
        self.assertEqual(len(fake.calls), 2)
        contents = fake.calls[1]["contents"]                                          # 앞선 일상 대화 1턴 + 지금 문장
        self.assertEqual([c.parts[0].text for c in contents], ["안녕하세요", "안녕하세요. 반가워요.", "고마워요"])

    def test_crisis_never_reaches_gemini(self):
        for sentence in ("죽고 싶어요", "안녕하세요"):                                    # 가짜 위기 시나리오는 모든 입력을 위기로 본다
            fake = FakeClient()
            app = self.say(self.app("위기", fake), sentence)
            self.assertEqual(fake.calls, [], sentence)
            self.assertEqual(app.session_state["page"], "safety")

    def test_welfare_sentences_never_reach_gemini(self):
        for scenario, sentence in (("일반", "월세가 걱정돼요"), ("욕구 없음", "월세가 걱정돼요"), ("일반", "안녕하세요"),
                                   ("욕구 없음", "뭘 해야 할지 모르겠어요"), ("욕구 없음", "일하고 싶은데 부담돼요")):
            fake = FakeClient()
            app = self.say(self.app(scenario, fake), sentence)
            self.assertEqual(fake.calls, [], (scenario, sentence))

    def test_no_key_uses_default_answer_without_crash(self):
        fake = FakeClient()
        app = self.say(self.app("욕구 없음", fake, key=None), "안녕하세요")
        self.assertEqual(fake.calls, [])
        text = self.texts(app)
        self.assertIn("말씀해 주셔서 고마워요", text)
        self.assertNotIn("AI가 만든 답이에요.", text)                                    # 기본 답에는 붙이지 않음

    def test_api_error_and_timeout_use_default_answer(self):
        for error in (RuntimeError("boom"), TimeoutError()):
            app = self.say(self.app("욕구 없음", FakeClient(error=error)), "고마워요")
            self.assertIn("말씀해 주셔서 고마워요", self.texts(app))

    def test_general_scenario_greeting_offers_help_first(self):
        fake = FakeClient()
        app = self.say(self.app("일반", fake), "안녕하세요")                              # 일반 시나리오는 필요한 도움이 잡힘 → 신호 있음
        self.assertEqual(fake.calls, [])
        self.assertIsNone(app.session_state["session"])
        self.assertIsNotNone(app.session_state["pending_offer"])
        app.button(key="offer_yes").click().run()
        self.assertIsNotNone(app.session_state["session"])

    def test_welfare_after_chitchat_offers_help_and_keeps_bubbles(self):
        fake = FakeClient("안녕하세요. 반가워요.")
        app = self.say(self.app("욕구 없음", fake), "안녕하세요")
        self.assertEqual(len(fake.calls), 1)
        app = self.say(app, "월세가 걱정돼요")
        self.assertEqual(len(fake.calls), 1)                                            # 복지 문장은 호출 없음
        self.assertIsNone(app.session_state["session"])                                 # 먼저 도움 제안
        self.assertIsNotNone(app.session_state["pending_offer"])
        app.button(key="offer_yes").click().run()
        self.assertIsNotNone(app.session_state["session"])                              # 기존 흐름 시작
        self.assertTrue(any(m.get("chat") for m in app.session_state["log"]))          # 앞의 일상 대화 말풍선은 남음

    def test_about_page_mentions_external_ai(self):
        app = self.app("일반", FakeClient())
        app.button(key="nav_about").click().run()
        self.assertIn("외부 AI(Google Gemini)", " ".join(m.value for m in app.markdown))


if __name__ == "__main__":
    unittest.main()
