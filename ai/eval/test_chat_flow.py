"""챗봇 흐름: 위기 우선 → 일상 대화(Gemini) / 도움 제안 → 기존 흐름, 활동 범위 단계별 말투, 건너뛴 질문 재질문 방지.
가짜 Gemini와 시연용 가짜 파이프라인만 쓴다(키·네트워크·GPU 불필요)."""
import copy
import logging
import os
import re
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml
from streamlit.testing.v1 import AppTest

import chitchat
from confirm import questions as Q
from confirm.session import new_session
from extract.schema import empty_record

AI = Path(__file__).resolve().parents[1]
APP = str(AI / "app.py")
SIGNAL = "월세가 걱정돼요"
FORBIDDEN_FOR_STAGE3 = re.compile(r"외출|나가|만나|연락|전화|방문")
# 건너뛰기 버튼 문구는 화면의 SKIP_QUESTION 상수를 그대로 읽는다(app.py는 스크립트라 import할 수 없음). 문구가 바뀌어도 테스트는 안 깨진다
SKIP_QUESTION = re.search(r'^SKIP_QUESTION = "(.+)"', Path(APP).read_text(encoding="utf-8"), re.M).group(1)


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
    def __init__(self, answer="네, 반가워요.", error=None):
        self.answer, self.error, self.calls = answer, error, []
        self.models = FakeModels(self)


def texts_of(contents):
    """보낸 내용을 글자 목록으로 (문자열 하나이거나 Content 목록)."""
    if isinstance(contents, str):
        return [contents]
    return [c.parts[0].text for c in contents]


class StageRuleTests(unittest.TestCase):
    def test_stage_mapping(self):
        self.assertEqual([chitchat.care_stage(v) for v in ("필요하면 외출", "집 근처", "집 안", "방 안")], [0, 1, 2, 3])
        for unknown in ("판단 불가", None, "", "이상한 값"):
            self.assertEqual(chitchat.care_stage(unknown), 3)       # 모르면 가장 조심스럽게

    def test_system_prompt_has_common_rules_and_only_the_stage_block(self):
        for stage, marker in ((0, "가볍게"), (1, "한 번에 하나만 해요"), (2, "괜찮아요"), (3, "권하지 않아요")):
            text = chitchat.system_prompt(stage)
            self.assertIn("의료·법률", text)
            self.assertIn(f"[말투 단계 {stage}]", text)
            self.assertIn(marker, text)
            for other in {0, 1, 2, 3} - {stage}:
                self.assertNotIn(f"[말투 단계 {other}]", text)
        self.assertIn("[말투 단계 3]", chitchat.system_prompt(None))  # 단계를 모르면 3

    def test_stage3_fixed_texts_are_gentle(self):
        for text in (chitchat.OFFER_QUESTION, chitchat.OFFER_NOTE[3], *chitchat.TALK_REPLIES[3], chitchat.FALLBACK_BY_STAGE[3]):
            self.assertIsNone(FORBIDDEN_FOR_STAGE3.search(text), text)     # 외출·연락 권유 없음
        self.assertIn("답하지 않아도 괜찮아요", chitchat.OFFER_NOTE[3])        # 질문(제안)에는 안내가 붙음
        for stage in range(4):
            replies = chitchat.TALK_REPLIES[stage]
            self.assertGreaterEqual(len(replies), 3)                          # 단계별 여러 개를 돌아가며
            self.assertEqual(len(set(replies)), len(replies))
            for text in replies:
                self.assertLessEqual(len(text), 60)
                self.assertNotIn("답하지 않아도", text)                         # 질문이 아닌 말에는 안내를 붙이지 않음
                self.assertNotIn("?", text)
        self.assertNotIn("답하지 않아도", chitchat.FALLBACK_BY_STAGE[3])
        self.assertTrue(any("괜찮아요" in t for t in chitchat.TALK_REPLIES[2]))

    def test_talk_replies_rotate(self):
        seen = [chitchat.talk_reply_text(3, n) for n in range(6)]
        self.assertEqual(seen[:3], seen[3:])                                  # 3개를 돌아감
        self.assertTrue(all(seen[i] != seen[i + 1] for i in range(5)))        # 연속으로 같지 않음
        self.assertEqual(chitchat.talk_reply_text(None, 0), chitchat.TALK_REPLIES[3][0])

    def test_stage_goes_into_gemini_instruction_and_history_is_capped(self):
        fake = FakeClient("답")
        history = [[f"질문{i}", f"답{i}"] for i in range(6)]
        with patch.object(chitchat, "resolve_api_key", return_value="k"), patch.object(chitchat, "make_client", return_value=fake):
            chitchat.reply("안녕", stage=2, history=history)
        call = fake.calls[0]
        self.assertIn("[말투 단계 2]", call["config"].system_instruction)
        sent = texts_of(call["contents"])
        self.assertEqual(sent, ["질문3", "답3", "질문4", "답4", "질문5", "답5", "안녕"])      # 최근 3턴 + 지금 문장

    def test_private_or_long_text_is_not_sent(self):
        fake = FakeClient()
        with patch.object(chitchat, "resolve_api_key", return_value="k"), patch.object(chitchat, "make_client", return_value=fake):
            for t in ("전화는 010-1234-5678 이에요", "메일 test@example.com", "9001011234567", "가" * 201):
                r = chitchat.reply(t, stage=0)
                self.assertEqual((r.source, r.error), ("fallback", "not_sendable"), t)
        self.assertEqual(fake.calls, [])


class FlowBase(unittest.TestCase):
    def app(self, scenario, fake=None, key="test-key"):
        self.fake = fake or FakeClient()
        for p in (patch.dict(os.environ, {"BANGEOL_PIPELINE_MODE": "시연용 가짜"}),
                  patch.object(chitchat, "resolve_api_key", return_value=key),
                  patch.object(chitchat, "make_client", return_value=self.fake)):
            p.start()
            self.addCleanup(p.stop)
        app = AppTest.from_file(APP, default_timeout=30)
        app.session_state["scenario"] = scenario
        return app.run()

    def say(self, app, text):
        app.chat_input[0].set_value(text).run()
        self.assertFalse(app.exception, [e.value for e in app.exception])
        return app

    def click(self, app, key):
        app.button(key=key).click().run()
        self.assertFalse(app.exception, [e.value for e in app.exception])
        return app

    def bubbles(self, app):
        return [m["text"].replace("\\", "") for m in app.session_state["log"]]      # esc()가 붙인 역슬래시 제거



class FlowTests(FlowBase):
    # ── 위기가 항상 먼저 ──
    def test_crisis_in_the_middle_of_chitchat_goes_to_safety_without_gemini(self):
        app = self.say(self.app("욕구 없음"), "안녕하세요")
        app = self.say(app, "고마워요")
        self.assertEqual(len(self.fake.calls), 2)
        app.session_state["scenario"] = "위기"
        app = self.say(app, "고마워요")                         # 가짜 위기 시나리오는 모든 입력을 위기로 본다
        self.assertEqual(len(self.fake.calls), 2)               # 위기 문장은 Gemini로 가지 않음
        self.assertEqual(app.session_state["page"], "safety")
        self.assertEqual(app.session_state["chat_hist"], [])    # 일상 대화 기록도 비움

    # ── 신호 있음: 도움 제안 ──
    def test_signal_gets_offer_with_two_buttons_and_no_gemini(self):
        app = self.say(self.app("욕구 없음"), "요즘 너무 힘들어요")
        self.assertEqual(self.fake.calls, [])
        self.assertIsNotNone(app.session_state["pending_offer"])
        self.assertIsNone(app.session_state["session"])
        labels = [b.label for b in app.button]
        self.assertIn("찾아볼래요", labels)
        self.assertIn("그냥 얘기할래요", labels)
        log = app.session_state["log"]
        self.assertEqual(log[-1]["text"], "그 부분, 도움이 될 만한 걸 같이 찾아볼까요?")
        self.assertEqual(log[-1]["note"], chitchat.OFFER_NOTE[3])        # 단계를 모르면 3

    def test_accepting_the_offer_starts_flow_with_the_same_sentence(self):
        app = self.say(self.app("일반"), SIGNAL)
        self.assertIsNone(app.session_state["session"])
        app = self.click(app, "offer_yes")
        session = app.session_state["session"]
        self.assertEqual(session["original_text"], SIGNAL)                # 사용자가 다시 말하지 않음
        self.assertEqual(session["status"], "awaiting_confirm")
        self.assertEqual(app.session_state["page"], "chat")
        self.assertIsNone(app.session_state["pending_offer"])
        self.assertEqual(self.fake.calls, [])
        self.assertTrue(any("어떤 쪽 도움을 살펴보면" in t or "살펴보면 좋겠다고" in t or "살고 계신" in t
                            for t in self.bubbles(app)))                   # 기존 질문이 이어짐

    def test_talk_choice_uses_fixed_reply_then_cooldown_then_asks_again(self):
        app = self.say(self.app("욕구 없음"), "요즘 너무 힘들어요")
        app = self.click(app, "offer_talk")
        self.assertIn(self.bubbles(app)[-1], chitchat.TALK_REPLIES[3])
        self.assertEqual(self.fake.calls, [])                              # 그 문장은 Gemini로 가지 않음
        self.assertEqual(app.session_state["offer_cooldown"], 3)
        previous = self.bubbles(app)[-1]
        for _ in range(3):                                                 # 3턴 동안은 같은 막연한 힘듦으로 다시 묻지 않음
            app = self.say(app, "요즘 너무 힘들어요")
            self.assertIsNone(app.session_state["pending_offer"])
            self.assertIn(self.bubbles(app)[-1], chitchat.TALK_REPLIES[3])
            self.assertNotEqual(self.bubbles(app)[-1], previous)           # 같은 문구가 연달아 나오지 않음
            previous = self.bubbles(app)[-1]
        self.assertEqual(self.fake.calls, [])
        app = self.say(app, "요즘 너무 힘들어요")                            # 4번째에 다시 묻는다
        self.assertIsNotNone(app.session_state["pending_offer"])

    def test_chitchat_after_talk_goes_to_gemini_without_the_signal_sentence(self):
        app = self.say(self.app("욕구 없음"), "요즘 너무 힘들어요")
        app = self.click(app, "offer_talk")
        app = self.say(app, "안녕하세요")
        self.assertEqual(len(self.fake.calls), 1)
        sent = texts_of(self.fake.calls[0]["contents"])
        self.assertEqual(sent, ["안녕하세요"])
        self.assertNotIn("힘들", " ".join(sent))

    # ── Gemini에 보내는 범위 ──
    def test_only_last_three_chitchat_turns_are_sent(self):
        app = self.app("욕구 없음", FakeClient("네 반가워요"))
        for t in ("안녕하세요", "고마워요", "잘 가요", "뭐해?", "하이"):
            app = self.say(app, t)
        sent = texts_of(self.fake.calls[4]["contents"])
        self.assertEqual(len(sent), 7)                                    # 3턴(6줄) + 지금 문장
        self.assertEqual(sent[0], "고마워요")                               # 가장 오래된 두 턴(안녕하세요)은 빠짐
        self.assertEqual(sent[-1], "하이")
        self.assertNotIn("안녕하세요", sent)

    def test_offer_accept_clears_chitchat_history(self):
        app = self.say(self.app("욕구 없음"), "안녕하세요")
        self.assertEqual(len(app.session_state["chat_hist"]), 1)
        app = self.say(app, SIGNAL)
        app = self.click(app, "offer_yes")
        self.assertEqual(app.session_state["chat_hist"], [])

    def test_private_text_gets_default_answer_without_gemini(self):
        app = self.say(self.app("욕구 없음"), "안녕하세요 010-1234-5678 이에요")
        self.assertEqual(self.fake.calls, [])
        self.assertIn("고마워요", self.bubbles(app)[-1])                 # 단계별 기본 답

    def test_gemini_failure_uses_stage_default_answer(self):
        app = self.say(self.app("욕구 없음", key=None), "안녕하세요")
        self.assertEqual(self.fake.calls, [])
        self.assertIn(chitchat.FALLBACK_BY_STAGE[3].split(".")[0], self.bubbles(app)[-1])

    # ── 단계 ──
    def test_stage_comes_from_extracted_activity_range(self):
        app = self.say(self.app("일반"), SIGNAL)                           # 가짜 추출은 활동 범위 '집 안' → 단계 2
        self.assertEqual(app.session_state["care_stage"], 2)
        self.assertEqual(app.session_state["log"][-1]["note"], chitchat.OFFER_NOTE[2])
        app2 = self.say(self.app("욕구 없음"), "요즘 너무 힘들어요")             # 활동 범위를 모르면 단계 없음 → 3처럼
        self.assertIsNone(app2.session_state["care_stage"])
        self.assertEqual(app2.session_state["log"][-1]["note"], chitchat.OFFER_NOTE[3])

    def test_example_button_starts_flow_directly(self):
        app = self.app("일반")
        app = self.click(app, "ex_0")
        self.assertIsNotNone(app.session_state["session"])                # 예시 버튼은 제안 질문 없이 바로
        self.assertIsNone(app.session_state["pending_offer"])

    # ── 로그에 사용자 입력 없음 ──
    def test_logs_do_not_contain_user_text(self):
        records = []

        class Collect(logging.Handler):
            def emit(self, record):
                records.append(record.getMessage())

        handler = Collect(level=logging.DEBUG)
        root = logging.getLogger()
        old = root.level
        root.addHandler(handler)
        root.setLevel(logging.DEBUG)
        try:
            app = self.say(self.app("욕구 없음"), "안녕하세요")
            app = self.say(app, "요즘 너무 힘들어요 비밀문장")
            app = self.click(app, "offer_talk")
        finally:
            root.removeHandler(handler)
            root.setLevel(old)
        joined = " ".join(records)
        self.assertNotIn("비밀문장", joined)
        self.assertNotIn("너무 힘들어요", joined)


class ReproTests(FlowBase):
    """Colab 실제 모델에서 본 문제: 쉬는 중에 새 도움 주제·분명한 요청·글자 답이 막히던 흐름."""

    def offer_then_talk(self):
        app = self.say(self.app("욕구 없음"), "요즘 너무 힘들어요")
        return self.click(app, "offer_talk")

    def test_reported_sequence(self):
        app = self.offer_then_talk()
        self.assertEqual(app.session_state["offer_cooldown"], 3)
        app = self.say(app, "월세가 걱정돼요")                               # 쉬는 중이어도 새 도움 주제면 제안이 다시 나온다
        self.assertIsNotNone(app.session_state["pending_offer"])
        self.assertEqual(self.bubbles(app)[-1], chitchat.OFFER_QUESTION)
        self.assertEqual(app.session_state["offer_cooldown"], 0)
        app = self.say(app, "찾아볼래요")                                    # 글자 답 = [찾아볼래요]
        self.assertIsNone(app.session_state["pending_offer"])
        self.assertEqual(app.session_state["session"]["original_text"], "월세가 걱정돼요")   # 첫 이야기는 제안이 나왔던 문장
        self.assertEqual(self.fake.calls, [])

    def test_explicit_request_during_cooldown_goes_straight_to_flow(self):
        for sentence in ("서비스 찾아보고싶어", "서비스 내놔", "서비스 찾아줘", "추천해 줘", "지원 제도 알려 줘", "혜택 알려줘",
                         "복지 알려 줘", "신청하고 싶어요"):
            app = self.offer_then_talk()
            app = self.say(app, sentence)
            self.assertIsNone(app.session_state["pending_offer"], sentence)         # 제안 없이
            self.assertEqual(app.session_state["session"]["original_text"], sentence, sentence)   # 그 문장이 첫 이야기
            self.assertEqual(self.fake.calls, [])

    def test_ask_to_know_is_explicit_only_with_welfare_words(self):
        for t in ("서비스 알려 줘", "지원 알려 줘", "혜택 알려줘", "복지 알려 줘", "제도 알려 줘", "신청 방법 알려 줘"):
            self.assertTrue(chitchat.is_explicit_request(t), t)
        for t in ("알려 줘", "오늘 날씨 알려 줘", "점심 메뉴 알려줘", "시간 알려 줘요"):
            self.assertFalse(chitchat.is_explicit_request(t), t)

    def test_weather_question_goes_to_chitchat(self):
        app = self.say(self.app("욕구 없음", FakeClient("맑을 것 같아요.")), "오늘 날씨 알려 줘")
        self.assertEqual(len(self.fake.calls), 1)                                         # Gemini 일상 대화
        self.assertIsNone(app.session_state["session"])                                   # 기존 흐름이 아님
        self.assertIsNone(app.session_state["pending_offer"])
        self.assertEqual(texts_of(self.fake.calls[0]["contents"]), ["오늘 날씨 알려 줘"])

    def test_explicit_request_without_cooldown_also_skips_the_offer(self):
        app = self.say(self.app("욕구 없음"), "서비스 찾아줘")
        self.assertIsNone(app.session_state["pending_offer"])
        self.assertEqual(app.session_state["session"]["original_text"], "서비스 찾아줘")

    def test_vague_hardship_is_still_quiet_during_cooldown(self):
        app = self.offer_then_talk()
        app = self.say(app, "요즘 너무 힘들어요")
        self.assertIsNone(app.session_state["pending_offer"])
        self.assertIn(self.bubbles(app)[-1], chitchat.TALK_REPLIES[3])

    def test_concrete_need_in_cooldown_with_general_scenario(self):
        app = self.say(self.app("일반"), "요즘 너무 힘들어요")                  # 일반 시나리오: 필요한 도움이 추출됨
        app = self.click(app, "offer_talk")
        app = self.say(app, "일자리 구하고 싶어요")
        self.assertIsNotNone(app.session_state["pending_offer"])

    # ── 제안 질문에 글자로 답하기 ──
    def test_text_yes_answers_equal_the_find_button(self):
        for answer in ("찾아볼래요", "네", "응", "좋아요", "찾아줘", "그래요"):
            app = self.say(self.app("욕구 없음"), SIGNAL)
            self.assertIsNotNone(app.session_state["pending_offer"], answer)
            app = self.say(app, answer)
            self.assertIsNone(app.session_state["pending_offer"], answer)
            self.assertEqual(app.session_state["session"]["original_text"], SIGNAL, answer)   # '네'가 아니라 제안이 나왔던 문장
            self.assertEqual(self.fake.calls, [])
            self.assertIn(answer, self.bubbles(app))                                          # 사용자가 쓴 말풍선은 그대로 남음

    def test_text_no_answers_equal_the_talk_button(self):
        for answer in ("그냥 얘기할래요", "아니요", "괜찮아요", "나중에", "아니"):
            app = self.say(self.app("욕구 없음"), SIGNAL)
            app = self.say(app, answer)
            self.assertIsNone(app.session_state["pending_offer"], answer)
            self.assertIsNone(app.session_state["session"], answer)                          # 기존 흐름을 시작하지 않음
            self.assertEqual(app.session_state["offer_cooldown"], 3, answer)
            self.assertIn(self.bubbles(app)[-1], chitchat.TALK_REPLIES[3], answer)
            self.assertEqual(self.fake.calls, [], answer)

    def test_other_text_while_offer_is_pending_is_a_new_input(self):
        app = self.say(self.app("욕구 없음"), SIGNAL)
        app = self.say(app, "안녕하세요")                                     # 제안과 상관없는 말 → 제안은 닫히고 일상 대화
        self.assertIsNone(app.session_state["pending_offer"])
        self.assertEqual(len(self.fake.calls), 1)

    def test_crisis_check_comes_before_a_typed_answer(self):
        app = self.say(self.app("욕구 없음"), SIGNAL)
        app.session_state["scenario"] = "위기"
        app = self.say(app, "네")                                           # 글자 답이어도 위기 확인이 먼저
        self.assertEqual(app.session_state["page"], "safety")
        self.assertIsNone(app.session_state["pending_offer"])
        self.assertEqual(self.fake.calls, [])

    # ── 안내 문구는 질문할 때만 ──
    def test_reassurance_note_only_on_the_question_bubble(self):
        app = self.offer_then_talk()
        log = app.session_state["log"]
        self.assertEqual(log[-1].get("note"), None)                           # 고정 공감 말풍선에는 안내가 없다
        self.assertEqual(sum(1 for m in log if m.get("note") == chitchat.OFFER_NOTE[3]), 1)   # 질문 말풍선에만 하나
        self.assertNotIn("답하지 않아도", self.bubbles(app)[-1])


class SkippedQuestionTests(unittest.TestCase):
    def app(self, scenario="일반"):
        with patch.dict(os.environ, {"BANGEOL_PIPELINE_MODE": "시연용 가짜"}):
            app = AppTest.from_file(APP, default_timeout=30)
            app.session_state["scenario"] = scenario
            return app.run()

    def skip_all_chat_questions(self, app):
        for _ in range(6):
            if app.session_state["page"] != "chat":
                break
            skip = [b for b in app.button if b.label == SKIP_QUESTION]
            if not skip:
                break
            skip[0].click().run()
            self.assertFalse(app.exception, [e.value for e in app.exception])
        return app

    def test_skipped_chat_questions_are_not_asked_again_after_find(self):
        app = self.app()
        app.chat_input[0].set_value(SIGNAL).run()
        app.button(key="offer_yes").click().run()
        app = self.skip_all_chat_questions(app)
        self.assertEqual(app.session_state["page"], "confirm")
        self.assertGreaterEqual(len(app.session_state["skipped"]), 2)
        app.button(key="c_submit").click().run()
        self.assertFalse(app.exception, [e.value for e in app.exception])
        self.assertEqual(app.session_state["page"], "results")            # 다시 묻지 않고 바로 결과
        shown = " ".join(m["text"] for m in app.session_state["log"])
        self.assertNotIn("어느 지역에 살고 계세요?", shown)                # 백엔드의 지역 질문 문구가 화면에 안 나옴
        self.assertEqual(app.session_state["session"]["status"], "ready")

    def backend_session(self):
        cfg = yaml.safe_load(open(AI / "configs" / "config.yaml", encoding="utf-8"))
        record = empty_record()
        record["needs"] = [{"domain": "경제", "level": "main", "status": "명시", "evidence": "생활비"}]
        s = new_session("생활비", {"crisis": False}, record)
        return Q.advance(s, cfg)

    def run_backend_skip(self, skipped):
        app = self.app()
        s = self.backend_session()
        self.assertEqual(Q.pending(s)["slot"], "region_sido")
        app.session_state["session"] = s
        app.session_state["page"] = "chat"
        app.session_state["skipped"] = skipped
        app.run()
        self.assertFalse(app.exception, [e.value for e in app.exception])
        next(b for b in app.button if b.label == SKIP_QUESTION).click().run()      # 지역 질문 건너뛰기
        self.assertFalse(app.exception, [e.value for e in app.exception])
        return app

    def test_backend_question_for_an_already_skipped_slot_is_answered_silently(self):
        app = self.run_backend_skip(["activity_range"])
        session = app.session_state["session"]
        self.assertEqual([(q["slot"], q["answer_type"]) for q in session["questions"]],
                         [("region_sido", "건너뜀"), ("activity_range", "건너뜀")])   # 활동 범위는 보여 주지 않고 건너뜀 처리
        self.assertEqual(session["status"], "ready")                                # 질문 한도(2)를 넘지 않고 바로 추천
        self.assertEqual(app.session_state["page"], "results")

    def test_backend_question_for_a_slot_not_skipped_is_still_asked(self):
        app = self.run_backend_skip([])
        session = app.session_state["session"]
        self.assertEqual(session["status"], "awaiting_answer")
        self.assertEqual(Q.pending(session)["slot"], "activity_range")                # 건너뛴 적 없으면 정상으로 묻는다
        self.assertEqual(app.session_state["page"], "chat")


if __name__ == "__main__":
    unittest.main()
