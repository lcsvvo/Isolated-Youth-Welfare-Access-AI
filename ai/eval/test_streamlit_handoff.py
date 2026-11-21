"""UI lifecycle regression checks. No real model download or GPU needed."""
import copy
import os
import unittest
from pathlib import Path
from unittest.mock import patch
from streamlit.testing.v1 import AppTest
from confirm.session import new_session
from confirm.questions import auto_finish
from extract.schema import empty_record

APP = Path(__file__).resolve().parents[1] / "app.py"


def session():
    record = empty_record()
    record["needs"] = [{"domain": "경제", "level": "main", "status": "명시", "evidence": "생활비"}]
    return new_session("생활비", {"crisis": False}, record)


class StreamlitHandoffTests(unittest.TestCase):
    def app(self):
        with patch.dict(os.environ, {"BANGEOL_PIPELINE_MODE": "시연용 가짜"}):
            app = AppTest.from_file(str(APP), default_timeout=30).run()
        self.assertFalse(app.exception)
        return app

    def test_empty_visit_links_to_browse(self):
        app = self.app()
        app.button(key="nav_visit").click().run()
        self.assertTrue(any("아직 담은 서비스가 없어요" in m.value for m in app.markdown))
        app.button(key="visit_browse").click().run()
        self.assertEqual(app.session_state["page"], "explore")
        self.assertFalse(app.exception)

    def test_compact_navigation_keeps_help_and_privacy_accessible(self):
        app = self.app()
        for key, page in (("compact_explore", "explore"), ("compact_visit", "visit"),
                          ("compact_activities", "activities"),
                          ("compact_about", "about"), ("compact_safety", "safety"), ("compact_new", "home")):
            app.button(key=key).click().run()
            self.assertEqual(app.session_state["page"], page)
            self.assertFalse(app.exception)

    def test_activity_consent_decline_and_rest(self):
        for action, expected in (("activity_yes", "activities"), ("activity_no", "chat"), ("activity_rest", "rest")):
            app = self.app()
            app.chat_input[0].set_value("뭘 해야 할지 모르겠어요").run()
            app.button(key="offer_yes").click().run()      # 도움 제안에서 [찾아볼래요]
            self.assertEqual(app.session_state["page"], "activity_offer")
            self.assertIsNone(app.session_state["activity_cards"])
            self.assertIsNone(app.session_state["out"])
            app.button(key=action).click().run()
            self.assertEqual(app.session_state["page"], expected)
            self.assertFalse(app.exception)

    def test_activities_require_explicit_show_and_new_story_resets_them(self):
        app = self.app()
        app.chat_input[0].set_value("의욕이 없어요").run()
        app.button(key="offer_yes").click().run()          # 도움 제안에서 [찾아볼래요]
        app.button(key="activity_yes").click().run()
        self.assertIsNone(app.session_state["activity_cards"])
        app.button(key="activity_show").click().run()
        self.assertEqual(len(app.session_state["activity_cards"]), 3)
        app.button(key="nav_new").click().run()
        self.assertIsNone(app.session_state["activity_cards"])
        self.assertEqual(app.session_state["page"], "home")

    def test_urgent_housing_request_bypasses_activities(self):
        app = self.app()
        app.chat_input[0].set_value("기운이 없는데 당장 월세가 급해요. 지원 서비스 찾아 주세요.").run()
        self.assertEqual(app.session_state["page"], "chat")
        self.assertIsNotNone(app.session_state["session"])
        self.assertIsNone(app.session_state["activity_cards"])
        self.assertFalse(app.exception)

    def test_search_outcomes_and_retry(self):
        cases = (
            ("결과 없음", "empty", "관련 서비스를 찾지 못했어요"),
            ("조건으로 전부 제외", "all_excluded", "지금 조건으로는 결과가 없어요"),
            ("일부 오류", "results", None),
            ("오류", "error", "잠시 연결이 원활하지 않아요"),
        )
        app = self.app()
        for scenario, outcome, heading in cases:
            with self.subTest(scenario=scenario):
                app.session_state["scenario"] = scenario
                app.session_state["session"] = auto_finish(session())
                app.session_state["page"] = "chat"
                app.session_state["outcome"] = None
                app.run()
                next(b for b in app.button if b.label == "다시 찾아보기").click().run()
                self.assertEqual(app.session_state["outcome"], outcome)
                self.assertFalse(app.exception)
                if heading:
                    self.assertTrue(any(heading in m.value for m in app.markdown))
                else:
                    self.assertTrue(any("일부 검색 경로" in w.value for w in app.warning))
                if outcome == "error":
                    app.session_state["scenario"] = "일반"
                    next(b for b in app.button if b.label == "다시 찾아보기").click().run()
                    self.assertEqual(app.session_state["outcome"], "results")
                else:
                    app.button(key="r_confirm").click().run()
                    self.assertEqual(app.session_state["page"], "confirm")

    def test_results_pair_small_activities_only_when_needed(self):
        app = self.app()
        for scenario, shown in (("결과 없음", True), ("일반", False)):
            with self.subTest(scenario=scenario):
                app.session_state["scenario"] = scenario
                app.session_state["session"] = auto_finish(session())
                app.session_state["page"] = "chat"
                app.session_state["outcome"] = None
                app.run()
                next(b for b in app.button if b.label == "다시 찾아보기").click().run()
                self.assertFalse(app.exception)
                found = any("👣 서비스와 함께 해볼 수 있는 작은 활동" in m.value for m in app.markdown)
                self.assertEqual(found, shown or len(app.session_state["out"]["cards"]) < 3)
                if found:                                         # 카드에는 활동 범위만 보이고 상호작용 표시는 없다
                    cards = [m.value for m in app.markdown if "small-activity-card" in m.value]
                    self.assertTrue(cards)
                    for value in cards:
                        self.assertIn("activity-meta", value)
                        self.assertNotIn("상호작용 없음", value)
                        self.assertNotIn("짧은 상호작용", value)

    def test_small_activity_labels_have_footprint_emoji(self):
        app = self.app()
        self.assertEqual(app.button(key="nav_activities").label, "👣 작은 활동")
        self.assertEqual(app.button(key="compact_activities").label, "👣 작은 활동")
        app.button(key="nav_activities").click().run()
        self.assertFalse(app.exception)
        self.assertTrue(any("👣 작은 활동 살펴보기" in m.value for m in app.markdown))     # 위쪽 이동 경로 제목

    def test_insufficient_information_opens_browsing_and_can_return(self):
        app = self.app()
        app.session_state["scenario"] = "욕구 없음"
        app.button(key="ex_0").click().run()
        self.assertEqual(app.session_state["page"], "unsure")
        self.assertIsNone(app.session_state["out"])
        self.assertTrue(any("탐색용 · 추천 순위 아님" in c.value for c in app.caption))
        app.button(key="un_chat").click().run()
        self.assertEqual(app.session_state["page"], "chat")
        self.assertFalse(app.exception)

    def test_current_answers_survive_stale_summary(self):
        app = self.app()
        before = session()
        for status in ("ready", "awaiting_answer", "explore_only"):
            current = copy.deepcopy(before)
            current["status"] = status
            current["actions"].append({"type": "set_age", "target": None, "value": 27, "accepted": True})
            current["confirmed_profile"]["age"] = 27
            current["questions"] = [{"slot": "age", "order": 1, "answer": None, "answer_type": None}]
            app.session_state["session"] = current
            app.session_state["snapshot"] = copy.deepcopy(before)
            app.session_state["page"] = "confirm"
            app.run()
            self.assertFalse(app.exception)
            self.assertEqual(app.session_state["session"]["status"], "awaiting_confirm")
            self.assertEqual(app.session_state["session"]["confirmed_profile"]["age"], 27)
            self.assertFalse(app.session_state["session"]["questions"])

    def test_combined_region_field_and_collapsed_conversation_summary(self):
        app = self.app()
        app.session_state["session"] = session()
        app.session_state["page"] = "confirm"
        app.run()
        toggle = app.toggle[0]
        region = next(widget for widget in app.selectbox if widget.label == "사는 지역")
        self.assertFalse(toggle.value)
        self.assertFalse(any("확인했어요" in badge.value for badge in app.get("badge")))
        self.assertEqual(region.options, sorted(region.options))
        district = next(option for option in region.options if option.startswith("서울 "))
        selected = tuple(district.split(" ", 1))
        region.select(selected).run()
        app.toggle[0].set_value(True).run()
        self.assertTrue(any(caption.value == "지역" for caption in app.caption))
        app.button(key="c_submit").click().run()
        self.assertFalse(app.exception)
        confirmed = app.session_state["session"]["confirmed_profile"]
        self.assertEqual((confirmed["region_sido"], confirmed["region_sgg"]), selected)

    def region_widget(self, app):
        return next(widget for widget in app.selectbox if widget.label == "사는 지역")

    def test_gwangju_and_jeonnam_are_separate_options_between_gyeongbuk_and_daegu(self):
        app = self.app()
        current = session()
        current["actions"] += [{"type": "set_region", "value": "광주", "accepted": True},
                               {"type": "set_region_sgg", "value": "북구", "accepted": True}]
        current["confirmed_profile"].update(region_sido="광주", region_sgg="북구")
        app.session_state["session"] = current
        app.session_state["page"] = "confirm"
        app.run()
        region = self.region_widget(app)
        areas = [option.split(" ")[0] for option in region.options]
        self.assertIn("광주", areas)
        self.assertIn("전남", areas)
        self.assertNotIn("전남광주통합특별시", " ".join(region.options))
        order = list(dict.fromkeys(areas))
        self.assertEqual(order[order.index("경북"):order.index("경북") + 3], ["경북", "광주", "대구"])
        self.assertEqual(region.value, ("광주", "북구"))                 # 광주 북구로 들어온 값이 그대로 선택됨
        for area in ("광주", "전남"):                                    # 시군구 목록은 두 곳 모두 합쳐서 보여 줌
            region.select((area, None)).run()
            app.button(key="c_submit").click().run()
            self.assertFalse(app.exception)
            confirmed = app.session_state["session"]["confirmed_profile"]
            self.assertEqual((confirmed["region_sido"], confirmed["region_sgg"]), (area, None))
            app.session_state["page"] = "confirm"
            app.run()
            region = self.region_widget(app)

    def test_typed_region_is_recognized_for_gwangju_and_jeonnam(self):
        for text, expected in (("광주 살아요", "광주"), ("전남에 살아요", "전남"), ("광주광역시", "광주")):
            app = self.app()
            current = session()
            app.session_state["session"] = current
            app.session_state["slot"] = "region_sido"
            app.session_state["page"] = "chat"
            app.run()
            app.chat_input[0].set_value(text).run()
            self.assertFalse(app.exception, text)
            profile = app.session_state["session"]["confirmed_profile"]
            self.assertEqual(profile["region_sido"], expected, text)

    def test_missing_session_recovers_without_rerun_loop(self):
        app = self.app()
        app.session_state["session"] = None
        app.session_state["page"] = "confirm"
        app.run()
        self.assertEqual(app.session_state["page"], "home")
        self.assertFalse(app.exception)

    def test_crisis_blocks_service_and_activity_pages(self):
        app = self.app()
        app.session_state["session"] = {"status": "crisis_stop"}
        for page in ("confirm", "results", "detail", "explore", "unsure", "visit", "activity_offer", "activities", "rest"):
            app.session_state["page"] = page
            app.run()
            self.assertFalse(app.exception)
            self.assertTrue(any("지금 바로 연락할 수 있는 도움" in m.value for m in app.markdown))
            self.assertFalse(any(b.key in ("activity_show", "activity_yes", "c_submit") for b in app.button))

    def test_source_first_details_generate_only_on_request(self):
        app = self.app()
        app.session_state["session"] = auto_finish(session())
        app.session_state["page"] = "chat"
        app.run()
        next(b for b in app.button if b.label == "다시 찾아보기").click().run()
        self.assertEqual(app.session_state["page"], "results")
        self.assertTrue(all(c["explanation_pending"] for c in app.session_state["out"]["cards"]))
        next(b for b in app.button if b.key and b.key.startswith("open_0_")).click().run()
        self.assertEqual(app.session_state["page"], "results")
        self.assertFalse(app.exception)
        # AppTest runs the whole script on click; dialog fragment reruns are
        # checked in the browser. Exercise the shared detail body here.
        app.session_state["page"] = "detail"
        app.run()
        next(b for b in app.button if b.key and b.key.startswith("explain_")).click().run()
        self.assertFalse(app.exception)
        self.assertFalse(app.session_state["out"]["cards"][0]["explanation_pending"])
        self.assertFalse(any(b.key and b.key.startswith("explain_") for b in app.button))
        next(b for b in app.button if b.key and b.key.startswith("save_")).click().run()
        self.assertIn(app.session_state["selected"], app.session_state["visit"])


if __name__ == "__main__":
    unittest.main()
