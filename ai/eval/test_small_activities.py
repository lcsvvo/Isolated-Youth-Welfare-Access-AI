import unittest
from small_activities import RANGES, pair_activities, select_activities, should_offer

class SmallActivitiesTests(unittest.TestCase):
    def test_unsure_and_low_motivation_offer(self):
        for text in ("모르겠어요", "뭘 해야 할지 모르겠어요", "의욕이 없어요", "아무것도 하기 싫어요"):
            self.assertTrue(should_offer(text), text)
    def test_demographics_do_not_trigger_offer(self):
        self.assertFalse(should_offer("모르겠어요", "region_sido"))
        self.assertFalse(should_offer("모르겠어요", "age"))
        self.assertFalse(should_offer("생활비 지원이 필요해요"))
    def test_default_has_no_outing_or_contact(self):
        cards = select_activities()
        self.assertEqual(len(cards), 3)
        self.assertTrue(all(a["range"] == "방 안" and a["effort"] == 1 and a["contact_max"] == 0 for a in cards))
    def test_choices_respect_limits(self):
        for scope in RANGES:
            for energy in (1, 2, 3):
                for contact in (0, 1):
                    cards = select_activities(scope, energy, contact)
                    self.assertLessEqual(len(cards), 3)
                    self.assertTrue(all(RANGES[a["range"]] <= RANGES[scope] and a["effort"] <= energy and a["contact_max"] <= contact for a in cards))
        self.assertTrue(all(RANGES[a["range"]] <= 1 for a in select_activities("필요하면 외출", 3, 1, True)))
    def test_invalid_limits_rejected(self):
        with self.assertRaises(ValueError):
            select_activities("판단 불가")
    def test_pair_only_when_few_services_or_daily_life(self):
        job = [{"domain": "미취업(구직)", "level": "main"}]
        self.assertEqual(pair_activities(job, "집 안", 5), [])
        self.assertTrue(pair_activities(job, "집 안", 0))
        self.assertTrue(pair_activities([{"domain": "일상·여가", "level": "sub"}], "집 안", 5))
    def test_pair_prefers_need_domain_and_stays_light(self):
        cards = pair_activities([{"domain": "미취업(구직)", "level": "main"}], "집 안", 1)
        self.assertEqual(cards[0]["matched"], "미취업(구직)")
        self.assertTrue(all(a["effort"] <= 2 and a["contact_max"] == 0 for a in cards))
    def test_pair_unknown_range_and_visit_block_are_conservative(self):
        self.assertTrue(all(a["range"] == "방 안" for a in pair_activities([], None, 0)))
        self.assertTrue(all(RANGES[a["range"]] <= 1 for a in pair_activities([], "필요하면 외출", 0, visit_blocked=True)))
    def test_pair_respects_earlier_decline(self):
        self.assertEqual(pair_activities([], "집 안", 1, declined=True), [])
        self.assertTrue(pair_activities([], "집 안", 0, declined=True))

if __name__ == "__main__":
    unittest.main()
