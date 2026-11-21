"""지역 버튼은 광주·전남으로 나뉘지만, 검색은 둘 다 전남광주통합특별시 데이터를 똑같이 찾는다."""
import unittest
from pathlib import Path

import pandas as pd

from rules.filter import EXCLUDE, check_region
from rules.normalize import build_sgg_map, norm_sido

MASTER = Path(__file__).resolve().parents[2] / "data/processed/services_master.csv"
MERGED = "전남광주통합특별시"


class GwangjuJeonnamSearchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = pd.read_csv(MASTER, dtype=str)
        cls.rows = cls.df.to_dict("records")

    def verdicts(self, sido):
        return [check_region(row, {"region_sido": sido}) for row in self.rows]

    def test_master_uses_merged_name_only(self):
        values = set(self.df["region_sido"].dropna())
        self.assertIn(MERGED, values)
        self.assertNotIn("광주광역시", values)
        self.assertNotIn("전라남도", values)

    def test_both_buttons_find_the_same_merged_services(self):
        gwangju, jeonnam = self.verdicts("광주"), self.verdicts("전남")
        self.assertEqual(gwangju, jeonnam)                         # 판정과 이유까지 같음
        merged = [v for v, row in zip(gwangju, self.rows) if row["region_sido"] == MERGED]
        self.assertEqual(len(merged), 555)
        self.assertTrue(all(code != EXCLUDE for code, _ in merged))   # 광주·전남 어느 쪽을 골라도 제외되지 않음

    def test_other_regions_are_still_excluded_for_both(self):
        for sido in ("광주", "전남"):
            others = [v for v, row in zip(self.verdicts(sido), self.rows)
                      if row["source"] != "central" and row["region_sido"] in ("서울특별시", "경기도", "부산광역시")]
            self.assertTrue(others)
            self.assertTrue(all(code == EXCLUDE for code, _ in others), sido)

    def test_district_lists_match(self):
        sgg = build_sgg_map(self.df, MASTER)
        self.assertEqual(sgg["광주"], sgg["전남"])
        self.assertTrue({"북구", "목포시"} <= sgg["광주"])

    def test_merged_name_resolves_to_both(self):
        self.assertEqual(norm_sido(MERGED), {"광주", "전남"})


if __name__ == "__main__":
    unittest.main()
