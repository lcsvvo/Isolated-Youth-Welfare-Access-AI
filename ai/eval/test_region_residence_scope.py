import unittest
from rules.region import check_region, residence_required
from ui_text import friendly_summary

class ResidenceScopeTests(unittest.TestCase):
    def service(self,criteria="전국 무주택자"):
        return {"servId":"WLF00004762","source":"local","region_sido":"전북특별자치도","region_sgg":"부안군",
                "target_text":"19세 이상 45세 이하인 부안군 거주 무주택 청년 가구",
                "criteria_text":criteria,"summary_text":"청년 주거비 부담 경감 및 안정적 지역정착 지원"}
    def test_nationwide_property_check_does_not_cancel_local_residency(self):
        for text in ("전국 무주택자", "전국 주택 소유 여부 확인", "전국 중복 수혜 조회"):
            svc=self.service(text)
            self.assertIs(residence_required(svc)[0],True)
            self.assertEqual(check_region(svc,{"region_sido":"서울"})[0],"제외")
            self.assertEqual(check_region(svc,{"region_sido":"전북","region_sgg":"부안군"})[0],"통과")
    def test_genuinely_open_nationwide_program_is_not_excluded(self):
        svc=self.service("거주지 제한 없이 전국 청년 누구나 신청 가능")
        self.assertIs(residence_required(svc)[0],False)
        self.assertNotEqual(check_region(svc,{"region_sido":"서울"})[0],"제외")
    def test_summary_keeps_numbers_and_eligibility_qualifiers(self):
        raw="부안군 거주 19~45세 청년에게 월 최대 20만원을 지원합니다."
        shown=friendly_summary(raw)
        self.assertEqual(shown,"부안군 거주 19~45세 청년에게 월 최대 20만원을 지원해요.")
        self.assertEqual(raw,"부안군 거주 19~45세 청년에게 월 최대 20만원을 지원합니다.")

if __name__=="__main__": unittest.main()
