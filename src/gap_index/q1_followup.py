"""원문 재검토 후 Q1 보조 규칙을 전체에 적용한다. 공통 마스터는 쓰지 않는다.

실행: python -m src.gap_index.q1_followup
원문 판정은 private/q1_followup에 별도 보존하며, 공식 라벨을 대체하지 않는다.
"""
from __future__ import annotations

import json
import re

import pandas as pd

from ..collect._common import ROOT
from ..labeling.q1_followup_review import comparison
from ..labeling.q1_label_audit import source_only
from . import q1_compare as C, q1_demand as D, q1_supply as S

VERSION = "q1-followup-rules-20261105-v1"
OUT = ROOT / "outputs/03_q1/tables"
# 아래 기준은 자동 후보 규칙이다. 이름·자격의 단순 언급으로 영역을 추가하지 않는다.
OPEN_SPACE = r"누구나.{0,25}(?:방문|이용)|자유롭게.{0,20}이용|쉼터\s*자율이용"
CORE_RECOVERY = (r"자기\s*돌봄비|(?:지역사회\s*복귀|사회\s*적응|일상\s*생활)\s*훈련|"
                 r"(?:조기\s*)?사회\s*복귀\s*지원\s*프로그램|생활\s*(?:리듬|습관).{0,15}(?:훈련|교육|회복)")
CARE = r"간병|가사\s*지원|가사\s*대행|아이\s*돌봄|활동\s*보조|밑반찬|안부\s*확인|병원\s*동행|ICT\s*기기|일상생활지원서비스"
LEARNING = r"금융\s*교육|교육비|수강료|수업료|강의|학습비|직업\s*능력\s*개발\s*훈련을\s*실시"


def revise(m: pd.DataFrame, *, care_extended: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """전 행에 같은 규칙을 적용한다. 서비스 ID별 수동 덮어쓰기를 사용하지 않는다.

    V3·v4a 스냅샷, 청년 범위, 입력 프레임은 보존한다. 인용은 매칭된 원문에서만 가져온다.
    """
    result = m.copy(deep=True)
    events = []
    for idx, row in m.iterrows():
        source = source_only(row)
        name = source.get("servNm", "")
        benefit = source.get("benefit_text_raw", "")
        summary = source.get("summary_text_raw", "")
        profile = name + " " + summary
        areas = set(str(row.need_area_q21).split("|")) - {"N"}
        originals = {"need_area_q21": row.need_area_q21, "q21_10_tier": row.q21_10_tier,
                     "q21_07_beneficiary": row.q21_07_beneficiary}
        tier, education = row.q21_10_tier, row.q21_07_beneficiary
        reasons = []

        def hit(pattern: str, fields=("benefit_text_raw",)):
            for field in fields:
                match = re.search(pattern, source.get(field, ""))
                if match:
                    return {"field": field, "quote": match.group(0)}
            return None

        def change(area: str, add: bool, rule: str, evidence: dict | None):
            if evidence is not None and ((area not in areas) if add else (area in areas)):
                areas.add(area) if add else areas.discard(area)
                reasons.append({"rule": rule, "action": ("add:" if add else "remove:") + area, **evidence})

        opened = hit(OPEN_SPACE, ("benefit_text_raw", "summary_text_raw"))
        space = re.search(r"공간|쉼터|마음편의점", profile + benefit)
        if opened and space:
            change("q21_01", True, "explicit_open_space", opened)
        if not opened:
            closed = hit(r"임시\s*보장시설|유휴\s*숙박시설|어린이날.{0,35}행사", ("benefit_text_raw", "servNm"))
            change("q21_01", False, "housing_or_event_is_not_drop_in", closed)

        recovery = hit(CORE_RECOVERY)
        if not recovery and re.search(r"고립|은둔", profile) and re.search(r"일상\s*복귀|일상\s*회복", summary):
            recovery = hit(r"맞춤형\s*프로그램")
        if recovery:
            change("q21_02", True, "direct_recovery_support", recovery)
        elif "q21_02" in areas:
            # 실제 지원에 핵심 회복 내용이 없는 기존 양성만 재검토한다.
            evidence = hit(CARE + r"|심리치료|심리치료비용|보험료|생활교육")
            change("q21_02", False, "care_or_purpose_only", evidence)
        if care_extended:
            change("q21_02", True, "care_extension", hit(CARE))

        learning = hit(LEARNING)
        if learning:
            change("q21_07", True, "explicit_learning_support", learning)
        elif "q21_07" in areas:
            exam = hit(r"시험\s*응시료|시험응시료|응시료\s*지원", ("servNm", "summary_text_raw", "benefit_text_raw"))
            emergency = hit(r"임시\s*보장시설")
            if not re.search(r"수강|수업|등록금|학자금|학비|학습|교육비|교육\s*(?:과정|프로그램)|장학", profile + benefit):
                change("q21_07", False, "exam_fee_or_emergency_only", exam or emergency)
        if "q21_07" in areas and pd.isna(education) and learning:
            # 수혜자를 새로 추정하지 않고 미상으로 기록한다.
            education = "N"
        elif "q21_07" not in areas:
            education = None

        housing = hit(r"주거\s*정착금|유휴\s*숙박시설|임시\s*보장시설|노숙인\s*시설\s*입소|입소\s*노숙인")
        if housing:
            change("q21_11", True, "explicit_housing_support", housing)
        room = hit(r"원룸사업", ("servNm",))
        if room and re.search(r"월\s*최대.{0,15}지원", benefit) and re.search(r"원룸|무주택", source.get("target_text_raw", "")):
            housing = room
            change("q21_11", True, "room_payment", room)
        # 교육을 함께 제공하는 적금·양육비를 지정 용도 수강료와 혼동하지 않는다.
        tuition = hit(r"수강료") if re.search(r"취업\s*능력\s*개발", name) else None
        job_cost = hit(r"구직활동을\s*위한\s*학습비.{0,100}내역\s*증빙")
        earmark = housing or tuition or job_cost
        separate_living = hit(r"생계|생활비|생활\s*보조금|생활\s*안정|아동\s*양육비|저축|적립|적금|매칭|건강보험료|국민연금|산재보험료")
        if earmark and not separate_living:
            if "q21_10" in areas or tier in ("main", "candidate"):
                tier = "excluded"
                change("q21_10", False, "earmarked_payment", earmark)
        # 건강보험료와 반복 교통비의 실제 지원: 제목이 의료비라는 이유로 함께 제외하지 않는다.
        insurance = hit(r"지역가입자에\s*대해서는\s*보험료\s*전액을\s*국고\s*지원")
        transport = hit(r"대중교통\s*이용금액.{0,100}(?:환급|지급)|대중교통비.{0,30}환급", ("benefit_text_raw", "servNm"))
        living = hit(r"소득[･·,\s]*생계[･·,\s]*금융지원")
        if insurance or transport or living:
            tier = "main"
            change("q21_10", True, "explicit_essential_cost", insurance or transport or living)
        if re.search(r"보충수당", name):
            allowance = hit(r"근로장애인\s*월\s*\d+만원.{0,30}훈련장애인\s*월\s*\d+만원")
            if allowance:
                tier = "candidate"
                change("q21_10", True, "explicit_allowance_candidate", allowance)
        # 책 구입·도서관 이름만으로 혼자 하는 취미 활동을 추정한 경우를 분리한다.
        if re.search(r"바우처", name) and learning and not re.search(r"취미|문화|체육|관람|여가", benefit):
            change("q21_03", False, "learning_cost_is_not_hobby", learning)

        updated = {"need_area_q21": "|".join(sorted(areas)) or "N", "q21_10_tier": tier,
                   "q21_07_beneficiary": education}
        for field, value in updated.items():
            old = originals[field]
            same = (pd.isna(old) and pd.isna(value)) or (not pd.isna(old) and not pd.isna(value) and old == value)
            if not same:
                result.at[idx, field] = value
                events.append({"servId": row.servId, "servNm": row.servNm, "field": field,
                               "before": old, "after": value, "rule_version": VERSION,
                               "evidence": json.dumps(reasons, ensure_ascii=False)})
    return result, pd.DataFrame(events, columns=["servId", "servNm", "field", "before", "after", "rule_version", "evidence"])


def sensitivity(m: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """개인 검토 파일 없이 같은 마스터·조사 입력에서 보조 결과를 재현한다."""
    frames = {}
    supply, ranks, changes = [], [], []
    hiki, iso = D.load_demand()
    demand = D.demand_by_group(hiki, iso)
    for name in ("official", "revised_core", "revised_care"):
        revised, events = (m, pd.DataFrame()) if name == "official" else revise(m, care_extended=name == "revised_care")
        table = C.supply_table(revised)
        # v3는 보존 스냅샷이며 개정 결과로 해석하지 않도록 비교에서 제외한다.
        table = table.loc[table.econ_rule.isin(["v4_main", "v4_all", "v4_main_edu_self"])].copy()
        table.insert(0, "variant", name)
        supply.append(table)
        rank = C.rank_stability_by_cell(C.demand_supply_rank(demand, table))
        rank.insert(0, "variant", name)
        ranks.append(rank)
        if not events.empty:
            events.insert(0, "variant", name)
            changes.append(events)
    frames["followup_supply"] = pd.concat(supply, ignore_index=True)
    frames["followup_rank_stability"] = pd.concat(ranks, ignore_index=True)
    frames["followup_rule_changes"] = pd.concat(changes, ignore_index=True)
    return frames


def run() -> dict[str, pd.DataFrame]:
    m = S.load_master()
    return {"followup_source_review": comparison(m), **sensitivity(m)}


def main() -> None:
    frames = run()
    OUT.mkdir(parents=True, exist_ok=True)
    for name, frame in frames.items():
        frame.to_csv(OUT / f"{name}.csv", index=False, encoding="utf-8-sig", lineterminator="\n")
    review = frames["followup_source_review"]
    print(review.assign(changed=review.changed_items.ne(""), unresolved=review.unresolved_items.ne(""))
          .groupby("review_groups")[["changed", "unresolved"]].agg(["size", "sum"]).to_string())
    s = frames["followup_supply"]
    print(s.loc[s.scenario.eq("S2_AB_strict") & s.econ_rule.eq("v4_main")]
          .pivot(index="area", columns="variant", values="n_services").to_string())


if __name__ == "__main__":
    main()
