"""독립 재판정에서 나온 가능한 오류 유형의 조건부 규모·결과 영향.

AI 응답을 정답으로 적용하지 않는다. 명시한 어휘 조건의 반사실 보조안만 계산한다.
공통 마스터와 개발 보조 규칙은 수정하지 않는다.
"""
from __future__ import annotations

import pandas as pd

from ..collect._common import ROOT
from . import q1_supply as S, q1_compare as C, q1_demand as D
from .q1_followup import revise

OUT = ROOT / "outputs/03_q1/tables"
PATTERNS = {
    "mixed_living": r"양육\s*보조금",
    "recurring_allowance": r"자립\s*지원\s*수당",
    "return_cost": r"귀향\s*(?:여비|교통비)|(?:행려|무연고).{0,12}장제비",
    "admission_cash": r"입학\s*축하금",
    "care_vocabulary": r"급식\s*(?:지원|제공)|무료\s*급식|숙식\s*제공|가사\s*도우미.{0,12}파견|가사활동\s*지원|이동\s*서비스",
}


def put(frame, mask, area, present):
    for idx in frame.index[mask]:
        areas = set(str(frame.at[idx, "need_area_q21"]).split("|")) - {"N"}
        areas.add(area) if present else areas.discard(area)
        frame.at[idx, "need_area_q21"] = "|".join(sorted(areas)) or "N"


def boundary_variants(m, core, care):
    """기존 정의별 대안을 계산할 분석 프레임과 어휘 매칭 근거."""
    benefit = m.benefit_text_raw.fillna("")
    name_benefit = m.servNm.fillna("") + " " + benefit
    masks = {key: benefit.str.contains(pattern, regex=True) for key, pattern in PATTERNS.items()}
    masks["recurring_allowance"] = (name_benefit.str.contains(PATTERNS["recurring_allowance"], regex=True)
                                      & benefit.str.contains(r"월\s*\d+", regex=True))
    masks["admission_cash"] &= ~benefit.str.contains(r"수강료|수업료|교육비|학자금|교재|영수증|납부\s*확인|사용처", regex=True)
    reference = S.in_scenario(m, {"A", "B"}, {"age_overlap"})
    groups = {q: core.need_area_q21.str.split("|").map(lambda a, q=q: q in a) for q in S.AREAS}
    changed_masks = {
        "mixed_living": masks["mixed_living"] & ~core.q21_10_tier.eq("main"),
        "recurring_allowance": masks["recurring_allowance"] & ~core.q21_10_tier.eq("main"),
        "return_cost": masks["return_cost"] & ~core.q21_10_tier.isin(["main", "candidate"]),
        "admission_cash": masks["admission_cash"] & groups["q21_07"],
        "care_vocabulary": masks["care_vocabulary"] & ~care.need_area_q21.str.split("|").map(lambda a: "q21_02" in a),
    }
    candidates, counts = [], []
    for typ, mask in changed_masks.items():
        counts.append({"type": typ, "pattern": PATTERNS[typ], "matched_rows": int(masks[typ].sum()),
                       "changed_candidates": int(mask.sum()), "reference_candidates": int((mask & reference).sum()),
                       "S3_candidates": int((mask & S.in_scenario(m, {"A", "B", "U"}, None)).sum())})
        for idx in m.index[mask]:
            row = m.loc[idx]
            candidates.append({"type": typ, "servId": row.servId, "servNm": row.servNm,
                               "in_reference": bool(reference.at[idx]), "youth_scope": row.youth_scope,
                               "source_field": "benefit_text_raw" if typ != "recurring_allowance" else "servNm+benefit_text_raw",
                               "source_quote": benefit.at[idx], "assumption_status": "conditional_candidate_not_gold"})

    def adjust(base, types):
        frame = base.copy(deep=True)
        for typ in types:
            mask = changed_masks[typ]
            if typ in ("mixed_living", "recurring_allowance", "return_cost"):
                frame.loc[mask, "q21_10_tier"] = "candidate" if typ == "return_cost" else "main"
                put(frame, mask, "q21_10", True)
            elif typ == "admission_cash":
                put(frame, mask, "q21_07", False)
                frame.loc[mask, "q21_07_beneficiary"] = None
            elif typ == "care_vocabulary":
                put(frame, mask, "q21_02", True)
        return frame

    variants = {
        "reference_core": core, "reference_care": care,
        "mixed_living": adjust(core, ["mixed_living"]),
        "economic_interpretation": adjust(core, ["recurring_allowance", "return_cost"]),
        "admission_interpretation": adjust(core, ["admission_cash"]),
        "care_vocabulary": adjust(care, ["care_vocabulary"]),
        "combined_core": adjust(core, ["mixed_living", "recurring_allowance", "return_cost", "admission_cash"]),
        "combined_care": adjust(care, list(PATTERNS)),
    }
    return variants, pd.DataFrame(counts), pd.DataFrame(candidates)


def analyze():
    m = S.load_master()
    core, _ = revise(m)
    care, _ = revise(m, care_extended=True)
    variants, counts, candidates = boundary_variants(m, core, care)
    hiki, iso = D.load_demand()
    demand = D.demand_by_group(hiki, iso)
    supply, rank = [], []
    for name, frame in variants.items():
        table = C.supply_table(frame)
        table = table.loc[table.econ_rule.isin(["v4_main", "v4_all"])].copy()
        stability = C.rank_stability_by_cell(C.demand_supply_rank(demand, table))
        table.insert(0, "audit_variant", name)
        stability.insert(0, "audit_variant", name)
        supply.append(table)
        rank.append(stability)
    result = {"final_audit_type_counts": counts,
              "final_audit_impact_candidates": candidates,
              "final_audit_impact_supply": pd.concat(supply, ignore_index=True),
              "final_audit_impact_rank": pd.concat(rank, ignore_index=True)}
    return result


def main():
    for name, frame in analyze().items():
        frame.to_csv(OUT / f"{name}.csv", index=False, encoding="utf-8-sig", lineterminator="\n")
        if name == "final_audit_type_counts":
            print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
