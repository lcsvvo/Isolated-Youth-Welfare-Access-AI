"""사람 정답을 Q1 분석용 사본에 반영하고 기존 시나리오를 재계산."""
from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd

from ..collect._common import ROOT
from ..labeling import q1_human as H
from ..labeling.q1_followup_review import previous_ids
from ..preprocess.ssis_master import extract_age, overlaps_youth
from . import q1_audit_impact as B, q1_compare as C, q1_followup as F, q1_supply as S

OUT = ROOT / "outputs/03_q1"
REFERENCE = "reference_core"
RULES = ["v4_main", "v4_all", "v4_main_edu_self"]


def development_flags(gold: pd.DataFrame) -> pd.DataFrame:
    """문서·채점키에 기록된 중복만 표시. 나머지는 미노출 확인 표본이 아님."""
    flags = H.read_csv(ROOT / "outputs/02_labeling/tables/labeling_key_rule_dev_overlap.csv")
    sources = {"step02_overlap": set(flags.loc[flags.in_rule_dev.eq("True"), "servId"]),
               "q1_audit_cases": previous_ids()}
    for label, path in {
        "prior_economy_review": "econ_v4_review/used_ids.csv",
        "prior_area_review": "label_qc_2026-11-05/area_key.csv",
        "prior_youth_review": "label_qc_2026-11-05/youth_key.csv",
    }.items():
        sources[label] = set(H.read_csv(ROOT / "logs/validation" / path).servId)
    validation_doc = (ROOT / "docs/q1_final_validation.md").read_text(encoding="utf-8")
    sources["prior_model_audit_cases"] = set(re.findall(r"WLF\d+", validation_doc))
    result = gold.copy()
    result["development_sources"] = result.servId.map(
        lambda sid: "|".join(source for source, ids in sources.items() if sid in ids))
    result["known_development_overlap"] = result.development_sources.ne("")
    return result


def overlay(m: pd.DataFrame, gold: pd.DataFrame, trace: pd.DataFrame) -> pd.DataFrame:
    """예·아니오·불명을 보존. 기존 공급 함수에는 명시적 후보 집합을 넘김."""
    h = m[["servId"]].merge(gold[["servId", *H.FIELDS]], on="servId", how="left", validate="one_to_one")
    h.index = m.index
    reviewed = h[H.YOUTH].notna()
    out = m.assign(q1_human_reviewed=reviewed)
    for field in H.FIELDS:
        out[field] = h[field]
    mat = S.area_matrix(m)
    for area in S.AREAS:
        mat.loc[reviewed, area] = h.loc[reviewed, f"h_{area}"].eq(H.YES)
    out.loc[reviewed, "need_area_q21"] = mat.loc[reviewed].apply(
        lambda row: "|".join(row.index[row]) or "N", axis=1)

    econ = h["h_q21_10"]
    out.loc[econ.eq(H.YES), "q21_10_tier"] = "main"
    out.loc[econ.eq(H.NO) & m.q21_10_tier.eq("main"), "q21_10_tier"] = "excluded"
    out.loc[econ.eq("불명"), "q21_10_tier"] = "unknown"
    edu = h["h_q21_07"]
    out.loc[reviewed & ~edu.eq(H.YES), "q21_07_beneficiary"] = None
    out.loc[edu.eq(H.YES) & m.q21_07_beneficiary.isna(), "q21_07_beneficiary"] = "N"

    youth = h[H.YOUTH]
    out.loc[reviewed, "elig_youth"] = youth.loc[reviewed].replace("확인필요", "N")
    out.loc[reviewed, "apply_barrier"] = h.loc[reviewed, H.APPLY]
    out.loc[reviewed, "use_outing"] = h.loc[reviewed, H.OUTING]
    out.loc[youth.eq(H.NO), ["youth_scope", "youth_scope_basis"]] = ["C", "human_not_eligible"]
    unknown = youth.eq("확인필요")
    out.loc[unknown, "youth_scope"] = "U"
    out.loc[unknown & ~m.youth_scope_basis.isin(S.EXCLUDE_FROM_S3), "youth_scope_basis"] = "human_uncertain"
    add = youth.eq(H.YES) & ~m.youth_scope.isin(["A", "B"])
    out.loc[add, ["youth_scope", "youth_scope_basis"]] = ["B", "human_confirmed"]
    # 사람 '예'의 명시적 연령 근거만 엄격 B에 반영. 일반 청년 언급은 넓은 B에 유지.
    proof = trace.loc[trace.label.eq(H.YOUTH) & trace.final_value.eq(H.YES)].set_index("servId")
    for idx in out.index[youth.eq(H.YES) & out.youth_scope.eq("B") & ~out.youth_scope_basis.eq("age_overlap")]:
        row = proof.loc[out.at[idx, "servId"]]
        if row.source == "resolution":
            texts = [row.source_quote, row.rationale]
        else:
            texts = [row.primary_evidence, row.second_evidence]
        if overlaps_youth(extract_age(texts)[-1]):
            out.at[idx, "youth_scope_basis"] = "age_overlap"
    return out


def error_cases(gold, trace, official, revised):
    before = H.predictions(official).stack().rename("automatic_before").rename_axis(["servId", "label"])
    after = H.predictions(revised).stack().rename("automatic_revised").rename_axis(["servId", "label"])
    cases = trace.merge(before, on=["servId", "label"]).merge(after, on=["servId", "label"])
    cases = cases.loc[cases.final_value.ne(cases.automatic_before) | cases.final_value.ne(cases.automatic_revised)].copy()
    cases["difference"] = np.select(
        [cases.final_unknown, cases.label.isin(H.AREA_LABELS) & cases.final_value.eq(H.YES),
         cases.label.isin(H.AREA_LABELS) & cases.final_value.eq(H.NO)],
        ["human_unknown", "missed_area", "false_positive_area"], default="attribute_disagreement")
    raw = official.set_index("servId")
    for field, labels in {
        "target": [H.YOUTH], "criteria": [H.YOUTH], "apply": [H.APPLY],
        "benefit": [*H.AREA_LABELS, H.OUTING],
    }.items():
        cases[f"{field}_source_text"] = cases.servId.map(raw[f"{field}_text_raw"]).where(cases.label.isin(labels))
    cases["rule_evidence"] = cases.servId.map(raw["need_area_q21_evidence"]).where(cases.label.isin(H.AREA_LABELS))
    return cases.merge(gold[["servId", "development_sources"]], on="servId", validate="many_to_one")


def candidate_details(frame, variant):
    rows = []
    for scenario, scopes, bases, _ in S.SCENARIOS:
        selected = S.in_scenario(frame, scopes, bases)
        for rule in RULES[:2]:
            mat = S.area_matrix(frame, rule)
            for area in S.AREAS:
                hit = selected & mat[area]
                g = frame.loc[hit]
                for field, levels in {"youth_scope": ["A", "B", "U", "C"],
                                      "apply_barrier": H.LEVELS[H.APPLY], "use_outing": H.LEVELS[H.OUTING]}.items():
                    values = g[field].fillna("N")
                    rows.extend({"variant": variant, "scenario": scenario, "econ_rule": rule, "area": area,
                                 "attribute": field, "value": v, "n_candidates": len(g),
                                 "n_value": int(values.eq(v).sum()),
                                 "n_value_reviewed": int((values.eq(v) & g.q1_human_reviewed).sum())}
                                for v in levels)
    return pd.DataFrame(rows)


def layers(m: pd.DataFrame) -> dict:
    """사람 판정 반영 프레임과 경계 대안 프레임. 집계 없이 서비스 단위 결과만 반환."""
    gold, trace, log = H.build_gold()
    H.require(m.servId.is_unique and set(gold.servId) <= set(m.servId), "마스터와 정답셋 servId 불일치")
    gold = development_flags(gold)
    auto_core, _ = F.revise(m)
    auto_care, _ = F.revise(m, care_extended=True)
    core = overlay(auto_core, gold, trace)
    care = core.copy()
    # ② 직접 회복 정답 위에 기존 일반 돌봄 어휘를 별도 대안으로만 추가.
    B.put(care, m.benefit_text_raw.fillna("").str.contains(F.CARE, regex=True), "q21_02", True)
    variants, _, candidates = B.boundary_variants(core, core, care)
    # 대안의 자동 어휘가 확정 main·⑦ 판정을 바꾸지 않도록 해당 칸 복원.
    for frame in variants.values():
        known_econ = core["h_q21_10"].isin([H.YES, H.NO, "불명"])
        original_candidate = core.q21_10_tier.eq("candidate")
        keep = known_econ & (frame.q21_10_tier.eq("main") | core["h_q21_10"].ne(H.NO) | original_candidate)
        frame.loc[keep, "q21_10_tier"] = core.loc[keep, "q21_10_tier"]
        known_edu = core["h_q21_07"].notna()
        B.put(frame, known_edu & core["h_q21_07"].eq(H.YES), "q21_07", True)
        frame.loc[known_edu, "q21_07_beneficiary"] = core.loc[known_edu, "q21_07_beneficiary"]
    return {"gold": gold, "trace": trace, "log": log, "auto_core": auto_core, "auto_care": auto_care,
            "core": core, "variants": variants, "candidates": candidates}


def analyze(m: pd.DataFrame, demand: pd.DataFrame, built: dict | None = None) -> tuple[dict[str, pd.DataFrame], dict]:
    built = built or layers(m)
    gold, trace, log, auto_core, auto_care, core, variants, candidates = (
        built[key] for key in ["gold", "trace", "log", "auto_core", "auto_care", "core", "variants", "candidates"])
    models = {"official": m, "revised_core": auto_core, "revised_care": auto_care}
    frames = H.evaluate(gold, trace, models)
    frames.update(human_gold=gold, human_label_trace=trace, human_auto_errors=error_cases(gold, trace, m, auto_core))

    supply, ranks, stability = [], [], []
    for variant, frame in {"official": m, "automatic_core": auto_core, "automatic_care": auto_care, **variants}.items():
        s = S.supply_by_scenario(frame).query("econ_rule in @RULES or econ_rule == 'v3'").copy()
        if variant != "official":
            s = s.loc[s.econ_rule.ne("v3")].copy()
        reviewed = frame.servId.isin(gold.servId)
        for (scenario, rule), group in s.groupby(["scenario", "econ_rule"]):
            _, scopes, bases, _ = next(spec for spec in S.SCENARIOS if spec[0] == scenario)
            included = S.in_scenario(frame, scopes, bases) & reviewed
            hits = S.area_matrix(frame, rule).loc[included].sum()
            s.loc[group.index, "n_reviewed_in_scenario"] = int(included.sum())
            s.loc[group.index, "n_candidates_from_reviewed_services"] = group.area.map(hits)
        for field in ["n_reviewed_in_scenario", "n_candidates_from_reviewed_services"]:
            s[field] = s[field].astype(int)
        s["n_candidates_from_unreviewed_services"] = s.n_services - s.n_candidates_from_reviewed_services
        s["n_input_services"] = len(m)
        supply.append(s.assign(variant=variant))
        comp = s.assign(in_demand_comparison=s.area.isin(S.DEMAND_AREAS),
                        is_reference=s.scenario.eq(C.REFERENCE_SCENARIO) & s.econ_rule.eq(C.REFERENCE_RULE),
                        main_spec_status="conditional")
        rank = C.demand_supply_rank(demand, comp)
        rho = C.rank_stability_by_cell(rank).assign(main_spec_status="conditional")
        ranks.append(rank.assign(variant=variant))
        stability.append(rho.assign(variant=variant))

    frames["final_supply"] = pd.concat(supply, ignore_index=True)
    frames["final_rank"] = pd.concat(ranks, ignore_index=True)
    frames["final_rank_stability"] = pd.concat(stability, ignore_index=True)
    frames["final_access_distribution"] = candidate_details(core, REFERENCE)
    frames["final_boundary_candidates"] = candidates
    layer_cols = ["servId", "q1_human_reviewed", "youth_scope", "youth_scope_basis", "need_area_q21",
                  "q21_10_tier", "q21_07_beneficiary", "apply_barrier", "use_outing", *H.FIELDS]
    frames["final_supply_layer"] = core[layer_cols]

    ref = frames["final_rank"].query("scenario == @C.REFERENCE_SCENARIO and econ_rule == @C.REFERENCE_RULE")
    reference_counts = ref.pivot(index="area", columns="variant", values="n_services")
    reference_ranks = ref.pivot(index="area", columns="variant", values="supply_rank")
    result = (reference_counts[["official", "automatic_core", REFERENCE, "combined_core"]]
              .rename(columns=lambda col: col + "_count").join(
                  reference_ranks[["official", "automatic_core", REFERENCE, "combined_core"]]
                  .rename(columns=lambda col: col + "_rank")))
    frames["final_reference_comparison"] = result.reset_index()
    core_top = set(reference_ranks.index[reference_ranks[REFERENCE].le(3)])
    combined_top = set(reference_ranks.index[reference_ranks["combined_core"].le(3)])
    log.update({"master_rows": len(m), "known_development_overlap": int(gold.known_development_overlap.sum()),
                "demand_areas": len(S.DEMAND_AREAS), "core_top3": sorted(core_top), "combined_top3": sorted(combined_top),
                "top3_changed_at_boundary": core_top != combined_top,
                "status": "conditional_pending_definition"})
    return frames, log


def save(frames: dict[str, pd.DataFrame], log: dict) -> None:
    tables = OUT / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    for name, frame in frames.items():
        frame.to_csv(tables / f"{name}.csv", **H.CSV)
    (tables / "human_gold_build.json").write_text(json.dumps(log, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
