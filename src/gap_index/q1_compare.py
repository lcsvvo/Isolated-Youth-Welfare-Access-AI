"""Q1 본 분석: 공급 후보 구성(서비스 건수)과 수요 순위의 비교표.

- 공급은 q1_supply의 시나리오·⑩ 규칙 격자(4 × 3)로 본다. 격자 밖 변형(v4_main_edu_self)은 in_grid=False로 따로 표시한다.
- 수요(응답자 비율)와 공급(서비스 건수)은 단위가 달라 나누지 않는다. 순위와 구성만 나란히 놓는다.
  rank_gap_4(= 공급 순위 − 수요 순위)는 순위 차이일 뿐 부족 지표가 아니다.
- S2_AB_strict × v4_main은 최종 대표 시나리오가 아니라 임시 참조값(provisional)이다.
- 라벨은 규칙 후보이고 사람 검수 전이다. 0건·소수는 '서비스 부재'와 '규칙 누락'을 구분할 수 없다.
- ⑪ 주거는 조사 문항이 없어 공급표에는 남기고 순위 비교에서는 뺀다.

이 모듈은 파일을 쓰지 않는다(저장은 notebooks/03_q1.ipynb).
"""
from __future__ import annotations

import math

import pandas as pd

from ..preprocess.keywords import Q21_LABELS
from .q1_supply import (AREAS, ECON_RULES, SCENARIOS, area_matrix, edu_beneficiary_table, in_scenario, local_access,
                        supply_by_scenario)

REFERENCE_SCENARIO = "S2_AB_strict"
REFERENCE_RULE = "v4_main"
MAIN_SPEC_STATUS = "provisional"
SCENARIOS4 = ["S1_A", "S2_AB_strict", "S2_AB", "S3_ABU"]
RULES3 = ["v4_main", "v4_all", "v3"]
EDU_RULE = "v4_main_edu_self"
SUPPLY_METRICS = ["n_services", "n_norm_names", "fractional_count"]
DEMAND_MEASURES = {"very_needed_4": "rate_very_needed_4", "needed_3_4": "rate_needed_3_4"}
PATTERN_NOTE = "검토 대상 패턴(공급 부족 확정 아님)"


def _scn(name: str):
    return next(s for s in SCENARIOS if s[0] == name)


# ----------------------------------------------------------------------------- 공급
def supply_table(m: pd.DataFrame, supply: pd.DataFrame | None = None) -> pd.DataFrame:
    """시나리오 4개 × ⑩ 규칙(RULES3 + edu_self) × 영역 11개. ⑪은 보존하고 수요 비교 순위(rank_in_demand_areas)는 빈값."""
    s = supply_by_scenario(m) if supply is None else supply
    t = s[s["scenario"].isin(SCENARIOS4) & s["econ_rule"].isin(RULES3 + [EDU_RULE])].copy()
    t["in_demand_comparison"] = t["area"].ne("q21_11")
    t["label_status"] = "provisional_rule_candidate"
    t["main_spec_status"] = MAIN_SPEC_STATUS
    t["is_reference"] = t["scenario"].eq(REFERENCE_SCENARIO) & t["econ_rule"].eq(REFERENCE_RULE)
    return t.reset_index(drop=True)


def supply_evidence_review(m: pd.DataFrame) -> pd.DataFrame:
    """v4_main 기준 시나리오·영역별로 청년 범위 값(youth_scope)과 근거(basis)의 고유 서비스 수, 검수 필요 표시 수."""
    mat = area_matrix(m, REFERENCE_RULE)
    rows = []
    for name in SCENARIOS4:
        _, scopes, bases, _ = _scn(name)
        sel = in_scenario(m, scopes, bases)
        for q in AREAS:
            hit = m[sel & mat[q]]
            if hit.empty:
                continue
            for (sc, basis), g in hit.groupby(["youth_scope", "youth_scope_basis"]):
                row = {"scenario": name, "econ_rule": REFERENCE_RULE, "area": q, "area_label": Q21_LABELS[q],
                       "youth_scope": sc, "youth_scope_basis": basis, "n_services": g["servId"].nunique(),
                       "n_youth_scope_review": int(g["youth_scope_review"].eq("True").sum()),
                       "n_area_review": int(g["need_area_q21_review"].eq("True").sum())}
                if "need_area_q21_method" in m.columns:
                    row["n_method_rule"] = int(g["need_area_q21_method"].fillna("").str.contains("rule").sum())
                rows.append(row)
    return pd.DataFrame(rows)


def multilabel_overlap(m: pd.DataFrame) -> pd.DataFrame:
    """영역별 건수의 합은 고유 서비스 수가 아니다(한 서비스가 여러 영역). 합, 고유 수, 다중 라벨 비중을 나란히 둔다."""
    rows = []
    for name in SCENARIOS4:
        _, scopes, bases, _ = _scn(name)
        sel = in_scenario(m, scopes, bases)
        for rule in RULES3 + [EDU_RULE]:
            mat = area_matrix(m, rule)[AREAS][sel]
            k = mat.sum(axis=1)
            any_area = int((k > 0).sum())
            rows.append({"scenario": name, "econ_rule": rule, "n_services_in_scenario": int(sel.sum()),
                         "n_with_any_area": any_area, "n_no_area": int(sel.sum()) - any_area,
                         "sum_area_counts": int(k.sum()),
                         "n_unique_any_area": int(m.loc[sel, "servId"][k.to_numpy() > 0].nunique()),
                         "sum_over_unique": k.sum() / any_area if any_area else math.nan,
                         "n_multi_label": int((k >= 2).sum()),
                         "share_multi_label": (k >= 2).sum() / any_area if any_area else math.nan})
    return pd.DataFrame(rows)


def local_access_table(m: pd.DataFrame) -> pd.DataFrame:
    """참조 시나리오에서 시군구 한 곳이 접근할 수 있는 후보 수 분포(중앙 + 해당 시도 본청 + 해당 시군구)의 요약. 잠정."""
    desc, _ = local_access(m, REFERENCE_SCENARIO)
    desc["label_status"] = "provisional_rule_candidate"
    desc["main_spec_status"] = MAIN_SPEC_STATUS
    return desc


# ----------------------------------------------------------------------------- 수요·공급 순위
def demand_supply_rank(demand: pd.DataFrame, supply: pd.DataFrame) -> pd.DataFrame:
    """영역 10개 × 시나리오 4 × 규칙(RULES3 + edu_self). 수요는 은둔 집단. ⑪ 제외.
    supply_rank = rank_in_demand_areas(method=min). rank_gap_4 = supply_rank − demand_rank_4(순위 차이일 뿐 부족 지표가 아님)."""
    d = demand[demand["group"].eq("hikikomori")] if "group" in demand.columns else demand
    dw = d.pivot(index="area", columns="measure", values=["rate", "rank"])
    dem = pd.DataFrame({"rate_very_needed_4": dw[("rate", "very_needed_4")], "rate_needed_3_4": dw[("rate", "needed_3_4")],
                        "demand_rank_4": dw[("rank", "very_needed_4")], "demand_rank_34": dw[("rank", "needed_3_4")]}).reset_index()
    s = supply[supply["in_demand_comparison"]].copy()
    cols = ["area", "area_label", "scenario", "econ_rule", "is_reference", "main_spec_status", "n_services", "n_central", "n_local",
            "n_norm_names", "fractional_count", "rank_in_demand_areas"]
    t = s[cols].merge(dem, on="area", how="left", validate="many_to_one")
    t = t.rename(columns={"rank_in_demand_areas": "supply_rank"})
    t["in_grid"] = t["econ_rule"].isin(RULES3)
    t[["supply_rank", "demand_rank_4", "demand_rank_34"]] = t[["supply_rank", "demand_rank_4", "demand_rank_34"]].astype(int)
    t["rank_gap_4"] = t["supply_rank"] - t["demand_rank_4"]
    order = ["area", "area_label", "scenario", "econ_rule", "in_grid", "is_reference", "main_spec_status", "rate_very_needed_4",
             "rate_needed_3_4", "demand_rank_4", "demand_rank_34", "n_services", "n_central", "n_local", "n_norm_names",
             "fractional_count", "supply_rank", "rank_gap_4"]
    t = t[order]
    t["_s"] = t["scenario"].map({n: i for i, n in enumerate(SCENARIOS4)})
    t["_r"] = t["econ_rule"].map({n: i for i, n in enumerate(RULES3 + [EDU_RULE])})
    return t.sort_values(["_s", "_r", "demand_rank_4", "area"]).drop(columns=["_s", "_r"]).reset_index(drop=True)


def _spearman(x: pd.Series, y: pd.Series) -> float:
    """평균순위(동점은 평균) Spearman. 어느 쪽이 상수면 NaN. p값은 만들지 않는다."""
    if x.nunique() < 2 or y.nunique() < 2:
        return math.nan
    return float(x.rank().corr(y.rank()))


def _n_tied(v: pd.Series) -> int:
    """다른 영역과 값이 같은 영역의 수."""
    c = v.value_counts()
    return int(c[c > 1].sum())


def rank_stability_by_cell(rank: pd.DataFrame) -> pd.DataFrame:
    """격자 12칸(+ edu_self 4칸, in_grid=False) × 수요 측정 2종 × 공급 지표 3종의 Spearman(수요 비율, 공급 값).
    양의 값 = 수요 비율이 높은 영역에 공급 건수도 많은 경향. n=10·동점 많음 → 탐색적 기술값이다."""
    ref = rank[rank["scenario"].eq(REFERENCE_SCENARIO) & rank["econ_rule"].eq(REFERENCE_RULE)].set_index("area")
    rows = []
    for (sc, rule), g in rank.groupby(["scenario", "econ_rule"], sort=False):
        g = g.set_index("area")
        for dm, rate_col in DEMAND_MEASURES.items():
            top3 = g[rate_col].rank(ascending=False, method="min")
            top3 = g.assign(_d=top3).sort_values(["_d"], kind="stable").head(3).index
            for sm in SUPPLY_METRICS:
                srank = g[sm].rank(ascending=False, method="min").astype(int)
                rows.append({"scenario": sc, "econ_rule": rule, "in_grid": bool(g["in_grid"].iloc[0]), "supply_metric": sm,
                             "demand_measure": dm, "spearman_rho": _spearman(g[rate_col], g[sm]), "n_areas": len(g),
                             "n_tied_supply_values": _n_tied(g[sm]), "n_areas_zero_supply": int((g[sm] == 0).sum()),
                             "n_tied_demand_values": _n_tied(g[rate_col]),
                             "rho_vs_reference_supply_rank": _spearman(g[sm], ref[sm].reindex(g.index)),
                             "top3_demand_supply_ranks": ",".join(str(int(srank[a])) for a in top3),
                             "main_spec_status": MAIN_SPEC_STATUS})
    return pd.DataFrame(rows)


def rank_stability_by_area(rank: pd.DataFrame) -> pd.DataFrame:
    """12칸(in_grid)에서 영역별 공급 순위의 범위. 수요 상위인데 공급 순위가 8위 이하인 칸 수를 센다(공급 부족 확정 아님).
    수요 3·4위의 비율 차이가 작아 3위 경계만 보면 결과가 흔들리므로, 보조 지표 3위 이내와 주 지표 5위 이내도 함께 센다(결과를 본 뒤 추가한 탐색적 기준)."""
    g = rank[rank["in_grid"]]
    ref = g[g["is_reference"]].set_index("area")["supply_rank"]
    rows = []
    for a, x in g.groupby("area"):
        low = x["supply_rank"] >= 8
        n_flag = int(((x["demand_rank_4"] <= 3) & low).sum())
        n_flag34 = int(((x["demand_rank_34"] <= 3) & low).sum())
        n_flag5 = int(((x["demand_rank_4"] <= 5) & low).sum())
        rows.append({"area": a, "area_label": x["area_label"].iloc[0], "demand_rank_4": int(x["demand_rank_4"].iloc[0]),
                     "demand_rank_34": int(x["demand_rank_34"].iloc[0]), "supply_rank_min": int(x["supply_rank"].min()),
                     "supply_rank_median": float(x["supply_rank"].median()), "supply_rank_max": int(x["supply_rank"].max()),
                     "supply_rank_range": int(x["supply_rank"].max() - x["supply_rank"].min()), "reference_supply_rank": int(ref[a]),
                     "n_services_min": int(x["n_services"].min()), "n_services_max": int(x["n_services"].max()),
                     "n_cells_zero": int((x["n_services"] == 0).sum()), "n_cells_demand_top3_supply_bottom3": n_flag,
                     "n_cells_demand34_top3_supply_bottom3": n_flag34, "n_cells_demand_top5_supply_bottom3": n_flag5,
                     "pattern_note": PATTERN_NOTE if max(n_flag, n_flag34, n_flag5) > 0 else "", "main_spec_status": MAIN_SPEC_STATUS})
    return pd.DataFrame(rows).sort_values(["demand_rank_4", "area"]).reset_index(drop=True)


# ----------------------------------------------------------------------------- ⑩·⑦ 규칙 비교
def econ_rule_comparison(m: pd.DataFrame, supply: pd.DataFrame | None = None) -> pd.DataFrame:
    """시나리오 4개 + ALL(전체 5,288, REF_all) × ⑩ 규칙 3종. ALL 행의 합계는 q1_supply.econ_comparison과 같아야 한다."""
    s = supply_by_scenario(m) if supply is None else supply
    names = SCENARIOS4 + ["REF_all"]
    rows = []
    for name in names:
        _, scopes, bases, _ = _scn(name)
        sel = in_scenario(m, scopes, bases)
        for rule in RULES3:
            mat = area_matrix(m, rule)
            hit = sel & mat["q21_10"]
            other = mat.drop(columns="q21_10").any(axis=1)
            r = s[s["scenario"].eq(name) & s["econ_rule"].eq(rule) & s["area"].eq("q21_10")].iloc[0]
            rows.append({"scenario": "ALL" if name == "REF_all" else name, "econ_rule": rule, "description": ECON_RULES[rule],
                         "n_q21_10": int(r["n_services"]), "n_central": int(r["n_central"]), "n_local": int(r["n_local"]),
                         "share_of_scenario": r["share_of_scenario"],
                         "overlap_with_other_area": (hit & other).sum() / hit.sum() if hit.sum() else math.nan,
                         "rank_in_demand_areas": r["rank_in_demand_areas"]})
    t = pd.DataFrame(rows)
    base = t[t["econ_rule"].eq(REFERENCE_RULE)].set_index("scenario")["n_q21_10"]
    t["delta_vs_v4_main"] = t["n_q21_10"] - t["scenario"].map(base)
    return t


def econ_type_by_scenario(m: pd.DataFrame) -> pd.DataFrame:
    """시나리오별 ⑩ 단계(tier)·유형(type) 건수. 마스터에서 type=none인 서비스는 tier가 빈칸(결측)이다."""
    rows = []
    for name in SCENARIOS4:
        _, scopes, bases, _ = _scn(name)
        sel = in_scenario(m, scopes, bases)
        t = (m[sel].groupby(["q21_10_tier", "q21_10_type"], dropna=False).size().rename("n_services").reset_index())
        t.insert(0, "scenario", name)
        t["share_of_scenario"] = t["n_services"] / int(sel.sum())
        rows.append(t)
    return pd.concat(rows, ignore_index=True)


def edu_beneficiary_effect(m: pd.DataFrame, supply: pd.DataFrame | None = None) -> pd.DataFrame:
    """⑦ 교육·학습에서 수혜자가 자녀·아동뿐인(dependent) 서비스를 빼면 ⑦ 건수와 순위가 어떻게 바뀌는가."""
    s = supply_by_scenario(m) if supply is None else supply
    et = edu_beneficiary_table(m)
    rows = []
    for name in SCENARIOS4:
        e = et[et["scenario"].eq(name)].set_index("beneficiary")
        n_by = e["n_q21_07"].astype(int)
        pick = lambda rule: s[s["scenario"].eq(name) & s["econ_rule"].eq(rule) & s["area"].eq("q21_07")].iloc[0]  # noqa: E731
        b, a = pick(REFERENCE_RULE), pick(EDU_RULE)
        total = int(n_by.sum())
        assert int(b["n_services"]) == total, f"{name}: 수혜자 합계와 v4_main ⑦ 건수 불일치"
        assert int(b["n_services"]) - int(a["n_services"]) == int(n_by["dependent"]), f"{name}: before − after ≠ dependent"
        rows.append({"scenario": name, "n_q21_07_v4_main": total, "n_dependent": int(n_by["dependent"]), "n_self": int(n_by["self"]),
                     "n_both": int(n_by["both"]), "n_N": int(n_by["N"]), "n_q21_07_excl_dependent": int(a["n_services"]),
                     "rank_before": int(b["rank_in_demand_areas"]), "rank_after": int(a["rank_in_demand_areas"]),
                     "n_added_by_econ_v4": int(e["n_added_by_econ_v4"].sum()),
                     "share_dependent": n_by["dependent"] / total if total else math.nan})
    return pd.DataFrame(rows)
