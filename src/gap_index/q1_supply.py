"""Q1 공급 후보 구성의 민감도 분석 (청년 범위 youth_scope 시나리오별).

질문: 은둔 청년의 영역별 지원 필요도(조사 응답 비율)와, 청년이 이용할 수 있는 후보로 볼 수 있는 SSIS 서비스의
영역 구성(등록 사업 건수)이 청년 범위 정의에 따라 어떻게 달라지는가?

- 수요(응답자 비율)와 공급(등록 사업 수)은 단위가 달라 나누지 않는다. 같은 영역의 순위·구성만 나란히 놓는다.
- 서비스 건수는 정원·예산·실제 수급 가능성이 아니다. 지자체 사업은 그 지역 주민만 쓸 수 있으므로
  전국 합계와 함께 '시군구 한 곳에서 접근 가능한 후보 수'(중앙 + 해당 시도 본청 + 해당 시군구)를 따로 본다.
- 라벨은 규칙 후보(사람 검증 전)다.

실행: python -m src.gap_index.q1_supply  → logs/validation/q1_supply_sensitivity/*.csv
"""
from __future__ import annotations

import math
import re

import pandas as pd

from ..collect._common import PROCESSED, ROOT
from ..preprocess.keywords import Q21_LABELS

OUT = ROOT / "logs" / "validation" / "q1_supply_sensitivity"
AREAS = list(Q21_LABELS)                       # q21_01 ~ q21_11
DEMAND_AREAS = AREAS[:10]                      # ⑪ 주거는 조사 문항이 없다

# 시나리오: (이름, 포함할 youth_scope, 포함할 basis 제한(None=전부), 설명)
SCENARIOS = [
    ("S1_A", {"A"}, None, "청년 특화만 (보수적)"),
    ("S2_AB_strict", {"A", "B"}, {"age_overlap"}, "A + 연령 근거가 있는 B (태그·언급만인 B 제외)"),
    ("S2_AB", {"A", "B"}, None, "A + B (확장)"),
    ("S3_ABU", {"A", "B", "U"}, None, "A + B + 판단 불가 U (상한, 불확실 포함, 피부양자 연령 U 제외)"),
    ("REF_all", {"A", "B", "U", "C"}, None, "전체 5,288건 (참고: 기존 EDA 분모)"),
]


def load_master() -> pd.DataFrame:
    return pd.read_csv(PROCESSED / "services_master.csv", dtype=str, encoding="utf-8-sig",
                       keep_default_na=False, na_values=[""])


ECON_RULES = {
    "v3": "개정 전(v3): 지급 방식 단어·제공유형만으로도 ⑩",
    "v4a_main": "v4 1차(2026-11-05 오전, 보존): 본 분석 기준",
    "v4a_all": "v4 1차(보존): 본 분석 + 후보",
    "v4_main": "v4 본 분석: 생계·생활비 명시 + 필수 생활비 경감만 ⑩",
    "v4_all": "v4 민감도: 본 분석 + 후보(일시금·보상·자산형성·용도 미상 현금)",
    "v4_main_edu_self": "v4 본 분석 + ⑦에서 수혜자가 자녀·아동만인 서비스(dependent) 제외",
}
# v4 1차 라벨은 재분류 전 스냅샷으로 남긴다(코드로 다시 만들 수 없음)
V4A_LABELS = ROOT / "logs" / "validation" / "econ_v4_review" / "econ_v4a_labels.csv"


def area_matrix(m: pd.DataFrame, econ: str = "v4_main") -> pd.DataFrame:
    """서비스 × 영역 0/1. ⑩은 econ 기준으로 정한다. v3는 개정 전 영역 전체(need_area_q21_v3)를 쓴다."""
    if econ.startswith("v4a"):
        snap = pd.read_csv(V4A_LABELS, dtype=str, encoding="utf-8-sig", keep_default_na=False).set_index("servId")
        area, tier = m["servId"].map(snap["need_area_q21"]), m["servId"].map(snap["q21_10_tier"])
    else:
        area = m["need_area_q21_v3"] if econ == "v3" else m["need_area_q21"]
        tier = m["q21_10_tier"]
    labs = area.fillna("N").str.split("|")
    mat = pd.DataFrame({q: labs.map(lambda v, q=q: q in v) for q in AREAS}, index=m.index)
    if econ.endswith("_main") or econ == "v4_main_edu_self":
        mat["q21_10"] = tier.eq("main")
    elif econ.endswith("_all"):
        mat["q21_10"] = tier.isin(["main", "candidate"])
    if econ == "v4_main_edu_self":
        mat["q21_07"] &= m["q21_07_beneficiary"].ne("dependent")
    if "q1_education_direct_only" in m and econ.startswith("v4"):
        mat["q21_07"] &= (~m["q1_education_direct_only"]
                          | m["q21_07_beneficiary"].isin(["self", "both"]))
    return mat


def edu_beneficiary_table(m: pd.DataFrame) -> pd.DataFrame:
    """시나리오별 ⑦ 교육 지원의 수혜자 구성(dependent = 자녀·아동·초중고생, self = 청년 본인·성인, both, N)."""
    rows = []
    for name, scopes, bases, desc in SCENARIOS:
        sel = in_scenario(m, scopes, bases) & m["need_area_q21"].fillna("").str.split("|").map(lambda v: "q21_07" in v)
        vc = m.loc[sel, "q21_07_beneficiary"].fillna("N").value_counts()
        from_econ = sel & ~m["need_area_q21_v3"].fillna("").str.split("|").map(lambda v: "q21_07" in v)
        for b in ["dependent", "self", "both", "N"]:
            rows.append({"scenario": name, "beneficiary": b, "n_q21_07": int(vc.get(b, 0)),
                         "share": vc.get(b, 0) / sel.sum() if sel.sum() else math.nan,
                         "n_added_by_econ_v4": int((from_econ & m["q21_07_beneficiary"].fillna("N").eq(b)).sum())})
    return pd.DataFrame(rows)


# Q21은 청년 본인의 필요이므로, 연령이 자녀 등 피부양자를 가리키는 U는 공급 후보 상한에서도 뺀다(전체 참고 REF_all만 포함)
EXCLUDE_FROM_S3 = {"age_may_refer_to_dependent"}


def in_scenario(m: pd.DataFrame, scopes: set[str], bases: set[str] | None) -> pd.Series:
    s = m["youth_scope"].isin(scopes)
    if bases is not None:
        s &= m["youth_scope"].ne("B") | m["youth_scope_basis"].isin(bases)
    if scopes != {"A", "B", "U", "C"}:
        s &= ~m["youth_scope_basis"].isin(EXCLUDE_FROM_S3)
    return s


def scenario_membership(m, scenario, scopes, bases):
    """Return base eligibility and the extension restricted to open-space area ①."""
    base = in_scenario(m, scopes, bases)
    if "q1_age_exception_only" in m:
        exception = m["q1_age_exception_only"]
        if scenario in {"S1_A", "S2_AB_strict"}:
            base &= ~exception
        elif scenario in {"S2_AB", "S3_ABU"}:
            base |= exception
    added = pd.Series(False, index=m.index)
    if scenario in {"S2_AB_strict", "S2_AB", "S3_ABU"} and "q1_open_space_universal" in m:
        added = m["q1_open_space_universal"] & ~base
    return base, added


def norm_name(m: pd.DataFrame) -> pd.Series:
    """지역명을 지운 서비스명(지역별 반복 사업을 가늠하는 보조 지표, 같은 사업이라는 확정이 아님)."""
    words = set(m["region_sido"].dropna()) | set(m["region_sgg"].dropna())
    stems = sorted({w for r in words for w in [r, re.sub(r"(특별자치시|특별자치도|특별시|광역시|통합특별시|도|시|군|구)$", "", r)]
                    if len(w) >= 2}, key=len, reverse=True)
    rx = re.compile("|".join(map(re.escape, stems)))
    return m["servNm"].fillna("").map(lambda s: re.sub(r"[\s\(\)\[\]·ㆍ,\-'\"‘’“”]", "", rx.sub("", s)))


def scope_table(m: pd.DataFrame) -> pd.DataFrame:
    t = m.groupby(["youth_scope", "youth_scope_basis", "source"]).size().unstack("source", fill_value=0)
    t["total"] = t.sum(axis=1)
    t["review_true"] = m.groupby(["youth_scope", "youth_scope_basis"])["youth_scope_review"].apply(lambda s: int(s.eq("True").sum()))
    return t.reset_index()


def supply_by_scenario(m: pd.DataFrame, *, rules=ECON_RULES) -> pd.DataFrame:
    """시나리오 × 영역: 고유 서비스 수, 시나리오 내 비율(다중 라벨이라 합이 100%를 넘음), 1/k 가중 건수, 정규화 사업명 수."""
    nn = norm_name(m)
    rows = []
    for econ in rules:
        mat = area_matrix(m, econ)
        k = mat.sum(axis=1).astype(float).replace(0.0, float("nan"))
        for name, scopes, bases, desc in SCENARIOS:
            sel, open_space = scenario_membership(m, name, scopes, bases)
            n_sc = int((sel | open_space).sum())
            for q in AREAS:
                eligible = sel | open_space if q == "q21_01" else sel
                hit = eligible & mat[q]
                rows.append({
                    "scenario": name, "econ_rule": econ,
                    "description": desc, "n_services_in_scenario": n_sc, "area": q, "area_label": Q21_LABELS[q],
                    "n_services": int(hit.sum()),
                    "n_central": int((hit & m["source"].eq("central")).sum()),
                    "n_local": int((hit & m["source"].eq("local")).sum()),
                    "share_of_scenario": hit.sum() / n_sc if n_sc else math.nan,
                    "fractional_count": float((1.0 / k)[hit & sel].sum() + (hit & open_space).sum()),
                    "n_norm_names": int(nn[hit].nunique()),
                    "n_rule_review": int((hit & m["need_area_q21_review"].eq("True")).sum()),
                })
    t = pd.DataFrame(rows)
    t["rank_in_scenario"] = (t.groupby(["scenario", "econ_rule"])["n_services"]
                             .rank(ascending=False, method="min").astype(int))
    # 수요와 나란히 볼 순위는 조사 문항이 있는 10개 영역 안에서만 매긴다(⑪ 주거 제외)
    d = t[t["area"].isin(DEMAND_AREAS)]
    t["rank_in_demand_areas"] = (d.groupby(["scenario", "econ_rule"])["n_services"]
                                 .rank(ascending=False, method="min").astype("Int64"))
    return t


def local_access(m: pd.DataFrame, scenario: str = "S2_AB") -> pd.DataFrame:
    """시군구 단위 접근 가능 후보 수 분포: 중앙 + 해당 시도 본청(시군구 빈칸) + 해당 시군구.
    시군구 목록은 지자체 원자료에 등장한 (시도, 시군구) 쌍이다(행정구역 전체 목록이 아님)."""
    name, scopes, bases, _ = next(s for s in SCENARIOS if s[0] == scenario)
    sel = in_scenario(m, scopes, bases)
    mat = area_matrix(m, "v4_main")
    c, loc = sel & m["source"].eq("central"), sel & m["source"].eq("local")
    pairs = m.loc[m["source"].eq("local") & m["region_sgg"].notna(), ["region_sido", "region_sgg"]].drop_duplicates()
    rows = []
    for sido, sgg in pairs.itertuples(index=False):
        acc = c | (loc & m["region_sido"].eq(sido) & (m["region_sgg"].isna() | m["region_sgg"].eq(sgg)))
        rows.append({"region_sido": sido, "region_sgg": sgg, "n_total": int(acc.sum()),
                     **{q: int((acc & mat[q]).sum()) for q in AREAS}})
    per = pd.DataFrame(rows)
    desc = per[["n_total", *AREAS]].describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]).T
    desc["n_regions_zero"] = [(per[c_] == 0).sum() for c_ in ["n_total", *AREAS]]
    desc.insert(0, "scenario", scenario)
    return desc.reset_index().rename(columns={"index": "area"}), per


def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    if n == 0:
        return math.nan, math.nan
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def demand_table(path=None) -> pd.DataFrame:
    """은둔 집단 Q21 영역별 '매우 필요(4)'·'필요(3+4)' 비율과 Wilson 95% 구간(비가중, 응답자 기준, 무응답 제외).
    비확률 표본이므로 구간은 '이 응답자 집단 안의 표집 변동'만 나타내며 모집단 추정 구간이 아니다."""
    d = pd.read_csv(path or PROCESSED / "demand_2023_hikikomori.csv", dtype=str, encoding="utf-8-sig")
    rows = []
    for i, q in enumerate(DEMAND_AREAS, start=1):
        v = d[q].dropna()
        n = len(v)
        for lab, ok in (("very_needed_4", v.eq("4")), ("needed_3_4", v.isin(["3", "4"]))):
            k = int(ok.sum())
            lo, hi = wilson(k, n)
            rows.append({"area": q, "area_label": Q21_LABELS[q], "measure": lab, "n_respondents": n, "k": k,
                         "rate": k / n if n else math.nan, "ci95_lo": lo, "ci95_hi": hi})
    t = pd.DataFrame(rows)
    t["rank"] = t.groupby("measure")["rate"].rank(ascending=False, method="min").astype(int)
    return t


def side_by_side(demand: pd.DataFrame, supply: pd.DataFrame) -> pd.DataFrame:
    """영역별 수요 순위와 시나리오별 공급 구성 순위를 나란히 둔다(나눗셈·비율 지표를 만들지 않는다)."""
    dm = demand.pivot(index="area", columns="measure", values="rank").add_prefix("demand_rank_")
    sp = (supply[supply["econ_rule"].eq("v4_main")]
          .pivot(index="area", columns="scenario", values="rank_in_demand_areas").add_prefix("supply_rank_"))
    sp2 = (supply[supply["econ_rule"].eq("v4_all")]
           .pivot(index="area", columns="scenario", values="rank_in_demand_areas").add_prefix("supply_rank_econall_"))
    out = dm.join(sp, how="left").join(sp2)
    out.insert(0, "area_label", [Q21_LABELS[a] for a in out.index])
    return out.reset_index()


# 신청 장벽 → 신청 외출 환산 두 안. 저장소 v2: 방문 신청 = 1(근처 짧은 외출). 마스터 v3(2026-11-05)는 회의 문서 안을 채택(잠정).
# 3차 회의 '외출 필요도 정의' 문서(2026-11-03): 방문 신청 = 외출 필수(2), 면접은 대면 확인 시 2.
APPLY_TO_OUTING = {"repo_v2": {"0": 0, "1": 0, "2": 1, "3": 2}, "meeting_doc": {"0": 0, "1": 0, "2": 2, "3": 2}}


def outing_mapping_sensitivity(m: pd.DataFrame) -> pd.DataFrame:
    """두 환산안의 outing_need 분포 비교. 면접 형태가 비대면이면 apply_channel, 불명이면 신청 쪽 N(마스터 규칙과 같음)."""
    a = m["apply_barrier"].where(m["apply_interview"].ne("비대면"), m["apply_channel"])
    a = a.where(m["apply_interview"].ne("불명"), "N")
    u = m["use_outing"]
    rows = []
    for name, mp in APPLY_TO_OUTING.items():
        def one(av, uv):
            parts = ([mp[av]] if av in mp else []) + ([int(uv)] if uv != "N" else [])
            if not parts:
                return "N"
            lo = max(parts)
            return str(lo) if len(parts) == 2 or lo == 3 else "N"
        v = pd.Series([one(x, y) for x, y in zip(a, u)], index=m.index)
        for val, n in v.value_counts().sort_index().items():
            rows.append({"mapping": name, "outing_need": val, "n": int(n)})
    # 마스터 현재값(v3: 방문 신청 2 + 원문상 방문 필수 반영)도 함께 둔다
    for val, n in m["outing_need"].value_counts().sort_index().items():
        rows.append({"mapping": "master_v3", "outing_need": val, "n": int(n)})
    t = pd.DataFrame(rows)
    chk = t[t["mapping"].eq("repo_v2")].set_index("outing_need")["n"]
    cur = m["outing_need_v2"].value_counts()
    if not chk.reindex(cur.index, fill_value=0).eq(cur).all():
        raise AssertionError("repo_v2 재계산이 마스터 outing_need_v2와 다릅니다")
    return t


def econ_comparison(m: pd.DataFrame) -> pd.DataFrame:
    """⑩ 재분류 전후 비교: 기준별 ⑩ 서비스 수, 다른 영역과의 중복률, 유형별 구성."""
    rows = []
    for econ, desc in ECON_RULES.items():
        mat = area_matrix(m, econ)
        hit = mat["q21_10"]
        other = mat.drop(columns="q21_10").any(axis=1)
        rows.append({"econ_rule": econ, "description": desc, "n_q21_10": int(hit.sum()),
                     "n_central": int((hit & m["source"].eq("central")).sum()), "n_local": int((hit & m["source"].eq("local")).sum()),
                     "share_of_all": hit.mean(), "overlap_with_other_area": (hit & other).sum() / hit.sum() if hit.sum() else math.nan,
                     "n_any_area": int(mat.any(axis=1).sum())})
    return pd.DataFrame(rows)


def main() -> None:
    m = load_master()
    for c in ("youth_scope", "youth_scope_basis", "q21_10_basis", "q21_10_tier", "need_area_q21_v3"):
        if c not in m.columns:
            raise SystemExit(f"services_master에 {c}가 없습니다. python -m src.preprocess.ssis_master로 다시 만드세요.")
    OUT.mkdir(parents=True, exist_ok=True)
    supply = supply_by_scenario(m)
    demand = demand_table()
    acc_desc, acc_per = local_access(m, "S2_AB")
    acc_desc1, _ = local_access(m, "S1_A")
    tables = {
        "youth_scope_counts": scope_table(m),
        "supply_by_scenario": supply,
        "demand_hikikomori_q21": demand,
        "rank_side_by_side": side_by_side(demand, supply),
        "local_access_summary": pd.concat([acc_desc1, acc_desc], ignore_index=True),
        "local_access_by_sgg_S2_AB": acc_per,
        "outing_apply_mapping_sensitivity": outing_mapping_sensitivity(m),
        "econ_comparison": econ_comparison(m),
        "edu_beneficiary_by_scenario": edu_beneficiary_table(m),
        "econ_type_by_v3_basis": (m.assign(v3=m["q21_10_basis_v3"].fillna("v3_no_q21_10"))
                                  .groupby(["q21_10_type", "q21_10_tier", "v3"], dropna=False).size().rename("n").reset_index()),
    }
    for k, t in tables.items():
        t.to_csv(OUT / f"{k}.csv", index=False, encoding="utf-8-sig")
    print("저장:", ", ".join(f"{k}.csv" for k in tables), "→", OUT.relative_to(ROOT))


if __name__ == "__main__":
    main()
