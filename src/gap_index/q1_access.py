"""신청·이용 조건을 적용했을 때 남는 Q1 등록 서비스 후보 집계.

- 확인 대상은 조건을 동시에 충족하는 등록 서비스 ID 수. 실제 이용률·이용 가능 인원은 추정하지 않음.
- 사람 판정 반영 레이어(final_supply_layer)와 공통 마스터를 읽기만 함.
- N은 접근 불가가 아닌 확인 불가로 별도 집계.
- 집 조건 후보의 원문 검증표(logs/validation/q1_access_verification/)는 사람 검수 원본과 별도 입력. 사람 판정은 바꾸지 않음.
- 엄격 기준은 본인 온라인 신청 또는 별도 신청 불필요, 그리고 외출 없이 지원 수령·이용.
- 전화·우편·팩스·이메일·대리·추천·직권 신청은 확장 기준. 전화 상담 자체는 집에서 이용 가능.
- 재학·재직·사업 운영 등 자격과 지원 과정에서 요구되는 대면 교육·활동을 구분함.
- 혼합 지원은 사업 전체의 집 조건을 확정할 수 없으면 확인 불가에 포함. 상한은 확인 불가 전체가 충족한다고 가정함.
- 신규 지원 중단은 운영 상태에 기록. 신규 신청 경로가 없으면 물리적 접근성을 미평가로 집계함.
"""
from __future__ import annotations

import pandas as pd

from ..collect._common import ROOT
from ..labeling import q1_human as H
from ..preprocess.keywords import Q21_LABELS
from . import q1_audit_impact as B, q1_followup as F, q1_supply as S

OUT = ROOT / "outputs/03_q1"
SCENARIO, RULE = "S2_AB_strict", "v4_main"
STAGES = {"all": "전체 등록", "youth": "청년 대상(엄격 S2)", "apply0": "온라인 신청·신청 불필요",
          "home": "신청·이용 모두 집에서", "home_final": "원문 검증 후 집", "home_final_upper": "검증 상한"}
LOW, UNKNOWN = "0", "N"
PROXY = r"대리|직권|추천"
SELF_REMOTE = r"전화|우편|팩스|메일|mail"
SMALL = 5
VERIFY = ROOT / "logs/validation/q1_access_verification"
# 집 조건 후보 원문 검증 판정. 사람 판정 선례(시술비·관람료 면제는 외출, 월세·이사비·저축 매칭은 집)에 맞춘 기준.
VERDICTS = {"home_confirmed": "집에서 신청·이용 가능", "cash_outing_activity": "현금 수령은 집, 목적 활동은 외출",
            "unclear": "신청·이용 조건 불명확", "auto_label_error": "자동 라벨 오류"}
# 누락 검증(집 조건이 아닌 청년 대상 후보) 판정. 최종 집 = home_confirmed + missed_home.
NONHOME_VERDICTS = {"missed_home": "누락 발견(집에서 신청·이용 가능)", "not_home": "집 조건 미충족 확인",
                    "unclear": "확인 불가", "extension_apply1_home": "확장안(신청 장벽 1, 이용 집)"}
ALL_LABELS = VERDICTS | NONHOME_VERDICTS | {
    "mixed": "확인 불가(혼합형)", "not_assessed": "미평가",
}
REVIEW_FIELDS = ["apply_method", "support_delivery", "eligibility_requirements", "outing_required",
                 "review_evidence", "review_source", "review_reason"]
OPERATION_FIELDS = ["operation_status", "operation_note", "operation_source"]
OPERATION_LABELS = {"new_support_discontinued": "신규 지원 중단", "round_closed": "확인한 모집 회차 마감", "round_open": "확인한 모집 회차 접수 중",
                    "rolling": "상시 접수·지급 안내", "unverified": "모집 상태 미확정"}
SOURCE_SETS = {"home": "auto_home", "nonhome": "auto_not_home"}
FINAL_HOME, FINAL_UNCLEAR = ("home_confirmed", "missed_home"), ("unclear", "mixed")
# 원문 사례. 영역별로 조건 통과·신청 단계 제외·이용 단계 제외·확인 불가를 고정 ID로 제시.
EXAMPLES = [
    ("q21_10", "WLF00006409", "온라인 신청과 계좌 입금으로 두 조건 충족"),
    ("q21_10", "WLF00004806", "현금 급여이나 방문 신청만 안내되어 신청 조건에서 제외"),
    ("q21_10", "WLF00003245", "방문 신청과 정기 참여가 함께 필요"),
    ("q21_02", "WLF00006550", "온라인 신청 경로가 있으나 프로그램 이용 방식은 원문에서 확인 불가"),
    ("q21_02", "WLF00006301", "고립·은둔 명시 사업이나 방문 신청, 이용 방식 확인 불가"),
    ("q21_02", "WLF00001139", "특정 피해 집단 대상, 신청·이용 모두 확인 불가"),
    ("q21_07", "WLF00006436", "온라인 신청과 계좌 입금. 급여 수령 기준이며 학원 수강 자체의 외출은 반영되지 않음"),
    ("q21_07", "WLF00002555", "온라인 신청 후 카드 사용처 방문이 필요해 이용 조건에서 제외"),
    ("q21_07", "WLF00000948", "센터 방문 신청과 정기 참여가 필요"),
    ("q21_03", "WLF00000055", "온라인 신청 가능, 이용권 사용처 방문이 필요"),
    ("q21_03", "WLF00006130", "온라인 신청과 현금 지원. 체육활동 자체는 외출을 전제"),
]


def layer_frame(m: pd.DataFrame, layer: pd.DataFrame | None = None) -> pd.DataFrame:
    """공통 마스터에 사람 판정 반영 레이어를 덮은 분석 사본."""
    if layer is None:
        layer = pd.read_csv(OUT / "tables/final_supply_layer.csv", dtype=str, encoding="utf-8-sig",
                            keep_default_na=False, na_values=[""])
    H.require(len(layer) == len(m) and (layer.servId.to_numpy() == m.servId.to_numpy()).all(),
              "마스터와 분석 레이어 servId 불일치")
    frame = m.copy()
    for col in layer.columns.drop("servId"):
        frame[col] = layer[col].to_numpy()
    frame["q1_human_reviewed"] = frame.q1_human_reviewed.astype(str).eq("True")
    return frame


def conditions(frame: pd.DataFrame) -> pd.DataFrame:
    """서비스별 청년 범위·신청·이용 조건 충족 여부."""
    _, scopes, bases, _ = next(spec for spec in S.SCENARIOS if spec[0] == SCENARIO)
    apply, outing = frame.apply_barrier.fillna(UNKNOWN), frame.use_outing.fillna(UNKNOWN)
    route = frame.apply_barrier_evidence.fillna("") + " " + frame.apply_cat.fillna("")
    c = pd.DataFrame({"all": True, "youth": S.in_scenario(frame, scopes, bases)}, index=frame.index)
    c["apply0"] = c.youth & apply.eq(LOW)
    c["home"] = c.apply0 & outing.eq(LOW)
    c["apply01"] = c.youth & apply.isin([LOW, "1"])
    c["home01"] = c.apply01 & outing.eq(LOW)
    c["apply_visit"] = c.youth & apply.isin(["2", "3"])
    c["apply_unknown"] = c.youth & apply.eq(UNKNOWN)
    c["apply0_outing_required"] = c.apply0 & outing.isin(["1", "2", "3"])
    c["apply0_outing_unknown"] = c.apply0 & outing.eq(UNKNOWN)
    # 확인 불가를 모두 저부담으로 가정한 상한.
    c["home_possible"] = c.youth & apply.isin([LOW, UNKNOWN]) & outing.isin([LOW, UNKNOWN])
    # 대리·추천·직권만 근거인 신청 장벽 1은 본인 비대면 경로로 보지 않음. 자동 근거 문구 기준.
    c["apply1_proxy_only"] = (c.youth & apply.eq("1") & ~frame.q1_human_reviewed
                              & route.str.contains(PROXY) & ~route.str.contains(SELF_REMOTE, case=False))
    c["home_no_contact_auto"] = c.home & frame.remote_complete.eq("2")
    c["reviewed"] = frame.q1_human_reviewed
    return c


def _stage(frame: pd.DataFrame, c: pd.DataFrame) -> pd.Series:
    """엄격 S2 후보가 도달한 마지막 조건. servId 색인."""
    stage = pd.Series("youth", index=frame.index).mask(c.apply0, "apply0").mask(c.home, "home")
    return stage.set_axis(frame.servId)


def _rank(counts: pd.Series) -> pd.Series:
    return counts.rank(method="min", ascending=False).astype(int)


def funnel(frame: pd.DataFrame, demand: pd.DataFrame) -> pd.DataFrame:
    """영역별 조건 누적 적용 후보 수. 한 서비스가 여러 영역에 걸치므로 영역 합은 고유 서비스 수가 아님."""
    mat, c = S.area_matrix(frame, RULE), conditions(frame)
    d = demand[demand.group.eq("hikikomori") & demand.measure.eq("very_needed_4")].set_index("area")
    count_cols = ["all", "youth", "apply0", "home", "apply01", "home01", "apply_visit", "apply_unknown", "apply0_outing_required",
                  "apply0_outing_unknown", "home_possible", "apply1_proxy_only", "home_no_contact_auto"]
    rows = []
    groups = {q: mat[q] for q in S.AREAS} | {"any_demand_area": mat[S.DEMAND_AREAS].any(axis=1)}
    for area, hit in groups.items():
        row = {"area": area, "area_label": Q21_LABELS.get(area, "수요 비교 10개 영역 고유 서비스"),
               "in_demand_comparison": area in S.DEMAND_AREAS,
               "rate_very_needed_4": d.rate.get(area), "demand_rank_4": d["rank"].get(area)}
        row.update({f"n_{col}": int((hit & c[col]).sum()) for col in count_cols})
        row.update({f"n_{col}_reviewed": int((hit & c[col] & c.reviewed).sum()) for col in ["youth", "apply0", "home"]})
        rows.append(row)
    t = pd.DataFrame(rows)
    base = t.n_youth.where(t.n_youth.gt(0))
    for col in ["apply0", "home", "apply01", "home01", "home_possible"]:
        t[f"share_{col}"] = t[f"n_{col}"] / base
    t["drop_youth_to_home"] = t.n_youth - t.n_home
    t["small_denominator"] = t.n_youth.lt(SMALL)
    in_rank = t.in_demand_comparison
    for col in ["youth", "apply0", "home", "home01"]:
        t.loc[in_rank, f"rank_{col}"] = _rank(t.loc[in_rank, f"n_{col}"])
    return t


def candidates(frame: pd.DataFrame) -> pd.DataFrame:
    """엄격 S2 후보의 서비스 ID 단위 근거표."""
    mat, c = S.area_matrix(frame, RULE), conditions(frame)
    keep = c.youth & mat.any(axis=1)
    source = frame.q1_human_reviewed.map({True: "human", False: "automatic"})
    t = frame.loc[keep, ["servId", "servNm", "source", "region_sido", "region_sgg", "provision_type", "apply_cat",
                         "target_condition", "apply_barrier", "use_outing"]].copy()
    t.insert(2, "areas", mat.loc[keep].apply(lambda row: "|".join(row.index[row]), axis=1))
    t[["apply_barrier", "use_outing"]] = t[["apply_barrier", "use_outing"]].fillna(UNKNOWN)
    t["label_source"] = source[keep]
    t["stage_reached"] = _stage(frame, c)[t.servId].to_numpy()
    for col in ["home01", "home_possible", "apply1_proxy_only", "home_no_contact_auto"]:
        t[col] = c.loc[keep, col]
    return t


def label_reliability(m: pd.DataFrame, gold: pd.DataFrame) -> pd.DataFrame:
    """사람 검수 236건에서 자동 라벨의 저부담 판정이 사람 판정과 맞는 정도."""
    auto = m.set_index("servId").loc[gold.servId]
    higher = ["1", "2", "3"]
    auto_apply, auto_outing = (auto[col].fillna(UNKNOWN).eq(LOW).to_numpy() for col in ["apply_barrier", "use_outing"])
    apply, outing = gold[H.APPLY], gold[H.OUTING]
    # (자동 저부담, 사람 저부담, 사람이 더 높은 단계로 확정). 나머지는 사람 확인 불가.
    pairs = {
        "apply0": (auto_apply, apply.eq(LOW).to_numpy(), apply.isin(higher).to_numpy()),
        "outing0": (auto_outing, outing.eq(LOW).to_numpy(), outing.isin(higher).to_numpy()),
        "home": (auto_apply & auto_outing, (apply.eq(LOW) & outing.eq(LOW)).to_numpy(),
                 (apply.isin(higher) | outing.isin(higher)).to_numpy()),
    }
    rows = []
    for name, (auto_yes, human_yes, human_higher) in pairs.items():
        rows.append({"condition": name, "n_reviewed": len(gold), "n_auto_yes": int(auto_yes.sum()),
                     "n_human_yes": int(human_yes.sum()), "n_both_yes": int((auto_yes & human_yes).sum()),
                     "n_auto_yes_human_unknown": int((auto_yes & ~human_yes & ~human_higher).sum()),
                     "n_auto_yes_human_higher": int((auto_yes & human_higher).sum())})
    t = pd.DataFrame(rows)
    t["auto_yes_confirmed_share"] = t.n_both_yes / t.n_auto_yes
    t["human_yes_found_share"] = t.n_both_yes / t.n_human_yes
    return t


def regional(frame: pd.DataFrame) -> pd.DataFrame:
    """시군구 한 곳의 주민이 볼 수 있는 후보 수 분포. 중앙 + 해당 시도 본청 + 해당 시군구."""
    mat, c = S.area_matrix(frame, RULE), conditions(frame)
    local = frame.source.eq("local")
    pairs = frame.loc[local & frame.region_sgg.notna(), ["region_sido", "region_sgg"]].drop_duplicates()
    rows = []
    for sido, sgg in pairs.itertuples(index=False):
        reach = ~local | (frame.region_sido.eq(sido) & (frame.region_sgg.isna() | frame.region_sgg.eq(sgg)))
        for stage in ["youth", "apply0", "home"]:
            hit = reach & c[stage]
            rows.append({"region_sido": sido, "region_sgg": sgg, "stage": stage,
                         **{q: int((hit & mat[q]).sum()) for q in S.AREAS}})
    per = pd.DataFrame(rows).melt(["region_sido", "region_sgg", "stage"], var_name="area", value_name="n")
    t = per.groupby(["area", "stage"], sort=False).n.agg(
        n_regions="size", minimum="min", median="median", maximum="max",
        n_regions_zero=lambda s: int(s.eq(0).sum())).reset_index()
    t.insert(1, "area_label", t.area.map(Q21_LABELS))
    return t


def regional_final(frame: pd.DataFrame, final: pd.DataFrame) -> pd.DataFrame:
    """수요 10개 영역의 최종 집·최종 상한 후보 수 분포. 도달 범위는 regional()과 동일."""
    mat = S.area_matrix(frame, RULE)
    flags = {"home_final": frame.servId.isin(final.servId[final.final_home]).to_numpy(),
             "home_final_upper": frame.servId.isin(final.servId[final.final_home | final.final_unclear]).to_numpy()}
    local = frame.source.eq("local")
    pairs = frame.loc[local & frame.region_sgg.notna(), ["region_sido", "region_sgg"]].drop_duplicates()
    rows = []
    for sido, sgg in pairs.itertuples(index=False):
        reach = ~local | (frame.region_sido.eq(sido) & (frame.region_sgg.isna() | frame.region_sgg.eq(sgg)))
        for stage, flag in flags.items():
            hit = reach & flag
            rows.append({"region_sido": sido, "region_sgg": sgg, "stage": stage,
                         **{q: int((hit & mat[q]).sum()) for q in S.DEMAND_AREAS}})
    per = pd.DataFrame(rows).melt(["region_sido", "region_sgg", "stage"], var_name="area", value_name="n")
    t = per.groupby(["area", "stage"], sort=False).n.agg(
        n_regions="size", minimum="min", median="median", maximum="max",
        n_regions_zero=lambda s: int(s.eq(0).sum())).reset_index()
    t.insert(1, "area_label", t.area.map(Q21_LABELS))
    return t


def recovery_layers(frame: pd.DataFrame, variants: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """② 직접 회복·일반 돌봄·돌봄 어휘 확장에서 추가되는 엄격 S2 서비스 목록."""
    layers = {"reference_core": "direct_recovery", "reference_care": "general_care", "care_vocabulary": "care_vocabulary"}
    c, seen, parts = conditions(frame), set(), []
    for variant, layer in layers.items():
        v = variants[variant]
        _, scopes, bases, _ = next(spec for spec in S.SCENARIOS if spec[0] == SCENARIO)
        hit = S.in_scenario(v, scopes, bases) & S.area_matrix(v, RULE)["q21_02"]
        new = hit & ~v.servId.isin(seen)
        seen |= set(v.servId[hit])
        parts.append(pd.DataFrame({"servId": v.servId[new], "layer": layer}))
    t = pd.concat(parts, ignore_index=True).merge(
        frame[["servId", "servNm", "source", "region_sido", "q1_human_reviewed", "h_q21_02", "target_condition",
               "apply_barrier", "use_outing", "target_text_raw", "benefit_text_raw"]], on="servId", validate="one_to_one")
    t["support_type"] = t.layer.map({"direct_recovery": "직접 회복"}).fillna("간접 지원")
    text = t.benefit_text_raw.fillna("")
    pattern = F.CARE + "|" + B.PATTERNS["care_vocabulary"]
    t["matched_term"] = text.str.findall(pattern).map(lambda found: "|".join(dict.fromkeys(found)))
    t.loc[t.layer.eq("direct_recovery"), "matched_term"] = ""
    t["stage_reached"] = t.servId.map(_stage(frame, c))
    t[["apply_barrier", "use_outing"]] = t[["apply_barrier", "use_outing"]].fillna(UNKNOWN)
    return t


def examples(frame: pd.DataFrame) -> pd.DataFrame:
    c = conditions(frame)
    stage, raw = _stage(frame, c), frame.set_index("servId")
    in_area = S.area_matrix(frame, RULE).loc[c.youth].set_axis(frame.servId[c.youth])
    rows = []
    for area, sid, note in EXAMPLES:
        r = raw.loc[sid]
        H.require(sid in in_area.index and bool(in_area.at[sid, area]), f"사례 {sid}가 엄격 S2의 {area} 후보가 아님")
        rows.append({"area": area, "area_label": Q21_LABELS[area], "servId": sid, "servNm": r.servNm,
                     "region": r.region_sido if isinstance(r.region_sido, str) else "중앙",
                     "label_source": "human" if r.q1_human_reviewed else "automatic",
                     "apply_barrier": r.apply_barrier if isinstance(r.apply_barrier, str) else UNKNOWN,
                     "use_outing": r.use_outing if isinstance(r.use_outing, str) else UNKNOWN,
                     "stage_reached": stage[sid], "benefit_source_text": r.benefit_text, "apply_source_text": r.apply_text,
                     "reading": note})
    return pd.DataFrame(rows)


def read_verification() -> tuple[pd.DataFrame, pd.DataFrame]:
    """집 조건 후보 원문 검증표와 검수 표본 보정 판정. 사람 검수 원본과 별도 파일."""
    read = lambda name: pd.read_csv(VERIFY / name, dtype=str, encoding="utf-8-sig", keep_default_na=False)
    verify, calibration = read("home_candidates_verification.csv"), read("reviewed_auto_home_calibration.csv")
    for t in (verify, calibration):
        H.require(t.servId.is_unique and t.verdict.isin(VERDICTS).all(), "검증 판정 값 또는 servId 중복 오류")
    for col in ["work_or_business_condition", "user_check"]:
        verify[col] = verify[col].eq("True")
    return verify, calibration


def read_nonhome_verification() -> pd.DataFrame:
    """누락 검증표(자동 라벨이 집으로 보지 않은 청년 대상 후보). 사람 검수 원본과 별도 파일."""
    t = pd.read_csv(VERIFY / "nonhome_candidates_verification.csv", dtype=str, encoding="utf-8-sig", keep_default_na=False)
    H.require(t.servId.is_unique and t.verdict.isin(NONHOME_VERDICTS).all(), "누락 검증 판정 값 또는 servId 중복 오류")
    for col in ["work_or_business_condition", "user_check"]:
        H.require(t[col].isin(["True", "False"]).all(), f"누락 검증표 {col} 값 오류")
        t[col] = t[col].eq("True")
    return t


def nonhome_verification(frame: pd.DataFrame, nonhome: pd.DataFrame) -> pd.DataFrame:
    """청년 대상이면서 집 조건이 아닌 수요 10개 영역 후보 전체에 판정을 붙인 표. 후보 집합이 바뀌면 중단."""
    mat, c = S.area_matrix(frame, RULE), conditions(frame)
    keep = c.youth & ~c.home & mat[S.DEMAND_AREAS].any(axis=1)
    H.require(set(frame.servId[keep]) == set(nonhome.servId), "집 조건 외 청년 대상 후보와 누락 검증표의 servId 집합 불일치")
    t = frame.loc[keep, ["servId", "servNm", "source", "region_sido", "provision_type", "apply_barrier", "use_outing"]].copy()
    t.insert(2, "areas", mat.loc[keep, S.DEMAND_AREAS].apply(lambda row: "|".join(row.index[row]), axis=1))
    t["label_source"] = frame.q1_human_reviewed[keep].map({True: "human", False: "automatic"})
    t = t.merge(nonhome.drop(columns="servNm"), on="servId", validate="one_to_one")
    t.insert(t.columns.get_loc("verdict") + 1, "verdict_label", t.verdict.map(NONHOME_VERDICTS))
    return t


def read_physical_review() -> pd.DataFrame:
    """기존 원문 검증과 사람 판정을 보존하는 서비스 ID 단위 재검토 기록."""
    review = pd.read_csv(VERIFY / "physical_access_review.csv", dtype=str, encoding="utf-8-sig", keep_default_na=False)
    H.require(review.servId.is_unique and review.verdict.isin(ALL_LABELS).all(), "재검토 판정 값 또는 servId 중복 오류")
    H.require(review[REVIEW_FIELDS].ne("").all().all(), "재검토 근거 또는 판정 필드 누락")
    H.require(review.operation_status.replace("", "unverified").isin(OPERATION_LABELS).all(), "운영 상태 값 오류")
    H.require(review.loc[review.operation_status.ne(""), ["operation_note", "operation_source"]].ne("").all().all(), "운영 상태 근거 누락")
    return review


def final_verification(home_checked: pd.DataFrame, nonhome_checked: pd.DataFrame,
                       review: pd.DataFrame | None = None) -> pd.DataFrame:
    """기존 판정에 별도 원문 재검토를 적용한 서비스별 물리적 접근성."""
    home = home_checked.assign(source_set=SOURCE_SETS["home"],
                               review_method=home_checked.label_source.map({"human": "human_reuse"}).fillna("source_review"),
                               official_source="")
    non = nonhome_checked.assign(source_set=SOURCE_SETS["nonhome"])
    t = pd.concat([home, non], ignore_index=True)
    H.require(t.servId.is_unique, "최종 검증표 servId 중복")
    t["previous_verdict"] = t.verdict
    t["previous_verdict_label"] = t.verdict.map(ALL_LABELS)
    detail_fields = REVIEW_FIELDS + OPERATION_FIELDS
    if review is None:
        for col in detail_fields:
            t[col] = ""
    else:
        expected = review.servId.map(t.set_index("servId").previous_verdict)
        H.require(expected.eq(review.previous_verdict).all(), "기존 판정과 재검토 기록의 servId 또는 판정 불일치")
        H.require(t.loc[t.user_check, "servId"].isin(review.servId).all(), "경계 사례 재검토 누락")
        t = t.merge(review[["servId", "verdict", *detail_fields]].rename(columns={"verdict": "review_verdict"}),
                    on="servId", how="left", validate="one_to_one")
        t["verdict"] = t.pop("review_verdict").fillna(t.verdict)
        t[detail_fields] = t[detail_fields].fillna("")
    t["operation_status"] = t.operation_status.replace("", "unverified")
    t["operation_status_label"] = t.operation_status.map(OPERATION_LABELS)
    t["verdict_label"] = t.verdict.map(ALL_LABELS)
    t["verdict_changed"] = t.verdict.ne(t.previous_verdict)
    t["final_home"] = t.verdict.isin(FINAL_HOME)
    t["final_unclear"] = t.verdict.isin(FINAL_UNCLEAR)
    t["strict_status"] = "not_home"
    t.loc[t.final_home, "strict_status"] = "home"
    t.loc[t.final_unclear, "strict_status"] = "unclear"
    t.loc[t.verdict.eq("not_assessed"), "strict_status"] = "not_assessed"
    t["extension_addition"] = t.verdict.eq("extension_apply1_home")
    t["final_extension"] = t.final_home | t.extension_addition
    cols = ["servId", "servNm", "areas", "label_source", "source_set", "previous_verdict", "previous_verdict_label",
            "verdict", "verdict_label", "verdict_changed", "strict_status", "final_home", "final_unclear",
            "extension_addition", "final_extension", "work_or_business_condition", "user_check", "review_method",
            "evidence_text", "official_source", "verify_note", "detail_url", *REVIEW_FIELDS,
            *OPERATION_FIELDS, "operation_status_label"]
    return t[cols]


def user_check_table(final: pd.DataFrame) -> pd.DataFrame:
    """기존 경계 사례 19건의 이전 판정, 재검토 판정과 근거."""
    cols = ["servId", "servNm", "areas", "source_set", "previous_verdict_label", "verdict_label", "strict_status",
            "verdict_changed", *REVIEW_FIELDS, "detail_url", *OPERATION_FIELDS, "operation_status_label"]
    return final.loc[final.user_check, cols].reset_index(drop=True)


def home_verification(frame: pd.DataFrame, verify: pd.DataFrame) -> pd.DataFrame:
    """수요 10개 영역 집 조건 후보 전체에 원문 검증 판정을 붙인 표. 후보 집합이 바뀌면 중단."""
    mat, c = S.area_matrix(frame, RULE), conditions(frame)
    keep = c.home & mat[S.DEMAND_AREAS].any(axis=1)
    H.require(set(frame.servId[keep]) == set(verify.servId), "집 조건 후보와 원문 검증표의 servId 집합 불일치")
    t = frame.loc[keep, ["servId", "servNm", "source", "region_sido", "provision_type", "apply_barrier", "use_outing"]].copy()
    t.insert(2, "areas", mat.loc[keep, S.DEMAND_AREAS].apply(lambda row: "|".join(row.index[row]), axis=1))
    t["label_source"] = frame.q1_human_reviewed[keep].map({True: "human", False: "automatic"})
    t = t.merge(verify.drop(columns="servNm"), on="servId", validate="one_to_one")
    t.insert(t.columns.get_loc("verdict") + 1, "verdict_label", t.verdict.map(VERDICTS))
    # 사람 검수 서비스는 검증 판정과 관계없이 사람 판정을 그대로 둠. 판정이 다르면 중단.
    H.require(t.verdict[t.label_source.eq("human")].eq("home_confirmed").all(), "사람 검수 서비스의 검증 판정이 사람 판정과 다름")
    t["human_label_retained"] = t.label_source.eq("human")
    return t


def verified_funnel(frame: pd.DataFrame, verification: pd.DataFrame, funnel_table: pd.DataFrame,
                    nonhome: pd.DataFrame | None = None, final: pd.DataFrame | None = None) -> pd.DataFrame:
    """영역별 집 조건 후보의 검증 전후 수와 비율. 분모는 영역별 청년 대상 후보.

    기존 두 검증표의 상태 수를 보존하고 최종 물리적 접근성은 재검토 반영 판정에서 집계함.
    """
    mat = S.area_matrix(frame, RULE).set_axis(frame.servId)
    v = verification.set_index("servId")
    hit = mat.loc[v.index]
    groups = {q: hit[q] for q in S.DEMAND_AREAS} | {"any_demand_area": hit[S.DEMAND_AREAS].any(axis=1)}
    base = funnel_table.set_index("area")
    if nonhome is not None:
        non = nonhome.set_index("servId")
        final = final_verification(verification, nonhome) if final is None else final
        adjudicated = final.set_index("servId")
        nm, fm = mat.loc[non.index], mat.loc[adjudicated.index]
    rows = []
    for area, h in groups.items():
        sub = v[h]
        row = {"area": area, "area_label": base.at[area, "area_label"], "rate_very_needed_4": base.at[area, "rate_very_needed_4"],
               "demand_rank_4": base.at[area, "demand_rank_4"], "n_youth": int(base.at[area, "n_youth"]),
               "n_home_auto": len(sub), "n_home_reviewed": int(sub.label_source.eq("human").sum())}
        row.update({f"n_{k}": int(sub.verdict.eq(k).sum()) for k in VERDICTS})
        if nonhome is not None:
            nsub = non[nm[S.DEMAND_AREAS].any(axis=1) if area == "any_demand_area" else nm[area]]
            fsub = adjudicated[fm[S.DEMAND_AREAS].any(axis=1) if area == "any_demand_area" else fm[area]]
            row.update({
                "n_nonhome": len(nsub), "n_missed_home": int(nsub.verdict.eq("missed_home").sum()),
                "n_unclear_nonhome": int(nsub.verdict.eq("unclear").sum()),
                "n_not_home_nonhome": int(nsub.verdict.eq("not_home").sum()),
                "n_extension_apply1_home": int(nsub.verdict.eq("extension_apply1_home").sum()),
                "n_home_final": int(fsub.final_home.sum()),
                "n_home_final_from_auto_home": int((fsub.final_home & fsub.source_set.eq(SOURCE_SETS["home"])).sum()),
                "n_home_final_from_auto_not_home": int((fsub.final_home & fsub.source_set.eq(SOURCE_SETS["nonhome"])).sum()),
                "n_not_home_final": int(fsub.strict_status.eq("not_home").sum()),
                "n_unclear_final": int(fsub.final_unclear.sum()),
                "n_not_assessed_final": int(fsub.strict_status.eq("not_assessed").sum()),
                "n_new_support_discontinued": int(fsub.operation_status.eq("new_support_discontinued").sum()),
                "n_extension_addition": int(fsub.extension_addition.sum()),
                "n_extension_final": int(fsub.final_extension.sum()),
            })
        rows.append(row)
    t = pd.DataFrame(rows)
    H.require((t.n_home_auto == base.loc[t.area, "n_home"].to_numpy()).all(), "검증표 영역별 건수와 access_funnel_by_area 불일치")
    t["n_not_home_auto_review"] = t.n_cash_outing_activity + t.n_auto_label_error
    t["n_home_verified"] = t.n_home_confirmed
    t["n_home_verified_upper"] = t.n_home_confirmed + t.n_unclear
    denom = t.n_youth.where(t.n_youth.gt(0))
    for col in ["home_auto", "home_verified", "home_verified_upper"]:
        t[f"share_{col}"] = t[f"n_{col}"] / denom
    t["small_denominator"] = t.n_youth.lt(SMALL)
    areas = t.area.ne("any_demand_area")
    for col in ["home_auto", "home_verified"]:
        t.loc[areas, f"rank_{col}"] = _rank(t.loc[areas, f"n_{col}"])
    if nonhome is not None:
        H.require((t.n_youth == t.n_home_auto + t.n_nonhome).all(), "영역별 청년 대상 수와 검증 대상 수(집 + 집 외) 불일치")
        H.require((t.n_home_auto == t.n_home_confirmed + t.n_not_home_auto_review + t.n_unclear).all(),
                  "기존 집 후보 판정 건수 불일치")
        H.require((t.n_nonhome == t.n_missed_home + t.n_not_home_nonhome + t.n_unclear_nonhome + t.n_extension_apply1_home).all(),
                  "기존 집 외 후보 판정 건수 불일치")
        H.require((t.n_youth == t.n_home_final + t.n_not_home_final + t.n_unclear_final + t.n_not_assessed_final).all(),
                  "최종 엄격 기준 판정 건수 불일치")
        t["n_home_final_upper"] = t.n_home_final + t.n_unclear_final
        for col in ["home_final", "home_final_upper", "extension_final"]:
            t[f"share_{col}"] = t[f"n_{col}"] / denom
        t.loc[areas, "rank_home_final"] = _rank(t.loc[areas, "n_home_final"])
    return t


def verification_metrics(m: pd.DataFrame, gold: pd.DataFrame, calibration: pd.DataFrame) -> pd.DataFrame:
    """검수 236건에서 집 조건 분류의 정밀도·재현율. 원문 검증 기준 적용 전후.

    - 적용 후 값은 사람 판정을 본 뒤 맞춘 보정 결과이므로 독립 검증이 아님.
    - 재현율은 검수 표본 안의 값. 검수 표본은 무작위 추출이 아니며 전체 데이터의 재현율이 아님.
    """
    auto = m.set_index("servId").loc[gold.servId]
    auto_yes = (auto.apply_barrier.fillna(UNKNOWN).eq(LOW) & auto.use_outing.fillna(UNKNOWN).eq(LOW)).to_numpy()
    human_yes = (gold[H.APPLY].eq(LOW) & gold[H.OUTING].eq(LOW)).to_numpy()
    cal = calibration.set_index("servId").verdict
    H.require(set(cal.index) == set(gold.servId[auto_yes]), "보정 판정표가 검수 표본의 자동 집 조건 충족 건과 다름")
    kept = gold.servId.map(cal).eq("home_confirmed").to_numpy()
    rows = []
    for rule, pred in {"automatic": auto_yes, "automatic_plus_source_rule": auto_yes & kept}.items():
        tp = int((pred & human_yes).sum())
        rows.append({"rule": rule, "n_reviewed": len(gold), "n_predicted_home": int(pred.sum()),
                     "n_human_home": int(human_yes.sum()), "n_true_positive": tp,
                     "precision": tp / pred.sum(), "recall_within_reviewed": tp / human_yes.sum(),
                     "independent": rule == "automatic"})
    return pd.DataFrame(rows)


def analyze(m: pd.DataFrame, demand: pd.DataFrame, layer: pd.DataFrame, gold: pd.DataFrame,
            variants: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    frame = layer_frame(m, layer)
    verify, calibration = read_verification()
    fn = funnel(frame, demand)
    checked = home_verification(frame, verify)
    nonhome = nonhome_verification(frame, read_nonhome_verification())
    final = final_verification(checked, nonhome, read_physical_review())
    # 기존 자격 자동 태그는 참고용으로 유지하고 원문 재검토에 기록한 자격 요건과 함께 제공.
    final["target_condition"] = final.servId.map(frame.set_index("servId").target_condition).fillna("")
    regional_all = pd.concat([regional(frame), regional_final(frame, final)], ignore_index=True)
    return {"access_funnel_by_area": fn, "access_candidates": candidates(frame),
            "access_label_reliability": label_reliability(m, gold), "access_regional": regional_all,
            "access_recovery_layers": recovery_layers(frame, variants), "access_examples": examples(frame),
            "access_home_verification": checked, "access_funnel_verified": verified_funnel(frame, checked, fn, nonhome, final),
            "access_verification_metrics": verification_metrics(m, gold, calibration),
            "access_final_verification": final, "access_user_check": user_check_table(final)}


def save(frames: dict[str, pd.DataFrame]) -> None:
    for name, frame in frames.items():
        frame.to_csv(OUT / "tables" / f"{name}.csv", **H.CSV)
