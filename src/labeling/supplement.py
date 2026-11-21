"""STEP 02 보충 표본 (2026-11-04): 청년 범위 youth_scope(A/B/U/C) 판정을 평가하기 위한 표본.

기존 표본(labeling_key.csv 352건, 1차 시트 236건, 이중 판정 74건)은 그대로 둔다. 기존 S 층은 '청년 근거 유무'로만
나뉘어 U(근거 없음)·C(연령 제외)·B의 근거 종류별 판정 정확도를 따로 추정하기 어렵다.

설계
  R  youth_scope × youth_scope_basis 층화 무작위(근거 종류별 정확도를 따로 추정). 층마다 같은 수, 가중치 = N_h / n_h.
     가중 추정의 대상 모집단은 '전체 5,288건 − 기존 채점 키 − 규칙 개발 사례'다(전체로 일반화하려면 기존 키와 합쳐 보정).
     v3 규칙을 고칠 때 본 사례(독립 검토의 예시 servId)는 V3_REVIEW_IDS에 기록해 모집단에서 뺀다.
     층은 이 파일을 처음 만든 시점의 마스터 값으로 고정하며, 기존 표본과 규칙 개발에 쓴 사례는 모집단에서 뺀다.
  규칙 개발 사례 = STEP 00 경계 사례 검토(outputs/00_data_preparation/review/boundary_cases_v2*.csv) +
                    review_overrides.TO_N. 이 사례는 평가 지표에서 별도 집계한다(독립 평가 아님).
blind 시트에는 규칙 값을 넣지 않는다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..collect._common import ROOT
from ..preprocess.review_overrides import TO_N
from .sampling import TEXT_COLS

SEED_R, SEED_R_DOUBLE = 20261107, 20261108
N_PER_STRATUM_R = 12
# v3 규칙 수정 때 검토한 서비스(규칙 개발 사례로 취급)
V3_REVIEW_IDS = {"WLF00003281", "WLF00001107", "WLF00003283", "WLF00001751", "WLF00003199", "WLF00006267", "WLF00006269",
                 "WLF00003245", "WLF00003945", "WLF00005426", "WLF00004691", "WLF00006302", "WLF00005667", "WLF00002666",
                 "WLF00006799", "WLF00000451", "WLF00002583", "WLF00000581", "WLF00000211", "WLF00004830", "WLF00005476",
                 "WLF00003666", "WLF00002896", "WLF00003719", "WLF00000074", "WLF00002275", "WLF00000100", "WLF00000888",
                 "WLF00006256", "WLF00001619", "WLF00002649", "WLF00004032", "WLF00005900", "WLF00001068", "WLF00003265",
                 "WLF00002138"}
DOUBLE_LABEL_SHARE_R = 0.2
TBL = ROOT / "outputs" / "02_labeling" / "tables"
REVIEW = ROOT / "outputs" / "00_data_preparation" / "review"
HUMAN_COLS_R = ["h_youth_scope", "h_youth_scope_evidence", "h_target_condition", "h_note", "labeler", "labeled_at"]
KEY_RULE_COLS_R = ["source", "youth_scope", "youth_scope_basis", "youth_scope_review", "age_min", "age_max",
                   "youth_tag", "youth_text", "target_condition"]


def rule_dev_ids() -> set[str]:
    """규칙을 만들거나 고칠 때 이미 본 서비스 ID."""
    ids = set(TO_N) | V3_REVIEW_IDS
    for f in ("boundary_cases_v2.csv", "boundary_cases_v2_review.csv"):
        p = REVIEW / f
        if p.exists():
            ids |= set(pd.read_csv(p, dtype=str, encoding="utf-8-sig")["servId"])
    return ids


def overlap_flags(key: pd.DataFrame) -> pd.DataFrame:
    """기존 채점 키의 각 서비스가 규칙 개발 사례와 겹치는지 표시(키 파일은 수정하지 않는다)."""
    dev = rule_dev_ids()
    return pd.DataFrame({"servId": key["servId"], "in_rule_dev": key["servId"].isin(dev)})


def coverage(m: pd.DataFrame, key: pd.DataFrame) -> pd.DataFrame:
    """현재 youth_scope 기준으로 기존 표본에 각 범주가 몇 건 들어 있는지."""
    k = key[["servId", "in_S", "in_P", "in_J"]].merge(m[["servId", "youth_scope", "youth_scope_basis"]], on="servId")
    for c in ("in_S", "in_P", "in_J"):
        k[c] = k[c].astype(str).eq("True")
    t = k.groupby(["youth_scope", "youth_scope_basis"]).agg(n_key=("servId", "count"), n_S=("in_S", "sum"),
                                                            n_P=("in_P", "sum"), n_J=("in_J", "sum")).reset_index()
    pop = m.groupby(["youth_scope", "youth_scope_basis"]).size().rename("N_population").reset_index()
    return pop.merge(t, on=["youth_scope", "youth_scope_basis"], how="left").fillna(0)


def draw_r(m: pd.DataFrame, exclude: set[str]) -> pd.DataFrame:
    frame = m[~m["servId"].isin(exclude)]
    st = frame["youth_scope"] + "|" + frame["youth_scope_basis"]
    rows = []
    for h, idx in st.groupby(st).groups.items():
        n = min(N_PER_STRATUM_R, len(idx))
        pick = frame.loc[idx].sample(n, random_state=SEED_R)
        rows.append(pd.DataFrame({"servId": pick["servId"], "r_stratum": h, "r_N_h": len(idx), "r_n_h": n,
                                  "r_weight": len(idx) / n}))
    key = pd.concat(rows, ignore_index=True).sort_values("servId").reset_index(drop=True)
    rng = np.random.default_rng(SEED_R_DOUBLE)
    key["double_label"] = rng.random(len(key)) < DOUBLE_LABEL_SHARE_R
    key["in_rule_dev"] = key["servId"].isin(rule_dev_ids())      # 규칙을 고칠 때 읽은 사례(정확도 추정에서 따로 본다)
    return key.merge(m[["servId", *KEY_RULE_COLS_R]], on="servId", how="left")


def make_sheet_r(m: pd.DataFrame, key: pd.DataFrame) -> pd.DataFrame:
    sheet = key[["servId", "source", "double_label"]].merge(m[["servId", *TEXT_COLS, "life_tags"]], on="servId", how="left")
    for c in HUMAN_COLS_R:
        sheet[c] = ""
    sheet.insert(0, "round", "round1b_youth_scope")
    return sheet.sample(frac=1, random_state=SEED_R_DOUBLE).reset_index(drop=True)


def build(m: pd.DataFrame, key: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """보충 표본·시트·겹침 표시·포함 현황. 이미 저장된 보충 표본이 있으면 다시 뽑지 않고 그대로 쓴다."""
    path_key = TBL / "labeling_key_round1b_youth_scope.csv"
    if path_key.exists():
        key_r = pd.read_csv(path_key, dtype=str, encoding="utf-8-sig", keep_default_na=False)
    else:
        key_r = draw_r(m, set(key["servId"]) | rule_dev_ids())
    sheet_path = TBL / "labeling_sheet_round1b_youth_scope.csv"
    if sheet_path.exists():
        sheet = pd.read_csv(sheet_path, dtype=str, encoding="utf-8-sig", keep_default_na=False)
    else:
        sheet = make_sheet_r(m, key_r)
    return {"labeling_key_round1b_youth_scope": key_r, "labeling_sheet_round1b_youth_scope": sheet,
            "labeling_key_rule_dev_overlap": overlap_flags(key), "labeling_youth_scope_coverage": coverage(m, key)}


def main() -> None:
    m = pd.read_csv(ROOT / "data/processed/services_master.csv", dtype=str, encoding="utf-8-sig",
                    keep_default_na=False, na_values=[""])
    key = pd.read_csv(TBL / "labeling_key.csv", dtype=str, encoding="utf-8-sig", keep_default_na=False)
    for name, t in {**build(m, key), **build_disagreement(m, key), **build_econ(m, key)}.items():
        path = TBL / f"{name}.csv"
        if name.startswith("labeling_sheet") and path.exists():
            continue                      # 작성 중일 수 있는 시트는 덮어쓰지 않는다
        t.to_csv(path, index=False, encoding="utf-8-sig")
        print(f"{name}.csv {len(t)}행")



# ------------------------------------------------------------------ 규칙 간 불일치 표본 (2026-11-05)
# services_master와 팀 라벨 파일(data/raw/team_labels, 미검수)이 외출 관련 판정에서 갈리는 서비스.
# 두 파일 모두 사람 검수 전이므로 어느 쪽도 정답으로 보지 않는다. 사람 판정으로 두 규칙을 함께 평가한다.
TEAM_LABELS = ROOT / "data" / "raw" / "team_labels" / "youth_service_labels_v1.csv"
TEAM_LABELS_SHA256 = "bb3fbaab98285bc2be9127fe3e061e5eb3afaa66879f23648d24afb9b57d7a91"
TEAM_APPLY_OUTING = {"외출 없는 경로 확인": "0", "외출 필수": "2", "확인 필요": "N", "N": "N"}
SEED_C, SEED_C_DOUBLE = 20261109, 20261110
N_PER_STRATUM_C = 12          # 한쪽만 N인 층의 표본 수. 해석이 다른 층은 전수
HUMAN_COLS_C = ["h_apply_barrier", "h_apply_visit", "h_apply_evidence", "h_use_outing", "h_use_outing_evidence",
                "h_note", "labeler", "labeled_at"]
# h_apply_visit: 방문 필수 / 방문 경로만 안내 / 비방문 경로 있음 / 신청 정보 없음


def load_team_labels() -> pd.DataFrame:
    import hashlib
    if hashlib.sha256(TEAM_LABELS.read_bytes()).hexdigest() != TEAM_LABELS_SHA256:
        raise ValueError("팀 라벨 파일이 바뀌었습니다. 새 판은 _v2로 추가하고 이 상수를 함께 고치세요.")
    return pd.read_csv(TEAM_LABELS, dtype=str, encoding="utf-8-sig", keep_default_na=False)


def disagreement_causes(m: pd.DataFrame, team: pd.DataFrame) -> pd.DataFrame:
    """서비스별 불일치 원인. 외출필요도가 같으면 '일치'. 다르면 신청 외출·이용 외출 중 어느 판정이 갈렸는지로 나눈다.
    '한쪽만 N'은 오분류가 아니라 판단 보류일 수 있으므로 별도 원인으로 둔다."""
    t = team.set_index("servId")
    r = m.set_index("servId").loc[t.index]
    d = pd.DataFrame({
        "team_apply_outing": t["apply_outing"].map(TEAM_APPLY_OUTING), "rule_apply_outing": r["apply_outing"],
        "team_use_outing": t["use_outing"], "rule_use_outing": r["use_outing"],
        "team_outing_need": t["outing_need"], "rule_outing_need": r["outing_need"],
        "team_apply_barrier": t["apply_barrier"], "rule_apply_barrier": r["apply_barrier"],
        "rule_apply_outing_basis": r["apply_outing_basis"],
    })

    def cause(x) -> str:
        if x.team_outing_need == x.rule_outing_need:
            return "agree"
        if "N" in (x.team_outing_need, x.rule_outing_need):
            side = "team_N" if x.team_outing_need == "N" else "rule_N"
            a_n = (x.team_apply_outing if side == "team_N" else x.rule_apply_outing) == "N"
            return f"one_side_N|{side}|{'apply' if a_n else 'use'}"
        a_diff = x.team_apply_outing != x.rule_apply_outing
        u_diff = x.team_use_outing != x.rule_use_outing
        return "interp|" + ("both" if a_diff and u_diff else "apply" if a_diff else "use")
    d["cause"] = d.apply(cause, axis=1)
    return d.reset_index()


def draw_c(causes: pd.DataFrame, exclude: set[str]) -> pd.DataFrame:
    """해석이 다른 층(interp|*)은 전수, 한쪽만 N인 층은 층마다 N_PER_STRATUM_C건. exclude는 이미 외출 판정 칸이 있는 표본."""
    frame = causes[causes["cause"].ne("agree") & ~causes["servId"].isin(exclude)]
    rows = []
    for h, g in frame.groupby("cause"):
        n = len(g) if h.startswith("interp") else min(N_PER_STRATUM_C, len(g))
        pick = g.sample(n, random_state=SEED_C)
        rows.append(pick.assign(c_stratum=h, c_N_h=len(g), c_n_h=n, c_weight=len(g) / n))
    key = pd.concat(rows, ignore_index=True).sort_values("servId").reset_index(drop=True)
    rng = np.random.default_rng(SEED_C_DOUBLE)
    key["double_label"] = rng.random(len(key)) < DOUBLE_LABEL_SHARE_R
    key["in_rule_dev"] = key["servId"].isin(rule_dev_ids())      # 규칙을 고칠 때 읽은 사례(정확도 추정에서 따로 본다)
    return key


def make_sheet_c(m: pd.DataFrame, key: pd.DataFrame) -> pd.DataFrame:
    sheet = key[["servId", "double_label"]].merge(m[["servId", "source", *TEXT_COLS]], on="servId", how="left")
    for c in HUMAN_COLS_C:
        sheet[c] = ""
    sheet.insert(0, "round", "round1c_rule_disagreement")
    return sheet.sample(frac=1, random_state=SEED_C_DOUBLE).reset_index(drop=True)


def build_disagreement(m: pd.DataFrame, key: pd.DataFrame) -> dict[str, pd.DataFrame]:
    causes = disagreement_causes(m, load_team_labels())
    path_key = TBL / "labeling_key_round1c_rule_disagreement.csv"
    if path_key.exists():
        key_c = pd.read_csv(path_key, dtype=str, encoding="utf-8-sig", keep_default_na=False)
    else:
        # 1차 시트(S∪P)는 이미 h_apply_barrier·h_use_outing 판정 칸이 있으므로 빼고, 그 판정을 함께 쓴다.
        # 보충 R(청년 범위) 시트에는 외출 칸이 없어 빼지 않는다. 규칙 개발 사례는 빼지 않고 표시만 한다(전수 원칙).
        r1 = key[key[["in_S", "in_P"]].astype(str).eq("True").any(axis=1)]
        key_c = draw_c(causes, set(r1["servId"]))
        key_c["in_rule_dev"] = key_c["servId"].isin(rule_dev_ids())
    sheet_path = TBL / "labeling_sheet_round1c_rule_disagreement.csv"
    sheet = (pd.read_csv(sheet_path, dtype=str, encoding="utf-8-sig", keep_default_na=False) if sheet_path.exists()
             else make_sheet_c(m, key_c))
    summary = causes.groupby("cause").size().rename("n").reset_index()
    return {"label_disagreement_causes": causes, "label_disagreement_summary": summary,
            "labeling_key_round1c_rule_disagreement": key_c, "labeling_sheet_round1c_rule_disagreement": sheet}


# ------------------------------------------------------------------ ⑩ 경제적 지원 검수 표본 (2026-11-05)
# q21_10_type 층화. v3에서 ⑩이었던 서비스(원문 단어 1,672건·제공유형 1,931건)와 v4 유형을 함께 덮는다.
# 1차 시트(S∪P)에는 이미 h_q21_10 칸이 있으므로 그 서비스는 뺀다. 가중치의 모집단은 '⑩ 신호가 있는 서비스 − S∪P'다.
# tier 빈칸(⑩ 신호 없음)은 표본 밖이라 재현율은 추정하지 않는다. h_q21_10 판정 기준은 docs/services_master_spec.md 5-2.
SEED_E, SEED_E_DOUBLE = 20261112, 20261113
N_PER_STRATUM_E = 10
HUMAN_COLS_E = ["h_q21_10", "h_q21_10_type", "h_other_area", "h_evidence", "h_note", "labeler", "labeled_at"]
# h_q21_10: 예/아니오/불명, h_q21_10_type: 생계·생활비 / 필수 생활비 경감 / 자산형성 / 일시금·축하금 / 보상·예우 /
#           용도 미상 현금 / 다른 용도(영역 기재) / 기관·사업자 대상 / 해당 없음


def draw_e(m: pd.DataFrame, exclude: set[str]) -> pd.DataFrame:
    frame = m[m["q21_10_type"].ne("none") & m["q21_10_type"].notna() & ~m["servId"].isin(exclude)]
    # 용도가 다른 지급(earmarked)은 용도 종류(⑪·⑦·⑧·의료·서비스·문화)로 나눈다. 오류가 특정 용도에 몰려도 드러나게
    sub = frame["q21_10_evidence"].fillna("").str.split(":").str[0]
    st = (frame["q21_10_type"] + "|" + frame["q21_10_basis_v3"].replace("", "v3_none").fillna("v3_none")).where(
        frame["q21_10_type"].ne("earmarked"), "earmarked|" + sub)
    rows = []
    for h, idx in st.groupby(st).groups.items():
        n = min(N_PER_STRATUM_E if not h.endswith("v3_none") else 5, len(idx))
        pick = frame.loc[idx].sample(n, random_state=SEED_E)
        rows.append(pd.DataFrame({"servId": pick["servId"], "e_stratum": h, "e_N_h": len(idx), "e_n_h": n,
                                  "e_weight": len(idx) / n}))
    key = pd.concat(rows, ignore_index=True).sort_values("servId").reset_index(drop=True)
    rng = np.random.default_rng(SEED_E_DOUBLE)
    key["double_label"] = rng.random(len(key)) < DOUBLE_LABEL_SHARE_R
    key["in_rule_dev"] = key["servId"].isin(rule_dev_ids())      # 규칙을 고칠 때 읽은 사례(정확도 추정에서 따로 본다)
    return key.merge(m[["servId", "need_area_q21_v3", "need_area_q21", "q21_10_type", "q21_10_tier", "q21_10_evidence"]],
                     on="servId", how="left")


def build_econ(m: pd.DataFrame, key: pd.DataFrame) -> dict[str, pd.DataFrame]:
    path_key = TBL / "labeling_key_round1d_econ.csv"
    if path_key.exists():
        key_e = pd.read_csv(path_key, dtype=str, encoding="utf-8-sig", keep_default_na=False)
    else:
        r1 = key[key[["in_S", "in_P"]].astype(str).eq("True").any(axis=1)]
        key_e = draw_e(m, set(r1["servId"]))
    sheet_path = TBL / "labeling_sheet_round1d_econ.csv"
    if sheet_path.exists():
        sheet = pd.read_csv(sheet_path, dtype=str, encoding="utf-8-sig", keep_default_na=False)
    else:
        sheet = key_e[["servId", "double_label"]].merge(m[["servId", "source", *TEXT_COLS]], on="servId", how="left")
        for c in HUMAN_COLS_E:
            sheet[c] = ""
        sheet.insert(0, "round", "round1d_econ")
        sheet = sheet.sample(frac=1, random_state=SEED_E_DOUBLE).reset_index(drop=True)
    return {"labeling_key_round1d_econ": key_e, "labeling_sheet_round1d_econ": sheet}


if __name__ == "__main__":
    main()
