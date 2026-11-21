"""STEP 02 사람 검수용 표본 설계.

세 설계를 각각 고정 seed로 뽑는다.
  S  Q1 층화 무작위 표본: 출처 × 청년 근거(any) × 영역 후보 유무 8개 층. 층마다 같은 수를 뽑고 가중치 = N_h / n_h.
     영역 후보가 없는 층(need_area_q21 = N)과 청년 근거가 없는 층을 포함하므로 규칙 누락(재현율)도 추정할 수 있다.
  P  영역별 정밀도 보충: need_area_q21 후보가 있는 서비스에서 영역마다 최대 k건.
  J  job_relevance 원인 구분: job_relevance_signal 층(keyword_present / text_no_keyword / short_text)과
     규칙 값(1, 2) 층에서 각각 뽑는다.
라벨링은 두 차례로 나눈다. 1차(Q1 영역) = S ∪ P, 2차(job_relevance) = J.
정답셋 후보(집중 검수)는 별도 목록으로 유지하며 여기서는 겹침 여부만 표시한다.

라벨링 시트는 규칙 후보 값을 보여주지 않는다(blind). 채점용 키 파일에만 규칙 값을 둔다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SEED_S, SEED_P, SEED_J, SEED_DOUBLE = 20261103, 20261104, 20261105, 20261106
N_PER_STRATUM_S = 20
K_PER_AREA_P = 8
N_PER_STRATUM_J = {"keyword_present": 30, "text_no_keyword": 30, "short_text": 20, "rule_2": 20, "rule_1": 20}
DOUBLE_LABEL_SHARE = 0.2

Q21 = [f"q21_{i:02d}" for i in range(1, 12)]
TEXT_COLS = ["servNm", "org", "region_sido", "region_sgg", "provision_type", "detail_url",
             "summary_text", "target_text", "criteria_text", "benefit_text", "apply_text"]
COMMON_HUMAN_COLS = ["h_note", "labeler", "labeled_at"]
ROUND_HUMAN_COLS = {
    "round1_q1": [f"h_{q}" for q in Q21] + ["h_q21_evidence", "h_youth_target", "h_youth_evidence",
                                            "h_apply_barrier", "h_apply_evidence", "h_use_outing", "h_use_outing_evidence"],
    "round2_job": ["h_job_relevance", "h_job_reason", "h_job_evidence"],
}
ROUND_MEMBERS = {"round1_q1": ("in_S", "in_P"), "round2_job": ("in_J",)}
KEY_RULE_COLS = ["source", "need_area_q21", "need_area_q21_evidence", "elig_youth", "youth_tag", "youth_text", "youth_age",
                 "job_relevance", "job_relevance_evidence", "job_relevance_signal",
                 "apply_barrier", "apply_interview", "use_outing", "outing_need", "outing_need_min", "use_barrier", "outing_burden"]


def _youth_any(m: pd.DataFrame) -> pd.Series:
    return m["youth_tag"].astype(bool) | m["youth_text"].astype(bool) | m["youth_age"].eq(True)


def strata_q1(m: pd.DataFrame) -> pd.Series:
    youth = np.where(_youth_any(m), "youth", "noyouth")
    area = np.where(m["need_area_q21"].ne("N"), "areahit", "noarea")
    return pd.Series([f"{s}|{y}|{a}" for s, y, a in zip(m["source"], youth, area)], index=m.index)


def draw_s(m: pd.DataFrame) -> pd.DataFrame:
    st = strata_q1(m)
    rows = []
    for h, idx in st.groupby(st).groups.items():
        n = min(N_PER_STRATUM_S, len(idx))
        pick = m.loc[idx].sample(n, random_state=SEED_S)
        rows.append(pd.DataFrame({"servId": pick["servId"], "s_stratum": h, "s_N_h": len(idx), "s_n_h": n,
                                  "s_weight": len(idx) / n}))
    return pd.concat(rows, ignore_index=True)


def draw_p(m: pd.DataFrame) -> pd.DataFrame:
    """영역당 1행 (한 서비스가 여러 영역에 뽑힐 수 있음)."""
    rows = []
    for q in Q21:
        hit = m[m["need_area_q21"].str.split("|").map(lambda v: q in v)]
        n = min(K_PER_AREA_P, len(hit))
        if n:
            pick = hit.sample(n, random_state=SEED_P)
            rows.append(pd.DataFrame({"servId": pick["servId"], "p_area": q, "p_N_area": len(hit), "p_n_area": n}))
    return pd.concat(rows, ignore_index=True)


def job_stratum(m: pd.DataFrame) -> pd.Series:
    sig = m["job_relevance_signal"].copy()
    sig[m["job_relevance"].eq("2")] = "rule_2"
    sig[m["job_relevance"].eq("1")] = "rule_1"
    return sig


def draw_j(m: pd.DataFrame) -> pd.DataFrame:
    st = job_stratum(m)
    rows = []
    for h, n_target in N_PER_STRATUM_J.items():
        idx = st.index[st.eq(h)]
        n = min(n_target, len(idx))
        pick = m.loc[idx].sample(n, random_state=SEED_J)
        rows.append(pd.DataFrame({"servId": pick["servId"], "j_stratum": h, "j_N_h": len(idx), "j_n_h": n,
                                  "j_weight": len(idx) / n}))
    return pd.concat(rows, ignore_index=True)


def build_design(m: pd.DataFrame, goldset_ids: set[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(설계 요약, 채점 키) 반환. 키는 servId 1행이며 설계 소속·가중치·규칙 값을 담는다."""
    s, p_rows, j = draw_s(m), draw_p(m), draw_j(m)
    p = p_rows.groupby("servId", as_index=False).agg(p_area=("p_area", "|".join))
    ids = pd.Series(sorted(set(s.servId) | set(p.servId) | set(j.servId)), name="servId")
    key = ids.to_frame().merge(s, on="servId", how="left").merge(p, on="servId", how="left").merge(j, on="servId", how="left")
    key["in_S"], key["in_P"], key["in_J"] = key.s_stratum.notna(), key.p_area.notna(), key.j_stratum.notna()
    key["in_goldset_candidates"] = key.servId.isin(goldset_ids)
    rng = np.random.default_rng(SEED_DOUBLE)
    key["double_label"] = rng.random(len(key)) < DOUBLE_LABEL_SHARE
    key = key.merge(m[["servId", *KEY_RULE_COLS]], on="servId", how="left")

    summary = pd.concat([
        s.groupby("s_stratum").agg(N_h=("s_N_h", "first"), n_h=("s_n_h", "first"), weight=("s_weight", "first"))
         .reset_index().rename(columns={"s_stratum": "stratum"}).assign(design="S Q1 층화 무작위"),
        p_rows.rename(columns={"p_area": "stratum", "p_N_area": "N_h", "p_n_area": "n_h"})[["stratum", "N_h", "n_h"]]
         .drop_duplicates().assign(weight=lambda x: x.N_h / x.n_h, design="P 영역별 정밀도 보충"),
        j.groupby("j_stratum").agg(N_h=("j_N_h", "first"), n_h=("j_n_h", "first"), weight=("j_weight", "first"))
         .reset_index().rename(columns={"j_stratum": "stratum"}).assign(design="J job_relevance 원인 구분"),
    ], ignore_index=True)[["design", "stratum", "N_h", "n_h", "weight"]]
    return summary, key


def make_sheet(m: pd.DataFrame, key: pd.DataFrame, round_name: str) -> pd.DataFrame:
    """차수별 blind 라벨링 시트. 규칙 값은 넣지 않고, 설계 순서가 드러나지 않게 섞는다."""
    members = key[key[list(ROUND_MEMBERS[round_name])].any(axis=1)]
    sheet = members[["servId", "source", "double_label"]].merge(m[["servId", *TEXT_COLS]], on="servId", how="left")
    for c in ROUND_HUMAN_COLS[round_name] + COMMON_HUMAN_COLS:
        sheet[c] = ""
    sheet.insert(0, "round", round_name)
    return sheet.sample(frac=1, random_state=SEED_DOUBLE).reset_index(drop=True)
