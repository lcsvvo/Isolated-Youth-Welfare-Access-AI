"""Q2 매칭 검수 배정·집계: 검수 시트 170쌍 중 72쌍을 등급별 층화추출해 R1~R4에게 나누고, 판정 파일을 읽어 일치도와 등급별 요약을 만든다.
AI 판정(170쌍 전수, 별도 저장)은 같은 72쌍의 사람 최종 판정과 비교해 등급별 사용 가능 여부를 가린다.

입력
  - logs/validation/q2_match/match_review_sheet.csv  검수 대상 170쌍의 색인 (바꾸지 않는다. item_no = 행 순서)
  - data/processed/q2_match.csv                      등급(tier)은 표본 추출·배정 균형과 집계에만 쓴다
  - data/processed/services_master.csv, data/raw/gov24/gov24_serviceList.csv  표시용 본문
  - logs/validation/q2_match/ai/ai_judgements.csv    AI 판정(servId, gov_id, ai_same, ...). 사람 판정이 생긴 뒤 compare_ai만 읽는다
출력 (logs/validation/q2_match/)
  - review_assignment.csv        사람 표본 72쌍: item_no, servId, gov_id, 1차·2차 검수자 (등급·점수 없음)
  - human/review_R1.csv ~ human/review_R4.csv  검수자별 판정 파일 (24행: 1차 18 + 2차 6, 중복 검수 여부 표시 없음)
  - gov24_search_index.csv       대안 검색용 보조금24 전체 목록
  - review_disagreements.csv     중복 검수 불일치 쌍 (판정이 있을 때만)
  - review_agreement.csv         kappa·일치율과 95% 신뢰구간(항목 단위 percentile bootstrap 2000회, 고정 시드, n<10이면 생략)
  - review_confusion.csv         중복 판정 3x3 교차표, 불일치 유형(동일↔다름 등) 건수(전체·등급별)
  - review_adjudication.csv      불일치 항목의 원 판정(A·B)과 최종 판정 칸(h_final 등). 다시 만들어도 채운 행은 보존
  - match_final.csv              최종 판정표(표본 72쌍): servId, gov_id, item_no, final_same, final_source, alt_gov_id
  - review_summary.csv           등급별 판정 현황 (match_final의 final_same 기준)
  - ai_vs_human.csv              등급별 AI-사람 일치도와 AI 판정 사용 가능 여부
  - ai_vs_human_confusion.csv    등급별 AI x 사람 3x3 교차표

표본
  - 등급별 층화추출(시드 HUMAN_SEED): A15 B15 C27 U15 = 72쌍. 이 중 24쌍(A5 B5 C9 U5)은 2명이 중복 검수한다.
  - 1차 18쌍씩(등급 균형), 2차 6쌍씩, 검수자 쌍 조합 6개에 4건씩.

원칙
  - 검수자에게 등급·점수를 보이지 않는다. 중복 검수 항목도 일반 항목과 구별되지 않게 하여 독립 판정을 받는다.
  - 판정이 없는 값은 채우지 않는다. 판정이 비어 있으면 일치도·비율을 계산하지 않고 '미완료'로 표시한다.
  - 일치도 해석: kappa 0.60은 자동 통과 기준이 아니라 검수 기준 재점검 참고값(미만이면 "검수 기준 재점검 참고"),
    0.80은 목표 수준이다. 중복 검수 24건은 표본이 작아 신뢰구간이 넓고, 범주가 한쪽(예: 대부분 동일)으로 쏠리면
    일치율이 높아도 kappa가 낮게 나올 수 있으므로(kappa 역설) 일치율을 함께 본다.
    20건 미만이 양쪽 판정이면 kappa를 내지 않고 '미완료'로 두되 교차표 건수는 남긴다.
  - 원 판정은 검수자 파일과 조정 파일에만 남기고, 최종 판정(match_final)에는 합친 값만 둔다.
    final_source: 단독(중복 아닌 항목의 1차 판정) / 일치(두 판정 같음) / 조정(불일치를 h_final로 확정) /
    미해결(불일치인데 h_final 없음) / 미판정(판정 없음, 중복 항목은 두 판정이 모두 있어야 판정으로 봄).
    h_final은 final_method와 final_by가 함께 있을 때만 최종값으로 인정한다.
  - 전체 일치율·재현율은 계산하지 않는다. 모집단 정밀도·재현율은 등급별 표본 가중치(등급별 모집단 크기는
    q2_match.csv)와 사람 판정이 모두 있어야 계산할 수 있으며, 판정이 끝나기 전에는 범위 밖이다.

AI 판정 사용 규칙 (compare_ai, 사전 고정: 결과를 보고 바꾸지 않는다)
  등급 단위로 AI 판정을 쓸 수 있는 것은 다음을 모두 만족할 때뿐이다.
    (1) 거짓 동일(AI 동일 & 사람 최종 다름) 0건
    (2) AI-사람 일치율 90% 이상
    (3) 그 등급 표본 전체에 사람 최종 판정(match_final)이 있고 AI 판정도 있음
  아니면 status는 '사람 전수 판정 필요'. 일부만 판정됐고 거짓 동일이 아직 없으면 '판정 미완료'. 사람 판정이 하나도 없으면 숫자 없이 헤더·상태만 쓴다.
  주의: 등급별 표본이 15~27쌍이라 일치율·kappa의 불확실성이 크다(신뢰구간 넓음). 이 규칙은 통과해도 해당 등급 전체의
  AI 정확도를 보증하지 않으며, 표본에서 거짓 동일이 없었다는 뜻일 뿐이다.

실행: python -m src.preprocess.q2_review
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd

from ..collect._common import PROCESSED, ROOT
from .q2_match import GOV_LIST, MASTER, SEED, VALID

MATCH = PROCESSED / "q2_match.csv"
REVIEWERS = ["R1", "R2", "R3", "R4"]
HUMAN_N = {"A": 15, "B": 15, "C": 27, "U": 15}            # 사람 판정 표본(등급별)
DOUBLE_N = {"A": 5, "B": 5, "C": 9, "U": 5}               # 표본 중 중복 검수(등급별)
HUMAN_SEED = 20261112
BOOT_N = 2000
BOOT_SEED = 20261105
MIN_CI_N = 10
KAPPA_REVIEW = 0.60
KAPPA_TARGET = 0.80
CLIP = 400
CLIP_INDEX = 200

SAME = ["동일", "다름", "판단불가"]
CODES = ["이름", "기관", "지역", "대상", "목적", "내용"]
ALT_SEARCHED = ["예", "아니오"]
ALT_RESULT = ["찾음", "못찾음", "해당없음"]
FINAL_METHOD = ["제3자", "합의"]
DIS_TYPES = [("동일", "다름"), ("동일", "판단불가"), ("다름", "판단불가")]
AGREE_NOTE = ("0.60은 자동 통과 기준이 아니라 기준 재점검 참고값, 0.80은 목표 수준. 중복 검수 24건은 표본이 작아 신뢰구간이 넓고, "
              "범주 불균형(예: 대부분 동일) 시 일치율이 높아도 kappa가 낮게 나올 수 있으므로(kappa 역설) 일치율을 함께 해석")

DISPLAY = ["ssis_name", "ssis_org", "ssis_region", "ssis_url", "ssis_summary", "ssis_target", "ssis_benefit",
           "gov_name", "gov_org", "gov_url", "gov_summary", "gov_target", "gov_benefit"]
JUDGE = ["h_same", "h_basis", "h_conflict", "h_alt_searched", "h_alt_query", "h_alt_gov_id", "h_alt_result",
         "h_note", "reviewer", "reviewed_at"]
FILE_COLS = ["item_no", "servId", "gov_id"] + DISPLAY + JUDGE
FINAL_FIELDS = ["h_final", "final_method", "final_by", "final_note", "final_at"]
ADJ_COLS = ["item_no", "servId", "gov_id", "reviewer_a", "h_same_a", "h_basis_a", "reviewer_b", "h_same_b", "h_basis_b"] + FINAL_FIELDS
FINAL_COLS = ["servId", "gov_id", "item_no", "final_same", "final_source", "alt_gov_id"]
AGREE_COLS = ["metric", "estimate", "ci_low", "ci_high", "n_double", "n_both", "status", "interpretation_note"]
CONF_COLS = ["table", "tier", "row", "col", "count"]
AI_AGREE_MIN = 0.90                                       # AI 판정 사용 조건(사전 고정)
AI_RULE = "AI 판정 사용 조건(사전 고정): 거짓 동일(AI 동일·사람 다름) 0건 그리고 일치율 90% 이상 그리고 표본 전체에 사람 최종·AI 판정 있음"
# 매칭 판정 기준 보완 (2026-11-05 결정, 사람·AI 공통). h_note/ai_note에 아래 표지를 적는다
#  - 포함관계: 한쪽이 여러 지원을 묶고 다른 쪽은 그중 일부만 다루면 '다름' + 표지 '포함관계'.
#    Q2 정보 비교 단위가 달라 비교에서 빼고, 출처 연결(Q2-1)에서는 '부분 대응'으로 따로 센다
#  - 시행 주체: 같은 지방자치단체(광역 또는 기초)여야 '동일'. 기초와 광역 사업은 '다름' + 표지 '주체 단계 차이'
#    단, 제주시·서귀포시는 자치단체가 아닌 행정시이므로 제주특별자치도와 같은 주체로 본다(표지 '행정시')
#  - 행정구역 개편(인천 2026): 승계 관계가 분명하면 같은 주체로 본다(표지 '개편 승계').
#    옛 중구처럼 둘로 나뉘어 승계 구가 불분명하면 '판단불가'
#  알려진 매칭 한계(이번 표본에서는 고치지 않음): 제주시·서귀포시 SSIS 78건과 인천 옛 중구·동구 7건은 보조금24에
#  같은 이름의 시군구가 없어 시도 단위로 넓혀 찾았고, 그래서 최고 C 등급이다. 매칭 규칙 개정은 사람 검수 후 별도로 한다
MATCH_NOTE_TAGS = ["포함관계", "주체 단계 차이", "행정시", "개편 승계"]

AI_USE, AI_NEED_HUMAN, AI_WAIT = "AI 판정 사용 가능", "사람 전수 판정 필요", "사람 판정 대기"
AI_PENDING = "판정 미완료"   # 일부만 판정됨: 거짓 동일이 아직 없으면 결론을 미룬다(거짓 동일이 나오면 바로 사람 전수 판정 필요)
VS_COLS = ["tier", "n_sampled", "n_human_final", "n_compared", "n_agree", "pct_agree", "kappa", "ci_low", "ci_high",
           "n_false_same", "n_ai_diff_human_same", "rule", "reason", "status"]
VS_CONF_COLS = ["tier", "ai_same", "human_same", "count"]
ASSIGN_COLS = ["item_no", "servId", "gov_id", "reviewer_primary", "reviewer_second"]


def read(path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, encoding="utf-8-sig", keep_default_na=False)


def clip(s: pd.Series, n: int = CLIP) -> pd.Series:
    return s.fillna("").astype(str).str.replace(r"\s+", " ", regex=True).str.strip().str[:n]


# ------------------------------------------------------------------ 배정
def assign(sheet: pd.DataFrame, tiers: pd.Series, seed: int = HUMAN_SEED) -> pd.DataFrame:
    """등급별로 섞어 HUMAN_N만큼 뽑고(나머지는 사람 판정 대상 아님), 뽑은 쌍을 검수자에게 돌아가며 배정한다. 시작 위치를 등급 사이에 이어 가서 검수자별 합계를 맞춘다.

    중복 검수는 등급별로 섞은 순서에서 뽑되, 1차 검수자가 고르게 퍼지도록 중복 건수가 적은 검수자의 항목을 먼저 고른다.
    2차 검수자는 1차와 달라야 하고, 2차 부담과 검수자 쌍 조합이 고르게 되도록 탐욕적으로 고른다.
    """
    rng = np.random.default_rng(seed)
    df = sheet[["servId", "gov_id"]].copy()
    df["item_no"] = np.arange(1, len(df) + 1)
    df["tier"] = df["servId"].map(tiers).to_numpy()
    if df["tier"].isna().any():
        raise ValueError("q2_match.csv에 없는 servId가 검수 시트에 있습니다")
    df["reviewer_primary"] = ""
    df["reviewer_second"] = ""

    offset, chosen = 0, []
    n1 = {r: 0 for r in REVIEWERS}
    doubles = []                                             # 등급별 뽑은 순서를 번갈아 처리하기 위한 목록
    for t in sorted(df["tier"].unique()):
        pos = df.index[df["tier"].eq(t)].to_numpy()
        pos = pos[rng.permutation(len(pos))][:HUMAN_N[t]]
        for i, p in enumerate(pos):
            df.at[p, "reviewer_primary"] = REVIEWERS[(offset + i) % len(REVIEWERS)]
        offset += len(pos)
        chosen.extend(pos)
        pick = []                                            # 이미 뽑힌 중복 건수가 적은 1차 검수자의 항목부터
        for _ in range(DOUBLE_N.get(t, 0)):
            left = [p for p in pos if p not in pick]
            pick.append(min(left, key=lambda p: n1[df.at[p, "reviewer_primary"]]))
            n1[df.at[pick[-1], "reviewer_primary"]] += 1
        doubles.append(pick)

    order = [p for grp in zip(*[g + [None] * (max(map(len, doubles)) - len(g)) for g in doubles]) for p in grp if p is not None]
    load = {r: 0 for r in REVIEWERS}
    pair = {}
    for p in order:
        prim = df.at[p, "reviewer_primary"]
        cand = [r for r in REVIEWERS if r != prim]
        r2 = min(cand, key=lambda r: (load[r], pair.get(frozenset((prim, r)), 0), REVIEWERS.index(r)))
        df.at[p, "reviewer_second"] = r2
        load[r2] += 1
        pair[frozenset((prim, r2))] = pair.get(frozenset((prim, r2)), 0) + 1

    def cost() -> int:
        # 2차 부담 편차를 쌍 조합 편차보다 우선한다
        ld = df["reviewer_second"].value_counts()
        pr = pd.Series([frozenset(x) for x in zip(df.loc[order, "reviewer_primary"], df.loc[order, "reviewer_second"])]).value_counts()
        pr = pr.reindex([frozenset(c) for c in combinations(REVIEWERS, 2)], fill_value=0)
        return 100 * int((ld.reindex(REVIEWERS, fill_value=0) ** 2).sum()) + int((pr ** 2).sum())

    best, improved = cost(), True
    while improved:                                          # 한 건씩 2차 검수자를 바꿔 보며 비용이 줄면 채택
        improved = False
        for p in order:
            cur = df.at[p, "reviewer_second"]
            for r in REVIEWERS:
                if r in (cur, df.at[p, "reviewer_primary"]):
                    continue
                df.at[p, "reviewer_second"] = r
                c = cost()
                if c < best:
                    best, cur, improved = c, r, True
                else:
                    df.at[p, "reviewer_second"] = cur
    return df.loc[sorted(chosen)].reset_index(drop=True)


# ------------------------------------------------------------------ 표시용 본문
def display_table(sheet: pd.DataFrame) -> pd.DataFrame:
    m = read(MASTER).set_index("servId")
    g = read(GOV_LIST).set_index("서비스ID")
    ms, gs = m.loc[sheet["servId"]], g.loc[sheet["gov_id"]]
    region = (ms["region_sido"].replace("-", "") + " " + ms["region_sgg"].replace("-", "")).str.strip()
    summary = pd.Series(np.where(ms["source"].eq("central"), ms["overview_text"], ms["summary_text"]), index=ms.index)
    out = pd.DataFrame({
        "ssis_name": ms["servNm"].to_numpy(),
        "ssis_org": (ms["org"] + " " + ms["org_dept"]).str.strip().to_numpy(),
        "ssis_region": region.to_numpy(),
        "ssis_url": ms["detail_url"].to_numpy(),
        "ssis_summary": clip(summary).to_numpy(),
        "ssis_target": clip(ms["target_text"]).to_numpy(),
        "ssis_benefit": clip(ms["benefit_text"]).to_numpy(),
        "gov_name": gs["서비스명"].to_numpy(),
        "gov_org": (gs["소관기관명"] + " " + gs["부서명"]).str.strip().to_numpy(),
        "gov_url": gs["상세조회URL"].to_numpy(),
        "gov_summary": clip(gs["서비스목적요약"]).to_numpy(),
        "gov_target": clip(gs["지원대상"]).to_numpy(),
        "gov_benefit": clip(gs["지원내용"]).to_numpy(),
    })
    out["item_no"] = sheet["item_no"].to_numpy()
    return out


def reviewer_path(valid, r: str):
    return valid / "human" / f"review_{r}.csv"


def legacy_reviewer_path(valid, r: str):
    return valid / f"review_{r}.csv"                         # 170쌍 시절 위치


def guard(valid) -> None:
    """검수자 파일(새·옛 위치)에 판정이 하나라도 있으면 아무것도 덮어쓰지 않는다."""
    for p in [f(valid, r) for r in REVIEWERS for f in (reviewer_path, legacy_reviewer_path)]:
        if not p.exists():
            continue
        prev = read(p)
        cols = [c for c in prev.columns if c.startswith("h_") or c in ("reviewer", "reviewed_at")]
        if cols and prev[cols].ne("").any().any():
            raise ValueError(f"판정값이 있어 덮어쓰지 않습니다: {p}")


# ------------------------------------------------------------------ 생성
def build(valid=VALID, match_path=MATCH, seed: int = HUMAN_SEED) -> pd.DataFrame:
    valid.mkdir(parents=True, exist_ok=True)
    guard(valid)
    (valid / "human").mkdir(exist_ok=True)
    sheet = read(valid / "match_review_sheet.csv")
    tiers = read(match_path).set_index("servId")["tier"]
    asg = assign(sheet, tiers, seed)
    disp = display_table(asg.assign(item_no=asg["item_no"]))
    full = asg[["item_no", "servId", "gov_id"]].reset_index(drop=True).merge(disp, on="item_no", validate="1:1")
    for c in JUDGE:
        full[c] = ""
    full = full[FILE_COLS]

    asg[ASSIGN_COLS].to_csv(valid / "review_assignment.csv", index=False, encoding="utf-8-sig")
    for k, r in enumerate(REVIEWERS):
        nos = asg.loc[asg["reviewer_primary"].eq(r) | asg["reviewer_second"].eq(r), "item_no"]
        part = full[full["item_no"].isin(nos)].sample(frac=1, random_state=seed + k + 1)
        part.to_csv(reviewer_path(valid, r), index=False, encoding="utf-8-sig")

    g = read(GOV_LIST)
    idx = g[["서비스ID", "서비스명", "소관기관명", "소관기관유형", "부서명", "서비스목적요약", "상세조회URL"]].copy()
    idx["서비스목적요약"] = clip(idx["서비스목적요약"], CLIP_INDEX)
    idx.to_csv(valid / "gov24_search_index.csv", index=False, encoding="utf-8-sig")
    return asg


# ------------------------------------------------------------------ 집계
def kappa(a, b, labels=SAME) -> float:
    """Cohen's kappa. 두 판정이 모두 한 범주뿐이라 우연 일치가 1이면 정의되지 않아 nan."""
    k = {c: i for i, c in enumerate(labels)}
    cm = np.zeros((len(labels), len(labels)))
    for x, y in zip(a, b):
        cm[k[x], k[y]] += 1
    n = cm.sum()
    po = np.trace(cm) / n
    pe = float((cm.sum(1) * cm.sum(0)).sum() / n ** 2)
    return float("nan") if pe == 1 else float((po - pe) / (1 - pe))


def validate(rows: pd.DataFrame, reviewer: str) -> pd.DataFrame:
    """허용값을 벗어난 판정 행. h_same이 잘못이어도 행은 남기되 집계에서는 판정 없음으로 본다."""
    bad = []

    def multi(v):
        return [x.strip() for x in v.split("|") if x.strip()] if v.strip() else []

    for r in rows.itertuples(index=False):
        why = []
        if r.h_same.strip() not in SAME + [""]:
            why.append(f"h_same={r.h_same!r}")
        for c in ("h_basis", "h_conflict"):
            wrong = [x for x in multi(getattr(r, c)) if x not in CODES]
            if wrong:
                why.append(f"{c}={wrong}")
        if r.h_alt_searched.strip() not in ALT_SEARCHED + [""]:
            why.append(f"h_alt_searched={r.h_alt_searched!r}")
        if r.h_alt_result.strip() not in ALT_RESULT + [""]:
            why.append(f"h_alt_result={r.h_alt_result!r}")
        if why:
            bad.append({"reviewer_file": reviewer, "item_no": r.item_no, "problem": "; ".join(why)})
    return pd.DataFrame(bad, columns=["reviewer_file", "item_no", "problem"])


def load_judgements(valid) -> tuple[dict, pd.DataFrame]:
    """{(검수자, item_no): 행} 과 오류 행 목록. 파일이 없으면 판정 없음."""
    jd, bad = {}, []
    for r in REVIEWERS:
        p = reviewer_path(valid, r)
        if not p.exists():
            continue
        f = read(p)
        bad.append(validate(f, r))
        for row in f.to_dict("records"):
            row["h_same"] = row["h_same"].strip() if row["h_same"].strip() in SAME else ""
            jd[(r, row["item_no"])] = row
    return jd, pd.concat(bad, ignore_index=True) if bad else pd.DataFrame(columns=["reviewer_file", "item_no", "problem"])


def boot_ci(a, b, stat, n_boot: int = BOOT_N, seed: int = BOOT_SEED) -> tuple[float, float]:
    """항목 단위 percentile bootstrap 95% 신뢰구간. n이 MIN_CI_N 미만이거나 계산 가능한 표본이 없으면 (nan, nan)."""
    a, b = np.asarray(a, dtype=object), np.asarray(b, dtype=object)
    n = len(a)
    if n < MIN_CI_N:
        return float("nan"), float("nan")
    idx = np.random.default_rng(seed).integers(0, n, size=(n_boot, n))
    vals = np.array([stat(a[i], b[i]) for i in idx], dtype=float)
    vals = vals[~np.isnan(vals)]
    if not len(vals):
        return float("nan"), float("nan")
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


def pct_agree(a, b) -> float:
    return float((np.asarray(a, dtype=object) == np.asarray(b, dtype=object)).mean())


def kappa_flag(k: float) -> str:
    if np.isnan(k):
        return ""
    if k < KAPPA_REVIEW:
        return "kappa 0.60 미만: 검수 기준 재점검 참고"
    return "kappa 0.80 이상: 목표 수준" if k >= KAPPA_TARGET else "kappa 0.60 이상 0.80 미만"


def agreement_table(agree: dict, both: pd.DataFrame) -> pd.DataFrame:
    """일치도 지표와 95% 신뢰구간. 판정(통과·탈락)이 아니라 해석 참고용 메모만 붙인다."""
    rows = []
    for metric, fn in (("kappa", kappa), ("pct_agree", pct_agree)):
        est = agree[metric]
        lo, hi = boot_ci(both["j1"], both["j2"], fn) if len(both) and not (metric == "kappa" and np.isnan(est)) else (np.nan, np.nan)
        note = AGREE_NOTE + (" / " + kappa_flag(est) if metric == "kappa" and kappa_flag(est) else "")
        rows.append({"metric": metric, "estimate": round(est, 4) if not np.isnan(est) else np.nan,
                     "ci_low": round(lo, 4) if not np.isnan(lo) else np.nan,
                     "ci_high": round(hi, 4) if not np.isnan(hi) else np.nan,
                     "n_double": agree["n_double"], "n_both": agree["n_both"], "status": agree["status"],
                     "interpretation_note": note})
    return pd.DataFrame(rows, columns=AGREE_COLS)


def confusion_table(both: pd.DataFrame) -> pd.DataFrame:
    """중복 판정 3x3 교차표(1차 x 2차)와 불일치 유형 건수(전체·등급별). 등급은 집계 때만 붙인다."""
    rows = [{"table": "confusion", "tier": "전체", "row": x, "col": y,
             "count": int(((both["j1"] == x) & (both["j2"] == y)).sum())} for x in SAME for y in SAME]
    groups = [("전체", both)] + [(t, g) for t, g in both.groupby("tier")]
    for t, g in groups:
        for x, y in DIS_TYPES:
            c = int((((g["j1"] == x) & (g["j2"] == y)) | ((g["j1"] == y) & (g["j2"] == x))).sum())
            rows.append({"table": "disagreement_type", "tier": t, "row": f"{x}↔{y}", "col": "", "count": c})
    return pd.DataFrame(rows, columns=CONF_COLS)


def validate_adjudication(adj: pd.DataFrame) -> pd.DataFrame:
    bad = []
    for r in adj.itertuples(index=False):
        why = []
        if r.h_final.strip() not in SAME + [""]:
            why.append(f"h_final={r.h_final!r}")
        if r.final_method.strip() not in FINAL_METHOD + [""]:
            why.append(f"final_method={r.final_method!r}")
        if r.h_final.strip() in SAME and not r.final_method.strip():
            why.append("final_method 없음(최종값 미인정)")
        if r.h_final.strip() in SAME and not r.final_by.strip():
            why.append("final_by 없음(최종값 미인정)")
        if why:
            bad.append({"reviewer_file": "adjudication", "item_no": r.item_no, "problem": "; ".join(why)})
    return pd.DataFrame(bad, columns=["reviewer_file", "item_no", "problem"])


def update_adjudication(valid, current: pd.DataFrame) -> pd.DataFrame:
    """조정 파일 갱신. 조정 칸이 하나라도 채워진 행은 그대로 보존하고, 아직 불일치인 미조정 행은 원 판정만 새로 쓴다.
    새 불일치는 추가한다. 더는 불일치가 아니고 조정 칸도 빈 행만 뺀다."""
    p = valid / "review_adjudication.csv"
    prev = read(p) if p.exists() else pd.DataFrame(columns=ADJ_COLS)
    for c in ADJ_COLS:
        if c not in prev.columns:
            prev[c] = ""
    prev = prev[ADJ_COLS]
    filled = prev[prev[FINAL_FIELDS].apply(lambda s: s.str.strip()).ne("").any(axis=1)]
    new = current[~current["item_no"].isin(filled["item_no"])]
    out = pd.concat([filled, new], ignore_index=True)
    out = out.iloc[np.argsort(out["item_no"].astype(int).to_numpy(), kind="stable")].reset_index(drop=True)
    out.to_csv(p, index=False, encoding="utf-8-sig")
    return out


def aggregate(valid=VALID, match_path=MATCH, min_double: int = 20) -> dict:
    asg = read(valid / "review_assignment.csv")
    tiers = read(match_path).set_index("servId")["tier"]
    asg["tier"] = asg["servId"].map(tiers)
    jd, invalid = load_judgements(valid)

    def get(r, no, col="h_same"):
        return jd.get((r, no), {}).get(col, "") if r else ""

    asg["j1"] = [get(r, no) for r, no in zip(asg["reviewer_primary"], asg["item_no"])]
    asg["j2"] = [get(r, no) for r, no in zip(asg["reviewer_second"], asg["item_no"])]
    asg["alt1"] = [get(r, no, "h_alt_result") for r, no in zip(asg["reviewer_primary"], asg["item_no"])]
    asg["alt_id"] = [get(r, no, "h_alt_gov_id").strip() for r, no in zip(asg["reviewer_primary"], asg["item_no"])]

    dbl = asg[asg["reviewer_second"].ne("")]
    both = dbl[dbl["j1"].ne("") & dbl["j2"].ne("")]
    agree = {"n_double": len(dbl), "n_both": len(both), "pct_agree": np.nan, "kappa": np.nan,
             "status": f"중복 판정 미완료 ({len(both)}/{len(dbl)})"}
    if len(both):
        agree["pct_agree"] = pct_agree(both["j1"], both["j2"])
    if len(both) >= min_double:
        agree["kappa"] = kappa(both["j1"], both["j2"])
        agree["status"] = "완료"
    agr = agreement_table(agree, both)
    agr.to_csv(valid / "review_agreement.csv", index=False, encoding="utf-8-sig")
    conf = confusion_table(both)
    conf.to_csv(valid / "review_confusion.csv", index=False, encoding="utf-8-sig")

    d = both[both["j1"] != both["j2"]]
    dis = d.rename(columns={"j1": "h_same_primary", "j2": "h_same_second"})[
        ["item_no", "servId", "gov_id", "reviewer_primary", "h_same_primary", "reviewer_second", "h_same_second"]]
    if len(both):
        dis.to_csv(valid / "review_disagreements.csv", index=False, encoding="utf-8-sig")

    cur = pd.DataFrame({
        "item_no": d["item_no"], "servId": d["servId"], "gov_id": d["gov_id"],
        "reviewer_a": d["reviewer_primary"], "h_same_a": d["j1"],
        "h_basis_a": [get(r, no, "h_basis") for r, no in zip(d["reviewer_primary"], d["item_no"])],
        "reviewer_b": d["reviewer_second"], "h_same_b": d["j2"],
        "h_basis_b": [get(r, no, "h_basis") for r, no in zip(d["reviewer_second"], d["item_no"])],
    }).reset_index(drop=True)
    for c in ADJ_COLS:
        if c not in cur.columns:
            cur[c] = ""
    adj = update_adjudication(valid, cur[ADJ_COLS])
    adj_bad = validate_adjudication(adj)
    invalid = pd.concat([invalid, adj_bad], ignore_index=True)
    # 최종값은 판정 방식(제3자/합의)과 판정자가 함께 기록된 경우에만 인정한다. 아니면 미해결로 남긴다
    h_final = {r.item_no: r.h_final.strip() for r in adj.itertuples(index=False)
               if r.h_final.strip() in SAME and r.final_method.strip() in FINAL_METHOD and r.final_by.strip()}

    src, same = [], []
    for r in asg.itertuples(index=False):
        if r.reviewer_second == "":
            s, f = ("단독", r.j1) if r.j1 else ("미판정", "미판정")
        elif r.j1 and r.j2:
            if r.j1 == r.j2:
                s, f = "일치", r.j1
            elif r.item_no in h_final:
                s, f = "조정", h_final[r.item_no]
            else:
                s, f = "미해결", "미판정"
        else:
            s, f = "미판정", "미판정"
        src.append(s)
        same.append(f)
    asg["final_same"], asg["final_source"] = same, src
    asg["alt_gov_id"] = np.where(asg["alt1"].eq("찾음"), asg["alt_id"], "")
    final = asg[FINAL_COLS].copy()
    final.to_csv(valid / "match_final.csv", index=False, encoding="utf-8-sig")

    rows = []
    for t, grp in asg.groupby("tier"):
        n, nj = len(grp), int(grp["final_same"].isin(SAME).sum())
        c = {k: int(grp["final_same"].eq(k).sum()) for k in SAME}
        row = {"tier": t, "n_items": n, "n_judged": nj, "n_same": c["동일"], "n_diff": c["다름"],
               "n_unknown": c["판단불가"], "n_unresolved": int(grp["final_source"].eq("미해결").sum()),
               "n_alt_found": int(grp["alt1"].eq("찾음").sum()),
               "same_share": np.nan, "diff_share": np.nan, "unknown_share": np.nan,
               "same_share_excl_unknown": np.nan, "status": "미완료"}
        if nj == n:
            row.update(same_share=round(c["동일"] / n, 4), diff_share=round(c["다름"] / n, 4),
                       unknown_share=round(c["판단불가"] / n, 4), status="완료")
            if c["동일"] + c["다름"]:
                row["same_share_excl_unknown"] = round(c["동일"] / (c["동일"] + c["다름"]), 4)
        rows.append(row)
    summ = pd.DataFrame(rows)
    summ.to_csv(valid / "review_summary.csv", index=False, encoding="utf-8-sig")
    return {"summary": summ, "agreement": agree, "agreement_table": agr, "confusion": conf,
            "disagreements": dis, "adjudication": adj, "final": final, "invalid": invalid}


def compare_ai(valid=VALID, match_path=MATCH, final: pd.DataFrame | None = None, ai_path=None) -> dict:
    """AI 판정(ai_same)을 사람 최종 판정(match_final의 final_same)과 등급별로 비교한다. 사용 규칙은 모듈 독스트링 참조(사전 고정).

    사람 최종 판정이 하나도 없으면 AI 파일을 읽지 않고 숫자 없이 헤더·상태만 쓴다.
    """
    f = (read(valid / "match_final.csv") if final is None else final).copy()
    f["tier"] = f["servId"].map(read(match_path).set_index("servId")["tier"])
    f["human"] = f["final_same"].where(f["final_same"].isin(SAME), "")
    ai_path = (valid / "ai" / "ai_judgements.csv") if ai_path is None else ai_path
    has_human = bool(f["human"].ne("").any())
    has_ai = False
    f["ai"] = ""
    if has_human and ai_path.exists():
        ai = read(ai_path)
        if ai.duplicated(["servId", "gov_id"]).any():
            raise ValueError(f"AI 판정에 중복 쌍이 있습니다: {ai_path}")
        ai["ai"] = ai["ai_same"].str.strip().where(ai["ai_same"].str.strip().isin(SAME), "")
        f = f.drop(columns="ai").merge(ai[["servId", "gov_id", "ai"]], on=["servId", "gov_id"], how="left")
        f["ai"] = f["ai"].fillna("")
        has_ai = True

    rows, conf = [], []
    for t, g in f.groupby("tier"):
        row = dict.fromkeys(VS_COLS, np.nan)
        row.update(tier=t, n_sampled=len(g), rule=AI_RULE)
        if not has_human:
            row.update(reason="사람 판정 없음", status=AI_WAIT)
            rows.append(row)
            continue
        hf = g[g["human"].ne("")]
        cmp = hf[hf["ai"].ne("")]
        n, agree = len(cmp), int((cmp["ai"] == cmp["human"]).sum())
        fs = int(((cmp["ai"] == "동일") & (cmp["human"] == "다름")).sum())
        sd = int(((cmp["ai"] == "다름") & (cmp["human"] == "동일")).sum())
        lo = hi = np.nan
        k = float("nan")
        if n:
            k = kappa(cmp["ai"], cmp["human"])
            if n >= MIN_CI_N and not np.isnan(k):
                lo, hi = boot_ci(cmp["ai"], cmp["human"], kappa)
        why = []
        if len(hf) < len(g):
            why.append(f"사람 최종 판정 미완료 ({len(hf)}/{len(g)})")
        if not has_ai:
            why.append("AI 판정 파일 없음")
        elif len(cmp) < len(hf):
            why.append(f"AI 판정 없음·무효 {len(hf) - len(cmp)}건")
        if fs:
            why.append(f"거짓 동일 {fs}건")
        if n and agree < AI_AGREE_MIN * n:
            why.append("일치율 90% 미만")
        ok = not why and n == len(g) and n > 0
        incomplete = len(hf) < len(g) or not has_ai or len(cmp) < len(hf)
        status = AI_USE if ok else (AI_PENDING if incomplete and not fs else AI_NEED_HUMAN)
        rd = lambda x: round(x, 4) if not np.isnan(x) else np.nan
        row.update(n_human_final=len(hf), n_compared=n, n_agree=agree, pct_agree=rd(agree / n) if n else np.nan,
                   kappa=rd(k), ci_low=rd(lo), ci_high=rd(hi), n_false_same=fs, n_ai_diff_human_same=sd,
                   reason="; ".join(why), status=status)
        rows.append(row)
        conf += [{"tier": t, "ai_same": x, "human_same": y,
                  "count": int(((cmp["ai"] == x) & (cmp["human"] == y)).sum())} for x in SAME for y in SAME]
    out, cout = pd.DataFrame(rows, columns=VS_COLS), pd.DataFrame(conf, columns=VS_CONF_COLS)
    out.to_csv(valid / "ai_vs_human.csv", index=False, encoding="utf-8-sig")
    cout.to_csv(valid / "ai_vs_human_confusion.csv", index=False, encoding="utf-8-sig")
    return {"table": out, "confusion": cout, "has_human": has_human}


def report(res: dict) -> None:
    ag, summ, inv = res["agreement"], res["summary"], res["invalid"]
    print(f"== 중복 검수 {ag['n_double']}건, 두 판정 모두 있는 {ag['n_both']}건")
    t = res["agreement_table"].set_index("metric")
    if ag["n_both"]:
        for m, lab in (("pct_agree", "일치율"), ("kappa", "Cohen's kappa")):
            r = t.loc[m]
            if np.isnan(r["estimate"]):
                continue
            ci = "" if np.isnan(r["ci_low"]) else f" (95% CI {r['ci_low']:.3f}~{r['ci_high']:.3f})"
            print(f"   {lab} {r['estimate']:.3f}{ci}")
    if ag["status"] != "완료":
        print("   " + ag["status"])
    elif kappa_flag(ag["kappa"]):
        print("   " + kappa_flag(ag["kappa"]) + " (통과·탈락 기준 아님)")
    adj = res["adjudication"]
    print(f"== 불일치 조정 {len(adj)}건 중 미조정 {int(adj['h_final'].eq('').sum())}건")
    print(f"== 오류 행 {len(inv)}건")
    if len(inv):
        print(inv.to_string(index=False))
    print("== 등급별 현황")
    for r in summ.itertuples(index=False):
        if r.status == "완료":
            print(f"   {r.tier}: 동일 {r.n_same} 다름 {r.n_diff} 판단불가 {r.n_unknown} (동일 비율 {r.same_share:.3f}), 대안 찾음 {r.n_alt_found}")
        else:
            print(f"   {r.tier}: 판정 미완료 ({r.n_judged}/{r.n_items}, 미해결 {r.n_unresolved})")


def report_ai(res: dict) -> None:
    print("== AI-사람 비교")
    if not res["has_human"]:
        print("   사람 판정 없음: " + AI_WAIT)
        return
    for r in res["table"].itertuples(index=False):
        pct = "" if np.isnan(r.pct_agree) else f", 일치 {r.n_agree}/{r.n_compared} ({r.pct_agree:.3f})"
        print(f"   {r.tier}: 사람 최종 {int(r.n_human_final)}/{r.n_sampled}{pct}, 거짓 동일 {int(r.n_false_same)} → {r.status}")


def main() -> None:
    if not all(reviewer_path(VALID, r).exists() for r in REVIEWERS):
        asg = build()
        print(f"== 검수 파일 생성: {len(asg)}쌍, 중복 {int(asg['reviewer_second'].ne('').sum())}쌍 → {VALID.relative_to(ROOT)}")
    res = aggregate()
    report(res)
    report_ai(compare_ai(final=res["final"]))


if __name__ == "__main__":
    main()
