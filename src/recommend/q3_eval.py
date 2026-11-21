"""Q3 평가: 제한적 Recall@10, Success@10, 자격 오류, 대응쌍 통계, 오류 유형.

원칙
- 정답은 사람 판정(judge_type=human, reviewer·reviewed_at 기재)만 사용한다. AI 제안값은 받지 않는다.
- 관련 서비스는 평가 후보 풀 안에서만 확인된다. Recall@10의 분모는 '풀에서 사람이 관련으로 판정한
  서비스 수'이며, 5,288건 전체에 대한 재현율이 아니다(풀 밖 관련 서비스는 누락됨).
- 관련성은 질의쌍(pair) 단위로 판정한다. 일상어·행정어가 같은 정답 집합을 공유해야 대응 비교가 성립한다.
- 집계 단위는 질의(쌍)다. 한 질의의 결과 여러 건을 독립 표본으로 다루지 않는다.
- 자격: 지역·연령은 '명백한 불일치'만 오류로 센다. 정보 부족은 판단 불가로 따로 센다(부적합 아님).
  소득·가구는 판정하지 않고 항상 '확인 필요'로 둔다(PR #13 decisions.md 초안과 같은 원칙).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

K = 10

# 결과 보고에 그대로 붙이는 문구. 결과표·메타 파일에 같이 기록한다.
CLAIM_SCOPE = ("SSIS 서비스 5,288건에 단순 키워드 검색 기준선(q3_kw)을 적용했을 때의 검색 한계. "
               "복지로·정부24의 실제 검색이나 BM25·임베딩 검색의 성능이 아니다.")
RECALL_NOTE = ("제한적 Recall@10: 분모는 평가 후보 풀(주 분석 조건 상위 10의 합집합 + 사람 추가분) 안에서 "
               "사람이 관련으로 판정한 서비스 수. 어느 조건으로도 검색되지 않은 관련 서비스는 분모에서 빠질 수 있어 "
               "5,288건 전체에 대한 재현율이 아니다.")

# 질의쌍 의도 검수 코드북. 관련성 검수보다 먼저 확정한다.
INTENT_EQUIVALENCE = ("동일 의도", "일부 동일", "불일치")
COMPARISON_CLASS = ("직접 비교", "욕구 추론 필요")
INTENT_CODEBOOK = [
    ("intent_equivalence", "동일 의도", "두 질의가 같은 욕구와 같은 해결 방향을 찾는다. 한쪽에만 있는 대상·조건이 없다."),
    ("intent_equivalence", "일부 동일", "욕구는 같으나 한쪽에 대상 집단·조건·지원 방식이 더 있거나 범위가 다르다."),
    ("intent_equivalence", "불일치", "찾는 욕구나 서비스가 다르다. 대응 비교와 관련성 검수에서 뺀다."),
    ("comparison_class", "직접 비교", "일상어가 원하는 도움(상담·돈·방문 등)을 직접 말한다. 행정어와 요청 명확성이 비슷하다."),
    ("comparison_class", "욕구 추론 필요", "일상어가 어려움·상태만 말하고 원하는 도움은 읽는 사람이 추론해야 한다. "
     "표현 차이와 도움 요청의 명확성 차이가 섞인다."),
]
RELEVANCE_PRIMARY_MIN = 2  # 2=직접 관련만 관련으로 본다(주 분석). 민감도는 1 이상.
BOOTSTRAP_N = 10_000
SEED = 20261105


# ---------------------------------------------------------------- 가상 검증 프로필 (실제 개인정보 아님)
@dataclass(frozen=True)
class Profile:
    profile_id: str
    age: int
    sido: str
    sgg: str


PROFILES = {
    "P_A": Profile("P_A", 25, "서울특별시", "관악구"),
    "P_B": Profile("P_B", 29, "경기도", "수원시"),
    "P_C": Profile("P_C", 25, "전남광주통합특별시", "순천시"),
}

ELIG_MISMATCH_REGION = "명백 불일치(지역)"
ELIG_MISMATCH_AGE = "명백 불일치(연령)"
ELIG_UNKNOWN = "판단 불가(연령 정보 없음)"
ELIG_NO_MISMATCH = "명백 불일치 없음"


def _int_or_none(v) -> int | None:
    try:
        return int(str(v))
    except (TypeError, ValueError):
        return None


def region_status(row: pd.Series, p: Profile) -> str:
    """구조화 필드(region_sido·region_sgg) 기준. 중앙 서비스는 필드상 지역 제한이 없어 불일치로 보지 않는다."""
    if str(row.get("source", "")) != "local":
        return "해당 없음(중앙)"
    sido = str(row.get("region_sido") or "")
    sgg = str(row.get("region_sgg") or "") if pd.notna(row.get("region_sgg")) else ""
    if not sido:
        return "판단 불가"
    if sido != p.sido:
        return "불일치"
    if sgg and sgg != p.sgg:
        return "불일치"
    return "일치"


def age_status(row: pd.Series, p: Profile) -> str:
    """age_min·age_max는 규칙 추출 후보(method=rule, 사람 검수 전)다. 자녀 연령 등 오추출 가능성이 있다."""
    lo, hi = str(row.get("age_min", "N")), str(row.get("age_max", "N"))
    if lo == "N" and hi == "N":
        return "판단 불가"
    lo_i, hi_i = _int_or_none(lo), _int_or_none(hi)
    if (lo_i is not None and p.age < lo_i) or (hi_i is not None and p.age > hi_i):
        return "불일치"
    return "일치"


def eligibility_auto(row: pd.Series, p: Profile) -> dict:
    """자동 사전 분류(잠정). 사람 판정이 아니다. 지역=field, 연령=rule 근거."""
    rs, ag = region_status(row, p), age_status(row, p)
    if rs == "불일치":
        overall = ELIG_MISMATCH_REGION
    elif ag == "불일치":
        overall = ELIG_MISMATCH_AGE
    elif ag == "판단 불가":
        overall = ELIG_UNKNOWN
    else:
        overall = ELIG_NO_MISMATCH
    return {"region_status": rs, "age_status": ag, "income_status": "확인 필요",
            "elig_auto": overall, "elig_method": "region=field; age=rule(미검수)"}


# ---------------------------------------------------------------- 검색 지표
def recall_at_k(ranked_ids: list[str], relevant: set[str], k: int = K) -> float:
    if not relevant:
        return float("nan")
    return len(set(ranked_ids[:k]) & relevant) / len(relevant)


def success_at_k(ranked_ids: list[str], relevant: set[str], k: int = K) -> float:
    if not relevant:
        return float("nan")
    return float(bool(set(ranked_ids[:k]) & relevant))


def expected_hits_random_ties(ranked: pd.DataFrame, relevant: set[str], k: int = K) -> float:
    """k번째 결과와 (score, tf)가 같은 집단을 무작위 순서로 뒀을 때 상위 k 안 관련 서비스 수의 기댓값.
    servId 동점 처리의 영향을 보는 민감도다. 동점 집단 중 사람 판정이 없는 서비스는 관련 아님으로
    계산되므로 하한 성격이다."""
    if ranked.empty:
        return 0.0
    if len(ranked) <= k:
        return float(ranked["servId"].isin(relevant).sum())
    key = list(zip(ranked["score"], ranked["tf"]))
    kth = key[k - 1]
    above = ranked[[x > kth for x in key]]
    tied = ranked[[x == kth for x in key]]
    slots = k - len(above)
    return float(above["servId"].isin(relevant).sum() + tied["servId"].isin(relevant).sum() * slots / len(tied))


# ---------------------------------------------------------------- 대응쌍 통계
def sign_test_p(diffs: np.ndarray) -> float:
    """정확 부호 검정(양측). 차이 0은 제외한다."""
    d = diffs[~np.isnan(diffs)]
    pos, neg = int((d > 0).sum()), int((d < 0).sum())
    n = pos + neg
    if n == 0:
        return float("nan")
    k = min(pos, neg)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2**n * 2
    return min(1.0, p)


def paired_summary(a: np.ndarray, b: np.ndarray, label: str) -> dict:
    """d = b - a. 질의쌍 부트스트랩 95% CI(백분위). 둘 다 값이 있는 쌍만 쓴다."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = ~(np.isnan(a) | np.isnan(b))
    d = b[ok] - a[ok]
    out = {"comparison": label, "n_pairs": int(ok.sum()), "mean_a": float(np.mean(a[ok])) if ok.any() else np.nan,
           "mean_b": float(np.mean(b[ok])) if ok.any() else np.nan,
           "mean_diff": float(np.mean(d)) if len(d) else np.nan,
           "n_pos": int((d > 0).sum()), "n_neg": int((d < 0).sum()), "n_zero": int((d == 0).sum())}
    if len(d) >= 2:
        rng = np.random.default_rng(SEED)
        boots = d[rng.integers(0, len(d), size=(BOOTSTRAP_N, len(d)))].mean(axis=1)
        out["ci95_lo"], out["ci95_hi"] = float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))
    else:
        out["ci95_lo"] = out["ci95_hi"] = np.nan
    out["sign_test_p"] = sign_test_p(d)
    return out


# ---------------------------------------------------------------- 오류 유형 (기계적 판정)
ERROR_TYPES = {
    "E1_vocab": "행정어(상세 포함)로는 상위 10 안, 일상어(상세 포함)로는 밖 → 어휘 불일치 후보",
    "E2_field": "상세 포함 범위로는 상위 10 안, 서비스명·요약 범위로는 밖(같은 표현) → 필드 위치 후보",
    "E3_none": "주 분석 4개 조건 모두 상위 10 밖 → 복합 욕구·기타 원인 후보(사람이 주된 원인 기록)",
    "E4_elig_mismatch": "관련 서비스가 상위 10 안이나 지역·연령 명백 불일치",
    "E5_elig_unknown": "관련 서비스가 상위 10 안이나 원문 정보 부족으로 자격 판단 불가",
}


def mechanical_error_flags(in_top: dict[str, bool]) -> list[str]:
    """in_top 키: daily_short, daily_full, admin_short, admin_full (관련 서비스 1건이 상위 10 안인지).
    여러 유형 동시 표시를 허용한다. 주된 원인 1개는 사람이 primary_cause 칸에 기록한다."""
    flags = []
    if in_top["admin_full"] and not in_top["daily_full"]:
        flags.append("E1_vocab")
    if (in_top["daily_full"] and not in_top["daily_short"]) or (in_top["admin_full"] and not in_top["admin_short"]):
        flags.append("E2_field")
    if not any(in_top.values()):
        flags.append("E3_none")
    return flags


# ---------------------------------------------------------------- 사람 판정 읽기
def _human_rows(df: pd.DataFrame, required: list[str]) -> pd.DataFrame:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"판정 파일에 열이 없음: {missing}")
    d = df.copy()
    for c in required:
        d[c] = d[c].fillna("").astype(str).str.strip()
    return d[(d["judge_type"] == "human") & (d["reviewer"] != "") & (d["reviewed_at"] != "")].copy()


INTENT_REQUIRED = ["pair_id", "intent_equivalence", "comparison_class", "reviewer", "reviewed_at", "judge_type"]


def load_human_intent(df: pd.DataFrame) -> pd.DataFrame:
    """의도 검수 합의본. 값이 코드북 밖이면 오류(조용히 버리지 않는다)."""
    d = _human_rows(df, INTENT_REQUIRED)
    bad = d[~d["intent_equivalence"].isin(INTENT_EQUIVALENCE) | ~d["comparison_class"].isin(COMPARISON_CLASS)]
    if len(bad):
        raise ValueError(f"코드북 밖 값: {bad['pair_id'].tolist()}")
    if d["pair_id"].duplicated().any():
        raise ValueError("pair_id 중복. 2인 판정은 합의본 1행으로 만든다.")
    return d


def cohen_kappa(a: list[str], b: list[str]) -> float:
    """명목 척도 Cohen's kappa. 기대 일치도가 1이면 nan."""
    if len(a) != len(b) or not a:
        raise ValueError("길이가 같고 비어 있지 않아야 함")
    cats = sorted(set(a) | set(b))
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    pe = sum((a.count(c) / n) * (b.count(c) / n) for c in cats)
    return float("nan") if pe == 1 else (po - pe) / (1 - pe)


HUMAN_REQUIRED = ["pair_id", "servId", "relevance", "reviewer", "reviewed_at", "judge_type"]


def load_human_relevance(df: pd.DataFrame) -> pd.DataFrame:
    """확정 관련성 판정만 남긴다. judge_type이 human이 아니거나 reviewer·reviewed_at이 비면 버린다."""
    d = _human_rows(df, HUMAN_REQUIRED)
    d = d[d["relevance"].isin(["0", "1", "2"])].copy()
    if d.duplicated(["pair_id", "servId"]).any():
        raise ValueError("pair_id·servId 중복 판정. 2인 판정은 합의본 1행으로 만든 뒤 넣는다.")
    d["relevance"] = d["relevance"].astype(int)
    return d
