"""4단계 규칙 검증의 나이 '확인 필요' 원인 분석 (item 10).

실행 (저장소 루트에서)
  python -m ai.eval.rules_age_audit            # 표 2종 갱신, 검수 CSV는 없을 때만 생성
  python -m ai.eval.rules_age_audit --refresh  # 검수 CSV가 이미 있어도 별도 파일(age_recheck_review_refresh.csv)을 새로 만듦

출력: outputs/07_rules/
- tables/age_check_reasons.csv    사용자 나이(19·25·34) x check_age 판정 사유 분류별 서비스 수
- tables/age_recheck_summary.csv  age_min이 N(원문 근거 없음)인 서비스의 재스캔 후보 유형별 수
- review/age_recheck_review.csv   후보 서비스 1행씩. human_* 열은 비워 둠(사람 검수용)

주의
- 결과는 사람 검수용 후보일 뿐임. 추출값(extracted_*)을 services_master나 규칙에 되쓰지 않음.
- 기존 검수 파일은 덮어쓰지 않음. --refresh는 별도 파일로만 씀.
- 추출 정규식은 판정 근거가 아니라 누락 후보 탐지용이며, 자녀·부모 등 가족 연령 표현은 후보에서 뺌.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

AI = Path(__file__).resolve().parents[1]
ROOT = AI.parent
sys.path.insert(0, str(AI))

from rules.filter import CHECK, EXCLUDE, PASS, check_age  # noqa: E402
from rules.normalize import parse_bound  # noqa: E402

MASTER = ROOT / "data" / "processed" / "services_master.csv"
OUT = ROOT / "outputs" / "07_rules"
USER_AGES = (19, 25, 34)

# ── 1. 판정 사유 분류 ───────────────────────────────────────────
CAT_A = "(a) 하한 N: 연령 원문 근거 없음"
CAT_B1 = "(b) 연령 밖이나 신청자 연령·다른 경로 불확실"
CAT_B2 = "(b) 연령 안이나 범위 여러 개·불확실"
CAT_C = "(c) 연령 범위 오류"
CAT_D_PASS = "(d) 기타: 통과"
CAT_D_EXC = "(d) 기타: 제외"
CAT_D_CHK = "(d) 기타: 확인 필요"
CATEGORIES = [CAT_A, CAT_B1, CAT_B2, CAT_C, CAT_D_PASS, CAT_D_EXC, CAT_D_CHK]


def classify_age_check(svc: dict, age: int) -> str:
    """check_age의 (판정, 이유)를 분류로 바꿈. 판정 로직은 ai/rules/filter.py가 단일 출처임."""
    verdict, reason = check_age(svc, {"age": age})
    if verdict == PASS:
        return CAT_D_PASS
    if verdict == EXCLUDE:
        return CAT_D_EXC
    if reason.startswith("서비스 연령 조건 원문 없음"):
        return CAT_A
    if reason.startswith("서비스 연령 범위 오류"):
        return CAT_C
    if "밖이나" in reason:
        return CAT_B1
    if "안이나" in reason:
        return CAT_B2
    return CAT_D_CHK


def reason_counts(df: pd.DataFrame, ages=USER_AGES) -> pd.DataFrame:
    rows = []
    recs = df.to_dict("records")
    for a in ages:
        vc = pd.Series([classify_age_check(s, a) for s in recs]).value_counts()
        for c in CATEGORIES:
            n = int(vc.get(c, 0))
            rows.append({"user_age": a, "category": c, "n_services": n, "ratio": round(n / len(recs), 4)})
    return pd.DataFrame(rows)


# ── 2. 연령 표현 추출 (누락 후보 탐지용) ────────────────────────
_N = r"(\d{1,3})"
_SE = r"\s*세(?!\s*(?:반|대|환자|아|계))"          # '2세반', '30세대', '2세 환자' 등 제외
_MAN = r"(?:만\s*)?"
_RANGE = re.compile(_MAN + _N + r"\s*(?:세)?\s*(?:~|∼|～|-|–|부터)\s*" + _MAN + _N + _SE + r"(?:\s*까지)?")
_BOUND = re.compile(_MAN + _N + _SE + r"\s*(이상|이하|미만|초과)")

# 신청자가 아닌 가족·자녀 연령일 가능성이 큰 표현
_DEP_NOUN = r"(?:자녀|영유아|영아|유아|아이|손자녀|손자|손녀|조카|피부양)"
_DEP_BEFORE = re.compile(_DEP_NOUN + r"[^\n]{0,10}$|(?:아동|아이)\s*(?:가|이)?\s*$")
_DEP_AFTER = re.compile(
    r"^[^\n]{0,12}?" + _DEP_NOUN
    + r"|^[^\n]{0,14}?(?:을|를)\s*(?:부양|돌보|양육)|^[^\n]{0,8}?(?:아동|아이)[^\n]{0,6}?(?:양육|이 있는|을 둔|를 둔|돌봄)"
    + r"|^\s*[\)）\]]?\s*(?:" + _DEP_NOUN
    + r"|(?:아동|어린이|청소년|학생)[^\n]{0,6}(?:의|을\s*둔|를\s*둔|을\s*양육|를\s*양육|을\s*키우|를\s*키우|을\s*돌보|를\s*돌보|가구|가정|부모|보호자|양육자)"
    + r"|(?:을|를|이)?\s*(?:부양|돌보|양육|키우)|(?:의\s*)?(?:부모|노부모|보호자|양육자|배우자))")

NO_LIMIT = re.compile(
    r"(?:연령|나이)\s*(?:에\s*)?(?:제한\s*(?:이\s*)?(?:없|무)|무관|불문|구분\s*없|관계\s*없|상관\s*없)"
    r"|전\s*연령|전연령|누구나|전\s*국민|모든\s*(?:연령|국민|연령대)"
)

TEXT_FIELDS = ["target_text", "criteria_text", "summary_text", "overview_text",
               "target_text_raw", "criteria_text_raw", "summary_text_raw", "overview_text_raw"]

T_EXTRACT = "연령 조건 누락 후보"
T_NOLIMIT = "연령 제한 없음 후보"
T_DEPONLY = "가족·자녀 연령 표현만 있음"
T_NONE = "원문 정보 부족"
TYPE_ORDER = {T_EXTRACT: 0, T_NOLIMIT: 1}


def _snippet(text: str, s: int, e: int, width: int = 120) -> str:
    left = max(0, s - (width - (e - s)) // 2)
    seg = re.sub(r"\s+", " ", text[left:left + width + 40]).strip()
    return seg[:width]


def _is_dependent(text: str, s: int, e: int) -> bool:
    before = text[max(0, s - 15):s]
    after = text[e:e + 24]
    return bool(_DEP_BEFORE.search(before) or _DEP_AFTER.search(after))


def extract_ages(text: str) -> tuple[list[dict], int]:
    """텍스트에서 연령 범위·경계 표현을 찾음.

    반환: (신청자 연령으로 보이는 후보 리스트, 가족·자녀 연령으로 보여 뺀 건수).
    각 후보: {min, max, start, end}. 미만은 N-1, 초과는 N+1로 환산함.
    """
    if not isinstance(text, str) or not text:
        return [], 0
    found, used = [], []
    dropped = 0

    def overlaps(s, e):
        return any(s < ue and us < e for us, ue in used)

    def add(lo, hi, s, e):
        nonlocal dropped
        used.append((s, e))
        vals = [v for v in (lo, hi) if v is not None]
        if any(not 0 <= v <= 120 for v in vals):
            return
        if lo is not None and hi is not None and lo > hi:
            return
        if _is_dependent(text, s, e):
            dropped += 1
            return
        found.append({"min": lo, "max": hi, "start": s, "end": e})

    for m in _RANGE.finditer(text):
        add(int(m.group(1)), int(m.group(2)), m.start(), m.end())
    for m in _BOUND.finditer(text):
        if overlaps(m.start(), m.end()):
            continue
        n, kind = int(m.group(1)), m.group(2)
        lo = hi = None
        if kind == "이상":
            lo = n
        elif kind == "초과":
            lo = n + 1
        elif kind == "이하":
            hi = n
        else:
            hi = n - 1
        add(lo, hi, m.start(), m.end())
    found.sort(key=lambda r: r["start"])
    return found, dropped


def detect_no_limit(text: str):
    """'연령 제한 없음' 계열 표현 위치. 없으면 None."""
    if not isinstance(text, str) or not text:
        return None
    m = NO_LIMIT.search(text)
    return (m.start(), m.end()) if m else None


def scan_service(svc: dict) -> dict:
    """한 서비스의 텍스트 필드를 순서대로 훑어 후보 유형을 정함. 먼저 찾은 필드를 근거로 씀."""
    dep_total = 0
    nolimit = None
    for f in TEXT_FIELDS:
        text = svc.get(f)
        if not isinstance(text, str) or not text.strip():
            continue
        found, dropped = extract_ages(text)
        dep_total += dropped
        if found:
            mins = [r["min"] for r in found if r["min"] is not None]
            maxs = [r["max"] for r in found if r["max"] is not None]
            first = found[0]
            return {"candidate_type": T_EXTRACT, "field": f,
                    "extracted_min": min(mins) if mins else "", "extracted_max": max(maxs) if maxs else "",
                    "evidence_snippet": _snippet(text, first["start"], first["end"]), "n_matches": len(found)}
        if nolimit is None:
            pos = detect_no_limit(text)
            if pos:
                nolimit = (f, text, pos)
    if nolimit:
        f, text, (s, e) = nolimit
        return {"candidate_type": T_NOLIMIT, "field": f, "extracted_min": "", "extracted_max": "",
                "evidence_snippet": _snippet(text, s, e), "n_matches": 0}
    base = {"field": "", "extracted_min": "", "extracted_max": "", "evidence_snippet": "", "n_matches": 0}
    return {"candidate_type": T_DEPONLY if dep_total else T_NONE, **base}


HUMAN_COLS = ["decision", "human_min", "human_max", "reason", "reviewer", "reviewed_at"]
REVIEW_COLS = ["servId", "servNm", "candidate_type", "extracted_min", "extracted_max",
               "evidence_snippet", "source_field"] + HUMAN_COLS
FIELD_RANK = {f: i for i, f in enumerate(TEXT_FIELDS)}


def _youth_overlap(lo, hi) -> int:
    lo = 0 if lo == "" else int(lo)
    hi = 200 if hi == "" else int(hi)
    return int(lo <= 34 and hi >= 19)


def recheck(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """age_min이 N인 서비스만 재스캔. (검수 행 DataFrame, 유형별 요약 DataFrame) 반환."""
    target = df[[parse_bound(v)[0] == "none" for v in df["age_min"]]]
    rows = []
    for rec in target.to_dict("records"):
        rows.append({"servId": rec["servId"], "servNm": rec.get("servNm", ""), **scan_service(rec)})
    allr = pd.DataFrame(rows, columns=["servId", "servNm", "candidate_type", "field", "extracted_min",
                                       "extracted_max", "evidence_snippet", "n_matches"])
    summ = allr.groupby("candidate_type").size().rename("n_services").reset_index()
    summ["ratio_of_age_N"] = (summ["n_services"] / max(len(allr), 1)).round(4)
    summ["in_review_csv"] = summ["candidate_type"].isin(TYPE_ORDER)
    order = {T_EXTRACT: 0, T_NOLIMIT: 1, T_DEPONLY: 2, T_NONE: 3}
    summ = summ.sort_values("candidate_type", key=lambda s: s.map(order)).reset_index(drop=True)
    rev = allr[allr["candidate_type"].isin(TYPE_ORDER)].copy()
    rev["_t"] = rev["candidate_type"].map(TYPE_ORDER)
    rev["_f"] = rev["field"].map(lambda f: FIELD_RANK.get(f, 99))
    rev["_y"] = [-_youth_overlap(a, b) for a, b in zip(rev["extracted_min"], rev["extracted_max"])]
    rev["_m"] = -rev["n_matches"]
    rev = rev.sort_values(["_t", "_y", "_f", "_m", "servId"], kind="stable")
    rev = rev.rename(columns={"field": "source_field"})
    for c in HUMAN_COLS:
        rev[c] = ""
    return rev[REVIEW_COLS].reset_index(drop=True), summ


def write_review(rev: pd.DataFrame, path: Path, refresh: bool = False) -> Path | None:
    """기존 검수 파일은 덮어쓰지 않음. 없으면 path에 쓰고, 있으면 --refresh일 때만 *_refresh.csv로 씀."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        rev.to_csv(path, index=False, encoding="utf-8-sig")
        return path
    if refresh:
        alt = path.with_name(path.stem + "_refresh" + path.suffix)
        rev.to_csv(alt, index=False, encoding="utf-8-sig")
        return alt
    return None


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="나이 확인 필요 원인 분석")
    ap.add_argument("--refresh", action="store_true", help="검수 CSV가 있어도 별도 *_refresh.csv를 새로 씀")
    args = ap.parse_args(argv)
    df = pd.read_csv(MASTER, dtype=str, keep_default_na=False)
    (OUT / "tables").mkdir(parents=True, exist_ok=True)
    reasons = reason_counts(df)
    reasons.to_csv(OUT / "tables" / "age_check_reasons.csv", index=False, encoding="utf-8-sig")
    rev, summ = recheck(df)
    summ.to_csv(OUT / "tables" / "age_recheck_summary.csv", index=False, encoding="utf-8-sig")
    written = write_review(rev, OUT / "review" / "age_recheck_review.csv", args.refresh)
    print(reasons.pivot(index="category", columns="user_age", values="n_services").to_string())
    print(summ.to_string(index=False))
    print("검수 CSV:", written if written else "기존 파일 유지(--refresh로 별도 파일 생성)", f"행 {len(rev)}")


if __name__ == "__main__":
    main()
