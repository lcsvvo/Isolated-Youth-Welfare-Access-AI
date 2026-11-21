"""지자체(local) 서비스의 지역 조건 구조화와 사람 검수 시트 생성.

실행 (저장소 루트에서)
  python -m ai.eval.rules_region_extract            # 요약표 갱신, 검수 CSV는 없거나 사람 입력이 비어 있을 때만 씀
  python -m ai.eval.rules_region_extract --refresh  # 검수 CSV를 건드리지 않고 region_condition_review_refresh.csv를 새로 만듦

출력: outputs/07_rules/
- review/region_condition_review.csv   우선(복합·불명확) 전부 + 명확 표본 60건. human 열은 비워 둠
- tables/region_condition_summary.csv  규칙 x 신뢰도 건수, in_excluded_2969 한정 건수, 검수 행 수

주의
- 이 환경에는 LLM이 없으므로 정규식·절 분리 기반 규칙 추출만 씀. LLM 추출이 아님.
- 목적은 제공 지역(사업 주체 위치)과 신청자 거주 요건을 구분해 검수 대상을 좁히는 것이며, 판정 근거가 아님.
- 추출값(ai_*)을 services_master나 ai/rules/*에 되쓰지 않음.
- in_excluded_2969는 현재 ai/rules/region.check_region이 사용자(서울, 노원구)를 제외하는 서비스 여부임.
- 사람 입력이 있는 검수 파일은 덮어쓰지 않음.
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

from rules.region import EXCLUDE, _NOT_APPLICANT, _RESIDENCE, check_region  # noqa: E402

MASTER = ROOT / "data" / "processed" / "services_master.csv"
OUT = ROOT / "outputs" / "07_rules"
REVIEW_PATH = OUT / "review" / "region_condition_review.csv"
SUMMARY_PATH = OUT / "tables" / "region_condition_summary.csv"

TEXT_COLS = ("target_text", "criteria_text", "summary_text")
USER = {"region_sido": "서울특별시", "region_sgg": "노원구"}
SAMPLE_N, SEED = 60, 42
EVIDENCE_MAX = 150

RULES = ["거주 필수", "거주 또는 재직·재학", "신청자 거주 조건 아님", "지역 제한 없음", "거주 조건 불명"]
R_REQ, R_ALT, R_NOT, R_FREE, R_UNK = RULES
CONFS = ["명확", "불명확", "복합"]
C_CLEAR, C_UNCLEAR, C_MIXED = CONFS
GROUP_PRIORITY, GROUP_SAMPLE = "우선", "표본"

HUMAN_COLS = ["region_rule", "region_area", "reason", "reviewer", "reviewed_at"]
REVIEW_COLS = [
    "servId", "servNm", "region_sido", "region_sgg", "ai_region_rule", "ai_region_area",
    "ai_confidence", "evidence", "in_excluded_2969", "review_group",
] + HUMAN_COLS

# ── 1. 절 단위 패턴 ───────────────────────────────────────────
# 절 분리: 줄바꿈, 문장 종결 뒤 공백, 목록 기호
_SPLIT = re.compile(r"[\n\r]+|(?<=[.。])\s+|[※○■▶●◦▪•]|ㅇ\s|(?:^|\s)-\s|(?<=[가-힣.)])-\s?(?=[가-힣(0-9])")
_WORK = r"(?:재직|재학|근무|직장|사업장|소재)"
# 거주 외 경로(재직·재학·전입 예정)로도 받을 수 있음
_ALT = re.compile(
    r"(?:거주|주소|주민등록)[^.]{0,15}(?:또는|혹은|이거나|및)[^.]{0,15}" + _WORK + r"|"
    + _WORK + r"[^.]{0,15}(?:또는|혹은|이거나)[^.]{0,15}(?:거주|주소|주민등록)|"
    r"(?:관내|지역|시|군|구)\s*(?:에\s*)?(?:재직|근무|재학)\s*(?:중인|하는)?\s*(?:자|사람|청년|근로자|학생)|"
    r"전입\s*(?:예정|희망)|이주\s*(?:예정|희망)"
)
# 거주와 무관하게 받을 수 있음
_FREE = re.compile(
    r"거주\s*(?:여부|지|지역)?\s*(?:와|에|에도)?\s*(?:무관|불문|관계\s*없|상관\s*없|제한\s*없)|"
    r"(?:지역|거주지|주소지)\s*(?:제한|구분|무관|불문)\s*(?:없|무관|불문)|지역\s*(?:무관|불문)|"
    r"(?<![거주민]\s)(?<![거주민])누구나\s*(?:이용|참여|신청|방문|참가|가능|수강|상담)|"
    r"전국(?!\s*(?:소재|최초|적|평균|표준|기준|규모|\d|[가-힣]*협회))|"
    r"타\s*(?:지역|시도|시·도)\s*(?:주민|거주|사람|에서)"
)
# 제공 지역만 보이는 표현(신청자 거주 아님)
_PROVIDER = re.compile(r"소재|센터\s*(?:를\s*)?이용|(?:시설|기관)\s*이용|방문\s*상담")

_SIDO_SUFFIX = re.compile(r"(?:특별자치시|특별자치도|특별시|광역시|통합특별시|도)$")


def _split_clauses(text: str) -> list[str]:
    parts = [p.strip() for p in _SPLIT.split(text) if p and p.strip()]
    seen, out = set(), []
    for p in parts:  # 세 열에 같은 문장이 반복되므로 중복 제거
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def build_area_names(master: pd.DataFrame) -> list[str]:
    """자료에 나오는 시도·시군구 이름과 시도 약칭(서울, 경기 등). 긴 이름 우선."""
    names: set[str] = set()
    for col in ("region_sido", "region_sgg"):
        for v in master[col].dropna().unique():
            if len(v) >= 2:
                names.add(v)
            for tok in v.split():
                if len(tok) >= 2:
                    names.add(tok)
    for v in list(names):
        short = _SIDO_SUFFIX.sub("", v)
        if len(short) >= 2 and short != v:
            names.add(short)
    names |= {"서울시", "부산시", "대구시", "인천시", "광주시", "대전시", "울산시"}
    return sorted(names, key=len, reverse=True)


def _find_area(clause: str, m: re.Match, names: list[str]) -> str:
    """일치 위치 앞 25자에서 가장 가까운 지역명, 없으면 뒤 15자의 첫 지역명."""
    before = clause[max(0, m.start() - 25):m.end()]
    best, best_end = "", -1
    for n in names:
        pos = before.rfind(n)
        if pos >= 0 and (pos + len(n) > best_end):  # 끝 위치가 가장 뒤인(가까운) 이름
            best, best_end = n, pos + len(n)
    if best:
        return best
    after = clause[m.end():m.end() + 15]
    first, first_pos = "", 10**6
    for n in names:
        pos = after.find(n)
        if 0 <= pos < first_pos or (pos == first_pos and len(n) > len(first)):
            first, first_pos = n, pos
    return first


def _clip(clause: str, m: re.Match) -> str:
    if len(clause) <= EVIDENCE_MAX:
        return clause
    s = max(0, m.start() - EVIDENCE_MAX // 2)
    return clause[s:s + EVIDENCE_MAX]


def _classify_clause(clause: str):
    """절 하나의 유형과 일치 위치. 유형: ALT, FREE, REQ, NOT, PROV, None."""
    m = _ALT.search(clause)
    if m:
        return "ALT", m
    m = _FREE.search(clause)
    if m:
        return "FREE", m
    hits = list(_RESIDENCE.finditer(clause))
    own = [h for h in hits if not _NOT_APPLICANT.match(clause, h.end())]
    if own:
        return "REQ", own[0]
    if hits:
        return "NOT", hits[0]
    m = _PROVIDER.search(clause)
    if m:
        return "PROV", m
    return None, None


_TYPE_RULE = {"ALT": R_ALT, "FREE": R_FREE, "REQ": R_REQ, "NOT": R_NOT}
_PRIORITY = ["ALT", "FREE", "REQ", "NOT"]  # 복합일 때 대표 규칙 우선순위


def extract_region(svc: dict, names: list[str]) -> dict:
    """서비스 1건의 지역 조건. 반환: ai_region_rule, ai_region_area, evidence, ai_confidence."""
    found: dict[str, tuple[str, re.Match]] = {}  # 유형 -> (절, 일치)
    prov = None
    for col in TEXT_COLS:
        v = svc.get(col)
        if v is None or (isinstance(v, float) and pd.isna(v)):
            continue
        for clause in _split_clauses(str(v)):
            kind, m = _classify_clause(clause)
            if kind == "PROV":
                prov = prov or (clause, m)
            elif kind and kind not in found:
                found[kind] = (clause, m)

    if not found:
        if prov:
            clause, m = prov
            return {"ai_region_rule": R_UNK, "ai_region_area": _area(clause, m, svc, names),
                    "evidence": _clip(clause, m), "ai_confidence": C_UNCLEAR}
        return {"ai_region_rule": R_UNK, "ai_region_area": "", "evidence": "", "ai_confidence": C_UNCLEAR}

    main = next(k for k in _PRIORITY if k in found)
    clause, m = found[main]
    return {
        "ai_region_rule": _TYPE_RULE[main],
        "ai_region_area": _area(clause, m, svc, names),
        "evidence": _clip(clause, m),
        "ai_confidence": C_MIXED if len(found) > 1 else C_CLEAR,
    }


def _area(clause: str, m: re.Match, svc: dict, names: list[str]) -> str:
    """절에 명시된 지역명. 지역명이 없고 '관내'면 서비스 자신의 시군구(없으면 시도)."""
    area = _find_area(clause, m, names)
    if area:
        return area
    if "관내" in clause:
        for k in ("region_sgg", "region_sido"):
            v = svc.get(k)
            if v is not None and not (isinstance(v, float) and pd.isna(v)):
                return str(v)
    return ""


# ── 2. 전체 실행과 산출물 ───────────────────────────────────────
def load_local() -> pd.DataFrame:
    d = pd.read_csv(MASTER, dtype=str, encoding="utf-8-sig", keep_default_na=False, na_values=[""])
    return d[d["source"] == "local"].reset_index(drop=True)


def extract_all(local: pd.DataFrame) -> pd.DataFrame:
    names = build_area_names(local)
    rows = []
    for svc in local.to_dict("records"):
        r = extract_region(svc, names)
        r["in_excluded_2969"] = check_region(svc, USER)[0] == EXCLUDE
        rows.append({"servId": svc["servId"], "servNm": svc["servNm"],
                     "region_sido": svc["region_sido"], "region_sgg": svc["region_sgg"], **r})
    return pd.DataFrame(rows)


def build_review(ext: pd.DataFrame) -> pd.DataFrame:
    exc = ext[ext["in_excluded_2969"]]
    prio = exc[exc["ai_confidence"].isin([C_MIXED, C_UNCLEAR])].copy()
    prio["review_group"] = GROUP_PRIORITY
    clear = exc[exc["ai_confidence"] == C_CLEAR].sort_values("servId")
    samp = clear.sample(n=min(SAMPLE_N, len(clear)), random_state=SEED).copy()
    samp["review_group"] = GROUP_SAMPLE
    prio = prio.sort_values(["ai_confidence", "servId"])
    out = pd.concat([prio, samp.sort_values("servId")], ignore_index=True)
    for c in HUMAN_COLS:
        out[c] = ""
    return out[REVIEW_COLS]


def build_summary(ext: pd.DataFrame, review: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scope, sub in (("전체", ext), ("in_excluded_2969", ext[ext["in_excluded_2969"]])):
        for rule in RULES:
            for conf in CONFS:
                n = int(((sub["ai_region_rule"] == rule) & (sub["ai_confidence"] == conf)).sum())
                rows.append({"section": scope, "ai_region_rule": rule, "ai_confidence": conf, "count": n})
        rows.append({"section": scope, "ai_region_rule": "합계", "ai_confidence": "", "count": len(sub)})
    for g in (GROUP_PRIORITY, GROUP_SAMPLE):
        rows.append({"section": "검수 행", "ai_region_rule": g, "ai_confidence": "",
                     "count": int((review["review_group"] == g).sum())})
    return pd.DataFrame(rows)


def has_human_input(path: Path) -> bool:
    """기존 검수 파일에 사람이 채운 칸이 하나라도 있는지."""
    try:
        d = pd.read_csv(path, dtype=str, encoding="utf-8-sig", keep_default_na=False)
    except Exception:
        return True  # 읽을 수 없으면 보수적으로 덮어쓰지 않음
    cols = [c for c in HUMAN_COLS if c in d.columns]
    return bool(cols) and bool((d[cols].apply(lambda s: s.str.strip()) != "").any().any())


def write_review(review: pd.DataFrame, path: Path = REVIEW_PATH, refresh: bool = False) -> Path | None:
    """검수 CSV 저장. 사람 입력이 있거나 --refresh면 *_refresh.csv로만 씀. 쓴 경로를 반환."""
    path.parent.mkdir(parents=True, exist_ok=True)
    target = path
    if refresh or (path.exists() and has_human_input(path)):
        target = path.with_name(path.stem + "_refresh" + path.suffix)
    review.to_csv(target, index=False, encoding="utf-8-sig")
    return target


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args(argv)
    local = load_local()
    ext = extract_all(local)
    review = build_review(ext)
    summary = build_summary(ext, review)
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(SUMMARY_PATH, index=False, encoding="utf-8-sig")
    print("검수 파일:", write_review(review, REVIEW_PATH, args.refresh))
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
