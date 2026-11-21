"""검수 반영 레이어. 사람이 검수한 조건을 규칙 추출값보다 먼저 적용함 (services_master는 바꾸지 않음).

검수 파일 (outputs/07_rules/review/, UTF-8-sig CSV). 사람 입력 열이 모두 비어 있으면 미검수로 보고 무시함
- age_recheck_review.csv       servId, candidate_type, extracted_min/max, decision, human_min/max, reason
- region_condition_review.csv  servId, region_rule, region_area, reason
- access_condition_review.csv  servId, apply_visit, apply_alt, use_visit, use_alt, reason

원칙
- 파일이 없거나 비어 있으면 빈 레이어이며 규칙 동작은 바뀌지 않음
- 허용값 밖의 행은 무시하고 warnings에 기록함 (규칙 동작에 영향 없음)
- 경로는 함수 인자 review_dir, 환경변수 RULES_REVIEW_DIR, 기본값 순으로 정함
- 조회는 처음 한 번만 읽고 캐시함. 테스트와 재검증에서는 reset_cache()로 비움
"""
from __future__ import annotations

import csv
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DIR = ROOT / "outputs" / "07_rules" / "review"
ENV_DIR = "RULES_REVIEW_DIR"

AGE_FILE = "age_recheck_review.csv"
REGION_FILE = "region_condition_review.csv"
ACCESS_FILE = "access_condition_review.csv"

AGE_TYPES = ("연령 조건 누락 후보", "연령 제한 없음 후보")
AGE_DECISIONS = ("유지", "수정", "제외", "보류")
REGION_RULES = ("거주 필수", "거주 또는 재직·재학", "신청자 거주 조건 아님", "지역 제한 없음", "거주 조건 불명")
APPLY_VISIT = ("필수", "선택", "불필요", "불명", "서비스별 상이")
APPLY_ALT = ("있음", "없음 명시", "불명", "해당 없음", "서비스별 상이")
USE_VISIT = ("사용자 방문 필수", "담당자 방문", "선택", "불필요", "불명", "서비스별 상이")
USE_ALT = APPLY_ALT

warnings: list[str] = []   # 무시된 행 기록 (reset_cache에서 비움)
_cache: dict = {}
_enabled = True


def set_enabled(flag: bool) -> None:
    """False면 모든 조회가 None을 돌려줌 (검수 반영 전후 비교용)."""
    global _enabled
    _enabled = bool(flag)


def reset_cache() -> None:
    _cache.clear()
    warnings.clear()


def review_dir(path=None) -> Path:
    if path is not None:
        return Path(path)
    env = os.environ.get(ENV_DIR)
    return Path(env) if env else DEFAULT_DIR


def _read(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _s(row, k) -> str:
    return str(row.get(k) or "").strip()


def _int(v):
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return None


def _warn(fname, sid, msg):
    warnings.append(f"{fname} {sid}: {msg}")


def _load_age(d: Path) -> dict:
    out = {}
    for row in _read(d / AGE_FILE):
        sid, dec, ctype = _s(row, "servId"), _s(row, "decision"), _s(row, "candidate_type")
        if not sid or not (dec or _s(row, "human_min") or _s(row, "human_max")):
            continue   # 미검수
        if dec not in AGE_DECISIONS or ctype not in AGE_TYPES:
            _warn(AGE_FILE, sid, f"허용되지 않은 값(decision={dec!r}, candidate_type={ctype!r})")
            continue
        if dec in ("제외", "보류"):
            # 잘못 추출한 후보가 기존 규칙으로 되돌아가지 않도록 명시적으로 확인 필요.
            rec = {"kind": "check", "min": None, "max": None, "reason": _s(row, "reason"),
                   "subject": _s(row, "condition_subject"), "condition": _s(row, "condition_detail")}
            if _s(row, "age_exception"):
                lo, hi = _int(row.get("human_min")), _int(row.get("human_max"))
                if lo is not None and hi is not None and 0 <= lo <= hi <= 120:
                    rec.update(kind="conditional", min=lo, max=hi, exception=_s(row, "age_exception"))
            out[sid] = rec
            continue
        reason = _s(row, "reason")
        if ctype == "연령 제한 없음 후보" and dec == "유지":
            out[sid] = {"kind": "none", "min": None, "max": None, "reason": reason}
            continue
        src = ("human_min", "human_max") if dec == "수정" else ("extracted_min", "extracted_max")
        lo, hi = _int(row.get(src[0])), _int(row.get(src[1]))
        if lo is None and hi is None:
            _warn(AGE_FILE, sid, f"{dec}인데 연령 값이 없음")
            continue
        if lo is not None and hi is not None and lo > hi:
            _warn(AGE_FILE, sid, f"연령 범위 오류({lo}~{hi})")
            continue
        out[sid] = {"kind": "range", "min": lo, "max": hi, "reason": reason}
    return out


def _load_region(d: Path) -> dict:
    out = {}
    for row in _read(d / REGION_FILE):
        sid, rule = _s(row, "servId"), _s(row, "region_rule")
        if not sid or not rule:
            continue
        if rule not in REGION_RULES:
            _warn(REGION_FILE, sid, f"허용되지 않은 region_rule={rule!r}")
            continue
        out[sid] = {"rule": rule, "area": _s(row, "region_area"), "reason": _s(row, "reason"),
                    "subject": _s(row, "condition_subject"), "condition": _s(row, "condition_detail"),
                    "quantifier": _s(row, "condition_quantifier")}
    return out


def _load_access(d: Path) -> dict:
    out = {}
    fields = (("apply_visit", APPLY_VISIT), ("apply_alt", APPLY_ALT), ("use_visit", USE_VISIT), ("use_alt", USE_ALT))
    for row in _read(d / ACCESS_FILE):
        sid = _s(row, "servId")
        if not sid or not any(_s(row, k) for k, _ in fields):
            continue   # 미검수
        bad = [(k, _s(row, k)) for k, allowed in fields if _s(row, k) and _s(row, k) not in allowed]
        if bad:
            _warn(ACCESS_FILE, sid, f"허용되지 않은 값 {bad}")
            continue
        rec = {k: _s(row, k) or "불명" for k, _ in fields}   # 일부만 채운 행의 빈 칸은 불명으로 봄
        rec["reason"] = _s(row, "reason")
        for k in ("receipt_visit", "activity_visit", "apply_route_note"):
            rec[k] = _s(row, k)
        out[sid] = rec
    return out


_LOADERS = {"age": _load_age, "region": _load_region, "access": _load_access}


def _get(kind, sid, review_dir_arg=None):
    if not _enabled or sid is None:
        return None
    d = review_dir(review_dir_arg)
    key = (str(d), kind)
    if key not in _cache:
        _cache[key] = _LOADERS[kind](d)
    return _cache[key].get(str(sid))


def get_age(sid, review_dir=None):
    """{kind: 'range'|'none', min, max, reason} 또는 None."""
    return _get("age", sid, review_dir)


def get_region(sid, review_dir=None):
    """{rule, area, reason} 또는 None."""
    return _get("region", sid, review_dir)


def get_access(sid, review_dir=None):
    """{apply_visit, apply_alt, use_visit, use_alt, reason} 또는 None."""
    return _get("access", sid, review_dir)
