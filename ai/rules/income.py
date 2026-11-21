"""소득 조건 규칙 검증. 정보가 없으면 '확인 필요'이며 이 버전은 '제외'를 내지 않음.

원칙은 src/recommend/eligibility.py와 같음. 사용자 정보 없음과 서비스 원문 없음(N)은 확인 필요이고 자격 미충족으로 보지 않음.

범위 결정 (data/processed/services_master.csv 5,288행 실측)
- elig_income는 소득 기준의 유무 라벨만 있음 (있음 1,901 / 없음 23 / N 3,364). 임계값 필드는 없음
- elig_income_evidence에서 '중위소득 ... N%'가 추출되는 행 527개, criteria_text·target_text까지 넓히면 671개.
  기초생활수급·수급자·차상위 키워드가 나오는 행 1,140개
- 따라서 임계값을 파싱하는 `parse_income_threshold`는 제공하되 일부 행에만 값이 나옴
- 2단계는 사용자 소득을 받지 않으므로 profile["income"]이 없으면 소득 조건 있음·N은 모두 확인 필요가 됨

판정 (profile["income"] = {"median_pct": int|None, "status": "수급"|"차상위"|"해당 없음"|None})
- elig_income 없음 → 통과 (소득 조건 없음이 원문에 명시됨)
- 사용자 소득 정보 없음 → 확인 필요
- 있음 + 사용자 status가 수급/차상위이고 서비스 원문에 같은 키워드 → 확인 필요(원문이 "수급자 제외"처럼 반대 뜻일 수 있어 통과로 보지 않음)
- 있음 + 임계값 파싱됨 + 사용자 median_pct 있음 → 이하이면 통과, 초과이면 확인 필요
- 그 외(N 포함) → 확인 필요
- 임계값 초과여도 제외하지 않음. 소득 기준은 수급자·차상위·특정 집단 등 대안 자격이나 가구원 수·소득인정액 산정
  차이가 흔해서 단일 임계값만으로 불가능을 확정할 수 없음. 완전히 모호하지 않은 제외 사례를 찾지 못해 이 버전은 제외 없음
"""
from __future__ import annotations

import re

PASS, CHECK, EXCLUDE = "통과", "확인 필요", "제외"

_MEDIAN = re.compile(r"중위\s*소득")
_PCT_BELOW = re.compile(r"(\d{2,3})\s*%\s*이하")
_PCT_ANY = re.compile(r"(\d{2,3})\s*%")
_FRACTION = re.compile(r"(\d)\s*분의\s*(\d)")
_STATUS_KEYS = {"수급": re.compile(r"기초생활|수급자|수급권자|의료급여"), "차상위": re.compile(r"차상위")}


def _s(v) -> str:
    return v if isinstance(v, str) else ""


def parse_income_threshold(svc: dict) -> tuple[int | None, str]:
    """중위소득 상한 비율(%)과 근거 문장을 돌려줌. 없으면 (None, "").

    elig_income_evidence, criteria_text, target_text 순으로 첫 매치를 씀.
    '이하'가 붙은 비율이 여럿이면 가장 큰 값(가장 느슨한 상한)을 씀. '2분의 1'은 50으로 환산함.
    """
    for key in ("elig_income_evidence", "criteria_text", "target_text"):
        text = _s(svc.get(key))
        for m in _MEDIAN.finditer(text):
            seg = text[m.start(): m.start() + 120]
            below = [int(x) for x in _PCT_BELOW.findall(seg)]
            if below:
                return max(below), seg[:80]
            anyp = _PCT_ANY.search(seg)
            if anyp:
                return int(anyp.group(1)), seg[:80]
            fr = _FRACTION.search(seg[:30])
            if fr and int(fr.group(1)):
                return round(int(fr.group(2)) / int(fr.group(1)) * 100), seg[:80]
    return None, ""


def _status_hit(svc: dict, status: str) -> bool:
    pat = _STATUS_KEYS.get(status)
    if pat is None:
        return False
    text = " ".join(_s(svc.get(k)) for k in ("elig_income_evidence", "criteria_text", "target_text"))
    return bool(pat.search(text))


def check_income(svc: dict, profile: dict) -> tuple[str, str]:
    elig = _s(svc.get("elig_income")) or "N"
    if elig == "없음":
        return PASS, "소득 조건 없음"
    if elig != "있음":
        return CHECK, "소득 조건 원문 없음, 확인 필요"
    inc = profile.get("income") or {}
    pct, status = inc.get("median_pct"), inc.get("status")
    if pct is None and status in (None, ""):
        return CHECK, "사용자 소득 정보 없음, 소득 조건 확인 필요"
    if status in _STATUS_KEYS and _status_hit(svc, status):
        return CHECK, f"{status} 관련 조건 언급, 대상·제외 여부 확인 필요"
    thr, _ = parse_income_threshold(svc)
    if thr is not None and pct is not None:
        if pct <= thr:
            return PASS, f"기준중위소득 {thr}% 이하 조건 충족"
        return CHECK, f"기준중위소득 {thr}% 초과, 대안 자격 여부 확인 필요"
    return CHECK, "소득 조건 임계값 확인 필요"
