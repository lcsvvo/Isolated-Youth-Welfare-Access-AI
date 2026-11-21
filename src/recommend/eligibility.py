"""사용자별 적격성·제약 대조 (서비스 분류와 분리된 층).

services_master의 youth_scope는 '서비스가 청년을 대상으로 하는가'를 분류할 뿐, 특정 사용자가 신청할 수 있는지는
말하지 않는다. 이 모듈은 사용자가 밝힌 정보와 서비스 라벨을 대조해 항목별로 적합 / 불일치 / 확인 필요를 돌려준다.

원칙 (3차 회의 '사용자 입력값 라벨링', '추천 이유 작성 기준' 7-4)
- 사용자 정보가 미확인·모르겠음·응답 원치 않음이면 '확인 필요'다. 자격 미충족으로 처리하지 않는다.
- 서비스 쪽 값이 N(원문 근거 없음)이어도 '확인 필요'다. 조건 없음으로 보지 않는다.
- '부담'(어려움)과 '불가능'을 구분한다. 불가능한 조건을 서비스가 필수로 요구할 때만 제외한다.
- 사용자 외출 단계(F7 차용 0~3)는 서비스 외출필요도와 같은 척도가 아니므로 제외 기준으로 쓰지 않는다.
- 특정 집단 조건(target_condition)은 제외 기준이 아니라 확인할 조건 목록이다.
"""
from __future__ import annotations

from dataclasses import dataclass, field

OK, MISMATCH, CHECK = "적합", "불일치", "확인 필요"
UNKNOWN = {None, "", "미확인", "모르겠음", "응답 원치 않음"}
CONSTRAINT = {"가능", "조건부 가능", "어려움", "불가능", "미확인", "모르겠음", "응답 원치 않음"}
PRIORITY, CONDITIONAL, EXCLUDE, HOLD = "우선 후보", "조건부 후보", "제외", "판단 보류"
UNCERTAIN_AGE_BASIS = {"age_may_refer_to_dependent", "conflict", "age_partial"}
VISIT_CONFIRMED_BASIS = {"visit_required_text", "interview_onsite"}


@dataclass
class UserProfile:
    age: int | str | None = None                  # 만 나이. 미확인이면 None 또는 상태 문자열
    region_sido: str | None = None
    region_sgg: str | None = None
    outing: str | None = None                     # 외출·이동 제약: CONSTRAINT 중 하나
    cost: str | None = None                       # 비용 지출 제약
    conditions: dict[str, bool] = field(default_factory=dict)   # 예: {"장애": False}. 말하지 않은 조건은 키 없음


def _val(s, k):
    v = s.get(k) if hasattr(s, "get") else getattr(s, k, None)
    return None if v is None or (isinstance(v, float) and v != v) else v


def check_age(service, user: UserProfile) -> tuple[str, str]:
    if user.age in UNKNOWN or not str(user.age).isdigit():
        return CHECK, "나이 미확인"
    lo, hi = _val(service, "age_min"), _val(service, "age_max")
    if lo in (None, "N"):
        return CHECK, "서비스 연령 조건 원문 없음"
    age = int(user.age)
    lo_i = 0 if lo == "NA" else int(lo)
    hi_i = 200 if hi in ("NA", None, "N") else int(hi)
    inside = lo_i <= age <= hi_i
    # 연령이 피부양자(자녀 등)를 가리킬 수 있거나, 근거가 충돌하거나, 연령 밖 다른 자격 경로가 있거나,
    # 범위가 여러 개(검수 표시)면 연령만으로 불일치를 확정하지 않는다
    uncertain = (_val(service, "youth_scope_basis") in UNCERTAIN_AGE_BASIS
                 or str(_val(service, "age_review")) == "True")
    if not inside:
        if uncertain:
            return CHECK, f"연령 조건 {lo}~{hi}세 밖이지만 신청자 연령인지·다른 자격 경로가 있는지 확인 필요"
        return MISMATCH, f"연령 조건 {lo}~{hi}세 밖"
    if uncertain:
        return CHECK, "연령 범위가 여러 개라 개별 조건 확인 필요"     # 전체 범위 안이어도 사이 빈 구간일 수 있다
    return OK, f"연령 조건 {lo}~{hi}세"


def check_region(service, user: UserProfile) -> tuple[str, str]:
    if _val(service, "source") == "central":
        return OK, "중앙부처 사업(전국)"
    sido, sgg = _val(service, "region_sido"), _val(service, "region_sgg")
    if user.region_sido in UNKNOWN:
        return CHECK, "거주 지역 미확인"
    # 지역명은 원자료 표기 그대로 비교한다(행정구역 개편·약칭 정규화는 미구현)
    if sido != user.region_sido:
        return MISMATCH, f"{sido} 사업"
    if sgg is None:
        return OK, f"{sido} 사업"
    if user.region_sgg in UNKNOWN:
        return CHECK, "시군구 미확인"
    return (OK, f"{sgg} 사업") if sgg == user.region_sgg else (MISMATCH, f"{sido} {sgg} 사업")


def check_outing(service, user: UserProfile) -> tuple[str, str]:
    """서비스가 '확인된 최소 외출'(outing_need_min)로 방문(2 이상)을 요구하는지와 사용자 제약을 대조한다."""
    need, need_min = _val(service, "outing_need"), _val(service, "outing_need_min")
    if user.outing in UNKNOWN:
        return CHECK, "외출·이동 가능 범위 미확인"
    if need_min in (None, "N"):
        return CHECK, "서비스 외출 조건 원문 없음"
    use_v, basis = _val(service, "use_outing"), _val(service, "apply_outing_basis")
    visit_confirmed = (use_v not in (None, "N") and int(use_v) >= 2) or basis in VISIT_CONFIRMED_BASIS
    if int(need_min) >= 2 and user.outing == "불가능":
        if visit_confirmed:
            return MISMATCH, "방문이 필요한 것으로 확인됨, 사용자는 외출 불가능"
        # 방문 신청 경로만 안내된 경우: 온라인·대리 경로가 없다는 확인이 아니므로 제외하지 않는다
        return CHECK, "방문 신청만 안내됨 — 온라인·대리 신청 가능 여부 확인 필요"
    if user.outing in ("어려움", "조건부 가능") and (int(need_min) >= 2 or need in (None, "N")):
        return CHECK, "방문이 필요하거나 외출 조건 일부 미확인 — 부담 조건으로 안내"
    if need in (None, "N"):
        return CHECK, "외출 조건 일부만 확인됨"
    return OK, f"외출필요도 {need}"


def check_cost(service, user: UserProfile) -> tuple[str, str]:
    cost = _val(service, "cost")
    if user.cost in UNKNOWN:
        return CHECK, "비용 제약 미확인"
    if cost in (None, "N"):
        return CHECK, "본인 부담 원문 없음"
    if cost == "NA" or cost == "0":
        return OK, "본인 부담 없음 또는 현금형"
    if user.cost == "불가능":
        return MISMATCH, "본인 부담·상환이 있는 서비스, 사용자는 비용 지출 불가능"
    return CHECK, "본인 부담 있음 — 부담 조건으로 안내"


def check_conditions(service, user: UserProfile) -> tuple[str, str]:
    """특정 집단 조건: 사용자가 '해당 없음'이라고 밝혔어도 원문이 '또는' 조건일 수 있으므로 제외하지 않고 확인 필요로 둔다."""
    cond = _val(service, "target_condition")
    if not cond:
        return OK, "특정 집단 조건 표시 없음(규칙 재현율 미검증)"
    names = cond.split("|")
    if all(user.conditions.get(n) is True for n in names):
        return OK, f"조건 해당: {', '.join(names)}"
    return CHECK, f"조건 확인 필요: {', '.join(names)}"


CHECKS = {"연령": check_age, "지역": check_region, "외출": check_outing, "비용": check_cost, "집단 조건": check_conditions}


def assess(service, user: UserProfile, relevant: bool | None) -> dict:
    """후보 판정. relevant = 사용자 필요와 서비스 지원 내용의 관련성(호출 측 판정, None = 판단 불가)."""
    items = {k: f(service, user) for k, f in CHECKS.items()}
    if relevant is None:
        verdict = HOLD
    elif relevant is False or any(v == MISMATCH for v, _ in items.values()):
        verdict = EXCLUDE
    elif any(v == CHECK for v, _ in items.values()):
        verdict = CONDITIONAL
    else:
        verdict = PRIORITY
    return {"verdict": verdict, "items": items,
            "to_confirm": [f"{k}: {why}" for k, (v, why) in items.items() if v == CHECK]}
