"""4단계 규칙 검증. 명백한 불일치만 '제외', 정보가 없으면 '확인 필요'.

원칙은 팀 합의(3차 회의 '사용자 입력값 라벨링')를 담은 src/recommend/eligibility.py와 같은 의미를 따름.
ai/는 src/를 import하지 않으므로 같은 규칙을 여기서 다시 구현함.
- 사용자 정보가 없으면 확인 필요임. 자격 미충족으로 처리하지 않음
- 서비스 값이 N(원문 근거 없음)이어도 확인 필요임. 조건 없음으로 보지 않음
- 활동범위(외출 단계)는 서비스 외출필요도와 같은 척도가 아니므로 제외 기준으로 쓰지 않음
- 부담과 불가능을 구분함. 필수로 확인된 조건과 사용자 불가가 겹칠 때만 제외 후보임
각 check 함수는 (판정, 이유)를 돌려줌. 판정은 '통과' / '확인 필요' / '제외'.
"""
import pandas as pd

from rules import overlay
from rules.normalize import NATIONWIDE, norm_age, norm_sido, parse_bound
from rules.region import check_region as check_region_sgg
from rules.access import check_access as _check_access, access_penalty
from rules.outing import outing_penalty
from rules.period import PERIOD_NO_INFO_REASON, check_period
from rules.income import check_income
from rules.source_age import corroborated_age_conflict

PASS, CHECK, EXCLUDE = "통과", "확인 필요", "제외"
UNCERTAIN_AGE_BASIS = {"age_may_refer_to_dependent", "conflict", "age_partial"}
VISIT_CONFIRMED_BASIS = {"visit_required_text", "interview_onsite"}


def _val(svc, k):
    v = svc.get(k)
    return None if v is None or (isinstance(v, float) and pd.isna(v)) else v


def _present(v):
    return v is not None and str(v).strip() not in ("", "N", "NA", "nan")


def _age_from_review(age, ov):
    why = f"({ov['reason']})" if ov["reason"] else ""
    if ov["kind"] == "check":
        detail = ov.get("condition") or "기존 연령 추출값 무효화 또는 복수 자격 확인 필요"
        return CHECK, f"검수 반영: {detail}{why}"
    if ov["kind"] == "conditional":
        if ov["min"] <= age <= ov["max"]:
            return PASS, f"검수 반영: 기본 연령 {ov['min']}~{ov['max']}세{why}"
        return CHECK, f"검수 반영: 기본 연령 범위 밖, {ov.get('exception', '예외 확인 필요')}{why}"
    if ov["kind"] == "none":
        return PASS, f"검수 반영: 연령 제한 없음{why}"
    lo, hi = ov["min"] or 0, 200 if ov["max"] is None else ov["max"]
    label = f"{ov['min'] if ov['min'] is not None else ''}~{ov['max'] if ov['max'] is not None else ''}세"
    if lo <= age <= hi:
        return PASS, f"검수 반영: 연령 조건 {label}{why}"
    return EXCLUDE, f"검수 반영: 연령 조건 {label} 밖{why}"


def check_age(svc, profile):
    age = norm_age(profile.get("age"))
    if age is None:
        return CHECK, "사용자 나이 미확인"
    ov = overlay.get_age(svc.get("servId"))   # 검수된 연령 조건이 있으면 추출값보다 우선함
    if ov:
        return _age_from_review(age, ov)
    lk, lv = parse_bound(svc.get("age_min"))
    hk, hv = parse_bound(svc.get("age_max"))
    if lk == "none":
        if profile.get("source_limit_checks") and (reason := corroborated_age_conflict(svc, age)):
            return EXCLUDE, reason
        return CHECK, "서비스 연령 조건 원문 없음"
    lo = 0 if lk == "open" else lv
    hi = 200 if hk != "int" else hv   # 상한이 열려 있거나 정보 없으면 상한 없음으로 봄
    if lo > hi:
        return CHECK, f"서비스 연령 범위 오류({lo}~{hi}세)"
    label = f"{lo if lk == 'int' else ''}~{hi if hk == 'int' else ''}세"
    uncertain = (_val(svc, "youth_scope_basis") in UNCERTAIN_AGE_BASIS
                 or str(_val(svc, "age_review")) == "True")
    if uncertain and profile.get("source_limit_checks") and (reason := corroborated_age_conflict(svc, age)):
        return EXCLUDE, reason
    if not lo <= age <= hi:
        if uncertain:
            return CHECK, f"연령 조건 {label} 밖이나 신청자 연령 여부·다른 자격 경로 확인 필요"
        return EXCLUDE, f"연령 조건 {label} 밖"
    if uncertain:
        return CHECK, f"연령 조건 {label} 안이나 범위가 여러 개라 개별 조건 확인 필요"
    return PASS, f"연령 조건 {label}"


def check_region(svc, profile):
    src, raw = _val(svc, "source"), _val(svc, "region_sido")
    if src == "central":
        return PASS, "중앙부처 사업(전국)"
    svc_set = norm_sido(raw)
    if svc_set == {NATIONWIDE}:
        return PASS, "전국 사업"
    if src is None and raw is None:
        return PASS, "지역 제한 표시 없음"      # sample_services 등 source 열이 없는 표
    user_set = norm_sido(profile.get("region_sido")) - {NATIONWIDE}
    if not user_set:
        return CHECK, "사용자 거주 지역 미확인"
    if not svc_set:
        return CHECK, f"서비스 지역 해석 불가({raw})"
    if not user_set & svc_set:
        return EXCLUDE, f"{raw} 사업"
    if _present(_val(svc, "region_sgg")):
        return CHECK, f"{raw} {_val(svc, 'region_sgg')} 사업, 시군구 미수집"   # 사용자 시군구는 2단계에서 받지 않음
    return PASS, f"{raw} 사업"


def check_outing(svc, profile):
    # 활동범위는 제외 기준이 아님(eligibility.py 원칙). 값이 없으면 판단 보류만 표시.
    # TODO: 외출 대응표 확정 후 순위 조정에만 사용.
    if profile.get("activity_range") in (None, "판단 불가"):
        return CHECK, "활동 범위 미확인"
    return PASS, "활동 범위는 제외 기준이 아님"


# 자격 판정: 제외 또는 확인 필요(감점 1). 지역은 시군구까지 비교(사용자 시군구 없으면 기존과 같음)
CHECKS = [("외출", check_outing), ("연령", check_age), ("지역", check_region_sgg), ("기간", check_period)]
# 접근: 확인된 필수 접근 방식과 사용자 '불가'가 겹칠 때만 제외. 순위는 access_penalty로만 반영(확인 필요 개수에 넣지 않음)
ACCESS_CHECKS = [("접근", _check_access)]
# 정보 표시만: 2단계에서 소득을 받지 않아 거의 모든 서비스가 확인 필요라 감점에 넣지 않음
INFO_CHECKS = [("소득", check_income)]
PENALTY_CHECKS = ("연령", "지역", "기간")


def check_access(svc, profile):
    """rules.access.check_access 재노출(호환용)."""
    return _check_access(svc, profile)


def domain_overlap(svc, needs):
    svc_domains = svc.get("q1_core_domains")
    if svc_domains is None:
        svc_domains = str(svc.get("app_domain", "")).split("|")
    return any(n["domain"] in svc_domains for n in needs)


def _domain_rank(svc, needs):
    """Return the mismatch penalty and the preference for a confirmed main match."""
    if "q1_core_domains" not in svc:
        return (0 if domain_overlap(svc, needs) else 5), 0
    main = [n for n in needs if n.get("level", "main") == "main"] or needs
    if domain_overlap(svc, main):
        return 0, 1
    if any(n["domain"] in svc["q1_unjudged_domains"] for n in main):
        return 0, 0
    return 5, 0


def domain_penalty(svc, needs):
    return _domain_rank(svc, needs)[0]


def apply_rules(candidate_ids, services, extracted, profile=None, excluded=None):
    """후보 판정과 순위 조정.

    - checks: {이름: 판정}, check_reasons: {이름: 이유}. 자격·접근 판정에 '제외'가 있으면 결과에서 뺌
    - excluded에 리스트를 넘기면 제외된 서비스의 {servId, rank, checks, check_reasons}를 추가함
    - score = 검색 순위 + 영역 불일치 5 + 자격 확인 필요 개수 + 접근 감점 + 외출 감점 - Q1 주 욕구 일치 우대 1
    """
    profile = {**extracted, **(profile or {})}
    results = []
    domain_preferences = {}
    for rank, sid in enumerate(candidate_ids):
        svc = services.loc[sid].to_dict()
        svc.setdefault("servId", sid)   # 검수 반영 조회용
        judged = {name: fn(svc, profile) for name, fn in CHECKS + ACCESS_CHECKS + INFO_CHECKS}
        checks = {n: j[0] for n, j in judged.items()}
        reasons = {n: j[1] for n, j in judged.items()}
        info = {n for n, _ in INFO_CHECKS}
        if any(v == EXCLUDE for n, v in checks.items() if n not in info):
            if excluded is not None:
                excluded.append({"servId": sid, "rank": rank, "checks": checks, "check_reasons": reasons})
            continue
        acc_pen, acc_why = access_penalty(svc, profile)
        out_pen, out_why = outing_penalty(svc, profile)
        penalty, preference = _domain_rank(svc, profile.get("needs", []))
        domain_preferences[sid] = preference
        # 약한 감점. 모집기간 원문 없음은 대부분 서비스에 해당해 감점하지 않음(표시만)
        penalty += sum(checks[n] == CHECK and not (n == "기간" and reasons[n] == PERIOD_NO_INFO_REASON)
                       for n in PENALTY_CHECKS)
        penalty += acc_pen + out_pen
        results.append({"servId": sid, "rank": rank, "score": rank + penalty - preference, "checks": checks,
                        "check_reasons": reasons,
                        "penalties": {"접근": acc_pen, "외출": out_pen, "접근_이유": acc_why, "외출_이유": out_why}})
    return sorted(results, key=lambda r: (r["score"], -domain_preferences[r["servId"]]))
