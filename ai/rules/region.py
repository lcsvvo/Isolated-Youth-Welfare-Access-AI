"""지역 규칙 (시도 + 선택 시군구). rules.filter.check_region과 같은 시도 판정에 시군구 비교를 더함.

- 시도 판정은 filter.check_region과 동일함(중앙·전국 통과, 사용자·서비스 시도 불명 확인 필요, 겹침 없음 제외).
- 시도가 겹치고 서비스에 시군구가 있으면: 사용자 시군구가 같으면 통과, 다르면 불일치로 봄, 없으면 확인 필요.
- 불일치(시도·시군구)는 원문에 거주 요건이 명시되고 거주 무관 표현이 없을 때만 제외함. 그 밖에는 확인 필요.
  제공 지역이 곧 거주 요건은 아니므로(누구나 이용 센터, 재직·재학자, 전입 예정자, 지자체의 전국 모집 등) 오제외를 막기 위함.
- 시군구 이름은 시도마다 겹치므로 시도가 겹친 뒤에만 비교함. 통합 표기(전남광주통합특별시)도 이름으로 비교함.
- 사용자 정보가 없으면 확인 필요이며 자격 없음으로 보지 않음(src/recommend/eligibility.py 원칙).
"""
import re

import pandas as pd

from rules import overlay
from rules.normalize import NATIONWIDE, norm_sgg, norm_sido

PASS, CHECK, EXCLUDE = "통과", "확인 필요", "제외"


def _val(svc, k):
    v = svc.get(k)
    return None if v is None or (isinstance(v, float) and pd.isna(v)) else v


# 원문에 거주 요건이 있음을 보이는 표현. 시민은 'OO시민' 형태만 보고, '주민등록상 만 N세'는 나이 기준이라 뺌
_RESIDENCE = re.compile(
    r"거주|주민등록(?!상\s*만)|주소를\s*둔|주소지|관내(?!외)|[가-힣]{1,4}[도시구군]민|전입(?!\s*(?:예정|희망))"
)
# 거주와 무관하게 받을 수 있음을 보이는 표현
_NON_RESIDENCE = re.compile(
    r"거주\s*(?:여부|지|지역)?\s*(?:와|에|에도)?\s*(?:무관|불문|관계\s*없|상관\s*없|제한\s*없)|"
    r"(?:지역|거주지|주소지)\s*(?:제한|구분|무관|불문)\s*(?:없|무관|불문)|지역\s*(?:무관|불문)|"
    r"(?<![거주민]\s)(?<![거주민])누구나\s*(?:이용|참여|신청|방문|참가|가능|수강|상담)|"
    # Nationwide property/duplicate checks do not make applicant residency nationwide.
    r"전국(?!\s*(?:소재|최초|적|평균|표준|기준|규모|무주택|주택|재산|소득|중복|조회|검색))|"
    r"타\s*(?:지역|시도|시·도)\s*(?:주민|거주|사람|에서)|"
    r"(?:거주|주소|주민등록)[^.]{0,15}(?:또는|혹은|이거나)[^.]{0,15}(?:재직|재학|근무|직장|사업장|소재)|"
    r"(?:재직|재학|근무|직장|사업장)[^.]{0,15}(?:또는|혹은|이거나)[^.]{0,15}(?:거주|주소|주민등록)|"
    r"(?:관내|지역|시|군|구)\s*(?:에\s*)?(?:재직|근무|재학)\s*(?:중인|하는)?\s*(?:자|사람|청년|근로자|학생)|"
    r"전입\s*(?:예정|희망)|이주\s*(?:예정|희망)"
)
_TEXT_COLS = ("target_text", "criteria_text", "summary_text")
_RESIDENT_ACCESS_ALTERNATIVE = re.compile(
    r"주민.{0,18}(?:또는|혹은|이거나).{0,20}(?:재직|재학|근무|사업장|전입)"
    r"|(?:재직|재학|근무|사업장|전입).{0,18}(?:또는|혹은|이거나).{0,20}주민"
    r"|주민(?:은|이|을|도)?\s*(?:제외|외의|이외|아닌)"
)
# 거주 표현 바로 뒤에 오면 신청자 본인 거주가 아님(대상 시설·기관·사망자 등)
_NOT_APPLICANT = re.compile(
    r"\s*(?:에\s*)?(?:소재(?:한|하는)?\s*)?(?:사망자|사망|기숙사|시설|업체|사업장|사업체|기관|학교|대학|병원|의료기관|단체|기업|"
    r"어린이집|경로당|복지관|센터|공장|농가|매장|점포|상가|주택(?!에\s*거주))"
)


def residence_required(svc):
    """원문에 거주 요건이 있는지 판정. (True, 근거) 거주 요건 있고 무관 표현 없음, (False, 근거) 무관 표현 있음,
    (None, '') 거주 관련 표현 없음. 무관 표현이 있으면 거주 표현이 함께 있어도 False(확인 필요 쪽으로 보수 처리)."""
    text = " ".join(str(_val(svc, c) or "") for c in _TEXT_COLS)
    non = _NON_RESIDENCE.search(text)
    if non:
        return False, _ctx(text, non)
    hits = list(_RESIDENCE.finditer(text))
    own = [m for m in hits if not _NOT_APPLICANT.match(text, m.end())]
    if own:
        return True, _ctx(text, own[0])
    if hits:
        # '관내 사망자', '관내 소재 기업'처럼 신청자 본인의 거주 조건이 아닌 표현만 있음 → 판단 보류
        return None, _ctx(text, hits[0])
    return None, ""


def _ctx(text, m, w=20):
    return text[max(0, m.start() - w):m.end() + w].replace("\n", " ")


def _region_from_review(ov):
    """검수된 거주 조건을 불일치 판정에 반영. 2단계는 재직지·재학지를 받지 않으므로 대체 요건은 확인 필요."""
    rule = ov["rule"]
    why = f"{rule}" + (f"({ov['area']})" if ov["area"] else "") + (f", {ov['reason']}" if ov["reason"] else "")
    if rule == "거주 필수":
        return EXCLUDE, f"검수 반영: {why}"
    if rule == "지역 제한 없음":
        return PASS, f"검수 반영: {why}"
    return CHECK, f"검수 반영: {why}, 확인 필요"


def _named_resident_required(svc):
    """Recognize a named service area's residents in eligibility text, including spaced names."""
    target = " ".join(str(_val(svc, col) or "") for col in _TEXT_COLS[:2])
    if _RESIDENT_ACCESS_ALTERNATIVE.search(target):
        return False
    names = {_val(svc, "region_sgg"), _val(svc, "region_sido")}
    names.update(norm_sido(_val(svc, "region_sido")) - {NATIONWIDE})
    for name in names:
        if not isinstance(name, str) or not re.fullmatch(r"[가-힣]+", name):
            continue
        pattern = rf"(?<![가-힣]){re.escape(name)}(?:특별자치도|특별시|광역시|시|군|구|도)?\s*주민(?!\s*(?:센터|자치|등록|대표|모임))"
        for match in re.finditer(pattern, target):
            prefix = target[max(0, match.start() - 22):match.start()]
            if not re.search(r"(?:보호자|부모|자녀|가구원|배우자)(?:는|가|이|의)?\s*$", prefix):
                return True
    return False


def _mismatch(svc, label, *, source_limit_checks=False):
    """제공 지역과 사용자 지역이 다를 때: 거주 요건이 명시되면 제외, 아니면 확인 필요."""
    ov = overlay.get_region(_val(svc, "servId"))
    if ov:
        return _region_from_review(ov)
    req, _ = residence_required(svc)
    if req is True or (req is None and source_limit_checks and _named_resident_required(svc)):
        return EXCLUDE, label
    return CHECK, "제공 지역 다름, 거주 요건 원문 확인 필요"


def _review_area(ov, svc, profile):
    """검수된 원문 자격 지역을 제공지역보다 우선. 통합 메타 이름은 확정 자격이 아님."""
    rule, area = ov["rule"], ov["area"]
    detail = ov.get("condition") or ov["reason"]
    why = f"검수 반영: {rule}({area or '지역 미확인'}), {detail}"
    if rule == "지역 제한 없음":
        return PASS, why
    if rule != "거주 필수" or ov.get("subject"):
        return CHECK, why + ", 추가 자격 확인 필요"
    # 기존 검수 데이터에서 광역시/도 + 시군구 또는 광역시/도 단독 사용.
    parts = area.split()
    if not parts:
        return None  # 오래된 검수의 지역 미입력: 기존 비교 유지
    sido = norm_sido(parts[0]) - {NATIONWIDE}
    if not sido:
        return CHECK, why + ", 원문 지역 정규화 필요"
    district = " ".join(parts[1:]) or None
    user = norm_sido(profile.get("region_sido")) - {NATIONWIDE}
    if len(user) != 1:
        return CHECK, why + ", 사용자 거주 지역 확인 필요"
    if not user & sido:
        return EXCLUDE, why
    if len(sido) != 1 and not district:
        # 통합 메타만으로 자격을 확정하지 않는다. 명시된 시군구가 있는 경우만 비교.
        return CHECK, why + ", 통합 표기의 실제 자격 지역 확인 필요"
    if district:
        user_district = norm_sgg(profile.get("region_sgg"))
        if not user_district:
            return CHECK, why + ", 사용자 시군구 미확인"
        return (PASS if user_district == district else EXCLUDE), why
    return PASS, why


def check_region(svc, profile):
    ov = overlay.get_region(_val(svc, "servId"))
    if ov:
        reviewed = _review_area(ov, svc, profile)
        if reviewed is not None:
            return reviewed
    src, raw = _val(svc, "source"), _val(svc, "region_sido")
    if src == "central":
        return PASS, "중앙부처 사업(전국)"
    svc_set = norm_sido(raw)
    if svc_set == {NATIONWIDE}:
        return PASS, "전국 사업"
    if src is None and raw is None:
        return PASS, "지역 제한 표시 없음"
    user_set = norm_sido(profile.get("region_sido")) - {NATIONWIDE}
    if not user_set:
        return CHECK, "사용자 거주 지역 미확인"
    if not svc_set:
        return CHECK, f"서비스 지역 해석 불가({raw})"
    if not user_set & svc_set:
        return _mismatch(svc, f"{raw} 사업", source_limit_checks=bool(profile.get("source_limit_checks")))
    svc_sgg = norm_sgg(_val(svc, "region_sgg"))
    if svc_sgg:
        user_sgg = norm_sgg(profile.get("region_sgg"))
        if user_sgg is None:
            ov = overlay.get_region(_val(svc, "servId"))
            if ov and ov["rule"] == "지역 제한 없음":
                return _region_from_review(ov)
            return CHECK, f"{raw} {svc_sgg} 사업, 사용자 시군구 미확인"
        if user_sgg == svc_sgg:
            return PASS, f"{raw} {svc_sgg} 사업"
        return _mismatch(svc, f"{raw} {svc_sgg} 사업(사용자 {user_sgg})",
                         source_limit_checks=bool(profile.get("source_limit_checks")))
    return PASS, f"{raw} 사업"
