"""Check the recipient and earmarked use of payments in the service's own text."""

import re

from .benefit_context import DECLINED, HARDSHIP, REQUEST, offer_clauses, request_clauses
from rules.applicant import institution_support_requested


_ORG_COST = re.compile(r"운영\s*(?:경비|비)|인건비|시설\s*(?:설치|개선|유지)\s*(?:비|비용)|운영\s*보조금")
_INFRASTRUCTURE_COST = re.compile(
    r"시설\s*(?:\([^)]*\))?\s*(?:에\s*(?:대한)?|의)?\s*기능\s*보강"
    r"|시설.{0,16}(?:신[/·]?증축|증축|개[/·]?보수|장비\s*구입)")
_ORG = re.compile(r"경로당|시설|복지관|센터|학교|법인|단체|공동체|사업주|사업장|마을|기관")
_ORG_DIRECT_USER = re.compile(
    r"(?:본인\s*부담(?:금|분)|이용료|수강료|보육료).{0,12}(?:경감|감면|환급|대납|지원|면제|없음)"
    r"|(?:개인|가구|가정|청년|학생|이용자|근로자)(?:에게|에)\s*직접.{0,10}(?:지원|지급|제공)"
)
_ORG_PAYEE = re.compile(
    r"(?:시설|기관|복지관|법인|단체|경로당|사업주|사업장)(?:장)?(?:에게|에)\s*"
    r"(?:직접\s*)?(?:보조금|지원금|운영비|인건비).{0,12}(?:지원|보조|지급)"
)
_PERSONAL_TARGET = re.compile(r"개인|가구|가정|청년|학생|이용자|근로자")
_DIRECT_SERVICE = re.compile(
    r"(?:상담|직업\s*훈련|수업|강의|돌봄|방문\s*진료|일자리|취업\s*알선|"
    r"급식|식사|문화\s*활동|모임|프로그램|활동\s*공간|임시\s*거처|심리\s*검사)"
    r"(?:을|를)?\s*(?:무료로\s*)?(?:제공|운영|개방|배달|지원)(?!\s*(?:비용|비\b|경비))"
)
_HOUSEHOLD_COST = (r"생계비|생활(?:지원)?비|주거비|월세|공과금|전기요금|상[·ㆍ/]?하수도요금|난방비|"
                   r"임대보증금|식비|생필품|식료품|의료비|본인\s*부담금")
_HOUSEHOLD_AID = re.compile(
    rf"(?:{_HOUSEHOLD_COST}).{{0,18}}(?:지원|지급|제공|보조|감면|경감|환급)"
)
_HOUSEHOLD_HEADING = re.compile(rf"(?:^|[\r\n]|\b\d+[.)])\s*(?:{_HOUSEHOLD_COST})\s*[:：-]")
_PAYMENT = re.compile(r"지원|보조|지급|환급|감면|경감|실비|\d\s*만?\s*원|이용\s*권")
_TRANSPORT_COST = re.compile(r"교통비|여비|항공료|항공권|운임|차비")
_UNREIMBURSED_TRANSPORT = re.compile(r"(?:교통비|운임|차비).{0,12}(?:자부담|본인\s*부담|지원\s*없)")
_JOB_FUNDING = re.compile(r"구직\s*활동비|취업\s*준비비")
_JOB_REQUEST = re.compile(r"구직|취업|일자리|일을\s*(?:다시\s*)?(?:찾|구)")

# A purpose identifies what the money pays for. It never establishes eligibility.
_TRAVEL_PURPOSES = {
    "가족 방문": r"(?:고국|모국|본국|친정|해외\s*가족).{0,10}(?:방문|갈|가고|가려|다녀)",
    "통학": r"통학|등하교|학교.{0,8}(?:오갈|다닐|가는)",
    "출퇴근": r"출퇴근|출근길|직장.{0,8}(?:오갈|다닐|가는)",
    "병원 이동": r"병원.{0,12}(?:이동|갈|가는|교통|왕복|차비)|진료\s*(?:교통비|여비)|병원비가\s*아닌\s*차비",
}
_FIXED_COSTS = {
    "장례": r"장례비|장제비|화장장\s*이용료|장례\s*비용",
    "영농": r"농자재(?:\s*구입비)?|영농\s*(?:비용|자재)|종자\s*구입비",
}
_TRAVEL_PURPOSES = {name: re.compile(text) for name, text in _TRAVEL_PURPOSES.items()}
_FIXED_COSTS = {name: re.compile(text) for name, text in _FIXED_COSTS.items()}


def _text(value):
    return " ".join(value.split()) if isinstance(value, str) else ""


def infrastructure_matches_request(service, passage, query, need_answers=()):
    """Final guard: facility-only construction cannot stand in for personal aid.

Run after the recommendation cap so rejecting a card does not promote an
unreviewed fallback. Separate personal programs and institution requests stay.
"""
    clauses = offer_clauses(passage)
    if not any(_INFRASTRUCTURE_COST.search(part) for part in clauses):
        return True
    institution_request = any(
        institution_support_requested(part) or _ORG_COST.search(part) or _INFRASTRUCTURE_COST.search(part)
        for text, is_answer in [(query, False), *((a, True) for a in need_answers)]
        for part in request_clauses(text)
        if not DECLINED.search(part) and (is_answer or REQUEST.search(part) or HARDSHIP.search(part))
    )
    return institution_request or any(
        _ORG_DIRECT_USER.search(part) or _HOUSEHOLD_AID.search(part) or _DIRECT_SERVICE.search(part)
        for part in clauses
    )


def _cost_purposes(clause, context):
    if not _PAYMENT.search(clause):
        return set()
    costs = {name for name, pattern in _FIXED_COSTS.items() if pattern.search(clause)}
    if _TRANSPORT_COST.search(clause) and not _UNREIMBURSED_TRANSPORT.search(clause):
        specific = {name for name, pattern in _TRAVEL_PURPOSES.items() if pattern.search(clause)}
        costs |= specific or {name for name, pattern in _TRAVEL_PURPOSES.items() if pattern.search(context)} or {"교통비"}
    return costs


def funding_matches_request(service, passage, query, need_answers=(), *, direct=False):
    """Reject clear cash-purpose or payee mismatches without relying on broad domain tags."""
    clauses = [part for part in offer_clauses(passage) if part]
    if not clauses:
        return False
    context = " ".join(_text(service.get(key)) for key in ("servNm", "target_text", "criteria_text"))
    request = [part for text, is_answer in [(query, False), *((answer, True) for answer in need_answers)]
               for part in request_clauses(text)
               if not DECLINED.search(part) and (is_answer or REQUEST.search(part) or HARDSHIP.search(part))]
    requested = {
        name for name, pattern in {**_TRAVEL_PURPOSES, **_FIXED_COSTS}.items()
        if any(pattern.search(part) for part in request)
    }
    if any(_TRANSPORT_COST.search(part) for part in request):
        requested.add("교통비")
    organization_request = any(institution_support_requested(text) or _ORG_COST.search(text) for text in request)
    household_budget = bool(_HOUSEHOLD_HEADING.search(passage) and _PAYMENT.search(passage))
    job_budget = any(_JOB_REQUEST.search(part) for part in request)

    organization_offer = any(
        _ORG_PAYEE.search(part) or (_ORG_COST.search(part) and _ORG.search(f"{part} {context}"))
        for part in clauses
    )
    if organization_offer and not organization_request:
        # Bills must be for the individual, even when they are paid through the provider.
        personal_target = _PERSONAL_TARGET.search(_text(service.get("target_text")))
        return any(_ORG_DIRECT_USER.search(part) or (direct and _DIRECT_SERVICE.search(part))
                   or (personal_target and _HOUSEHOLD_AID.search(part)) for part in clauses)

    unsupported = False
    for part in clauses:
        purposes = _cost_purposes(part, context)
        if purposes and not requested.intersection(purposes):
            # Separate personal benefits in the same offer keep their own evidence.
            if _HOUSEHOLD_AID.search(part) or _ORG_DIRECT_USER.search(part) or _DIRECT_SERVICE.search(part):
                return True
            unsupported = True
            continue
        if purposes or (organization_offer and organization_request):
            return True
        if (_HOUSEHOLD_AID.search(part) or _ORG_DIRECT_USER.search(part) or _DIRECT_SERVICE.search(part)
                or (job_budget and _JOB_FUNDING.search(part) and _PAYMENT.search(part))):
            return True
    return not unsupported or household_budget
