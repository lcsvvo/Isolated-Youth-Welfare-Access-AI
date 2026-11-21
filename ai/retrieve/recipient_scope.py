"""Find explicit special recipient requirements and evidence in a request."""

import re

from .benefit_context import DECLINED, HARDSHIP, REQUEST, offer_clauses, request_clauses


_GROUPS = {
    "민주화운동 관련자·유족": r"민주화\s*운동\s*관련자|민주\s*유공자",
    "참전유공자": r"참전\s*유공자",
    "독립유공자": r"독립\s*유공자",
    "국가유공자": r"국가\s*유공자",
    "보훈 대상자": r"(?:국가\s*)?보훈\s*(?:보상\s*)?대상자",
    "의사상자·유족": r"의사상자|의상자|의사자",
}
_GROUPS = {name: re.compile(pattern) for name, pattern in _GROUPS.items()}
_BROAD_ROUTE = re.compile(
    r"(?:국민|시민|주민|청년|가구|개인)\s*누구나|자격\s*(?:여부\s*)?(?:무관|불문)"
    r"|(?:또는|혹은|,|/|\d+[.)])\s*(?:일반\s*(?:인|청년|주민|가구)|저소득\s*(?:주민|가구)"
    r"|미취업\s*청년|등록\s*장애인|학교\s*밖\s*청소년|다문화\s*(?:가족|가정)\s*청소년"
    r"|북한이탈주민\s*청소년|가족돌봄청년)"
)
_OTHER_PERSON = re.compile(
    r"(?:부모님?|아버지|어머니|배우자|가족|자녀|친구|동생|조부모)(?:은|는|이|가|의)(?=\s|국가|참전|보훈)"
)
_NEGATED = re.compile(r"아니|아닌|아닙|해당\s*없|없(?:는|어요)|미등록|지\s*않|받지\s*못|예정|신청\s*중")
_CONFIRMED = re.compile(
    r"^(?:으?로|이|가|은|는)?\s*(?:입니다|이에요|예요|이고|이어서|이라서|이라|"
    r"(?:등록|인정|선정)(?:됐|되었|받|되|된)|자격(?:이|은)?\s*(?:있|확인))"
)
_NOT_REQUIRED = re.compile(r"^\s*(?:은|는|을|를)?\s*(?:대상에서\s*)?(?:제외|우대|가점|여부\s*무관)")
_EXCLUSION_HEADING = re.compile(r"(?:제외\s*대상|지원\s*제외)\s*(?:[:：·]|$)")
_LAW_TITLE = re.compile(r"「[^」]+」")
_LIVING_REQUEST = re.compile(r"생활비|생계비|생활\s*(?:지원|안정)비")
_LIVING_OFFER = re.compile(r"생활비|생계\s*(?:지원)?비|생활\s*(?:지원|안정)비")
_PAYMENT = re.compile(r"지원|지급|보조|\d\s*만?\s*원")
_PSYCHOLOGICAL_NEED = re.compile(
    r"상담(?!\s*(?:원|사|교사))|(?:심리|정신)\s*(?:치료|검사|건강)|마음\s*건강|불안|우울|트라우마"
)
_PSYCHOLOGICAL_CARE = re.compile(r"(?:심리|정신건강|정서)\s*(?:상담|치료|검사|재활)")


def _field(service, name):
    value = service.get(name)
    return " ".join(value.split()) if isinstance(value, str) else ""


def recipient_evidence(query, answers=()):
    """Only accept an affirmative statement about the requester, including a free answer."""
    found = set()
    for text in (query, *answers):
        for clause in request_clauses(text):
            if _OTHER_PERSON.search(clause) or _NEGATED.search(clause) or DECLINED.search(clause):
                continue
            for name, pattern in _GROUPS.items():
                for m in pattern.finditer(clause):
                    if _CONFIRMED.search(clause[m.end():]):
                        found.add(name)
    return found


def missing_recipient_requirements(service, evidence):
    target = _LAW_TITLE.sub(" ", _field(service, "target_text"))
    target = _EXCLUSION_HEADING.split(target, maxsplit=1)[0]
    conditions = f"{target} {_field(service, 'criteria_text')}"
    if _BROAD_ROUTE.search(conditions):
        return ()
    groups = {name for name, pattern in _GROUPS.items()
              if any(not _NOT_REQUIRED.search(target[m.end():]) for m in pattern.finditer(target))}
    return tuple(sorted(groups)) if groups and not evidence.intersection(groups) else ()


def requests_living_cost(query, answers=()):
    return any(
        _LIVING_REQUEST.search(clause) and not DECLINED.search(clause)
        and (is_answer or REQUEST.search(clause) or HARDSHIP.search(clause))
        for text, is_answer in [(query, False), *((answer, True) for answer in answers)]
        for clause in request_clauses(text)
    )


def offers_living_cost(service, passage):
    title = _field(service, "servNm")
    return any(
        _PAYMENT.search(clause) and (_LIVING_OFFER.search(clause) or _LIVING_OFFER.search(title))
        for clause in offer_clauses(passage)
    )


def matches_psychological_care(query, answers, passage):
    return any(
        _PSYCHOLOGICAL_NEED.search(clause) and not DECLINED.search(clause)
        for text in (query, *answers)
        for clause in request_clauses(text)
    ) and any(_PSYCHOLOGICAL_CARE.search(clause) for clause in offer_clauses(passage))
