"""Compare a specific leisure request and narrowly targeted recovery services."""

import re

from .benefit_context import DECLINED, REQUEST, offer_clauses, request_clauses


_LEISURE = re.compile(r"취미|동호회|동아리|문화\s*(?:활동|수업)|여가\s*(?:활동|수업)")
_ACTIVITY = re.compile(
    r"(?:취미|문화|여가|예술|공예|체육)\s*(?:교실|수업|강좌|강습|교육|워크숍|프로그램)"
    r"|(?:동아리|동호회|소모임)\s*(?:모집|참여|운영|활동|지원|가입)"
    r"|취미\s*활동비.{0,14}(?:지원|지급)"
)
_VOUCHER = re.compile(r"바우처|지역화폐|수강료|이용권")
_LEISURE_USE = re.compile(r"(?:여가|문화|예술|스포츠|체육)\s*활동|문화생활|문화시설|체육시설|스포츠센터|운동\s*관련\s*학원")
_RESIDENTIAL_ONLY = re.compile(r"입소(?:가|이)?\s*(?:결정|승인)된|입소한\s*(?:사람|노인|장애인|경우)")
_TARGETED_RECOVERY = re.compile(
    r"^\s*(?:[○ㅇ□•-]\s*)?(성폭력|가정폭력|교제폭력|스토킹)\s*피해자(?:를|가|는|에게|에\s*한해|\s*대상|$)"
)
_ALTERNATIVE = re.compile(r"누구나|또는|그\s*외|외에도|일반\s*(?:인|주민|시민|청년)")


def requests_leisure_activity(query, answers=()):
    for text in (query, *answers):
        clauses = [part for part in request_clauses(text) if not DECLINED.search(part)]
        if any(_LEISURE.search(part) for part in clauses) and any(REQUEST.search(part) for part in clauses):
            return True
    return False


def offers_leisure_activity(service, passage):
    target = service.get("target_text")
    if isinstance(target, str) and _RESIDENTIAL_ONLY.search(target):
        return False
    clauses = offer_clauses(passage)
    if any(_ACTIVITY.search(clause) for clause in clauses):
        return True
    summary = service.get("summary_text")
    return any(_VOUCHER.search(clause) for clause in clauses) and (
        any(_LEISURE_USE.search(clause) for clause in clauses)
        or (isinstance(summary, str) and any(_LEISURE_USE.search(clause) for clause in offer_clauses(summary)))
    )


def recovery_target_matches(target, query, answers=()):
    if not isinstance(target, str) or _ALTERNATIVE.search(target):
        return True
    match = _TARGETED_RECOVERY.search(target)
    return not match or any(
        match[1] in clause and not DECLINED.search(clause)
        for text in (query, *answers) for clause in request_clauses(text)
    )
