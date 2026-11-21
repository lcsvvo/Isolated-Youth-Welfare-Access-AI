"""Corroborate a mandatory applicant age bound from the two eligibility fields."""

import re


_AGE = re.compile(
    r"(?:만\s*)?(?P<lo>\d{1,3})\s*(?:세\s*(?:이상\s*)?)?[~～∼–-]\s*"
    r"(?:만\s*)?(?P<hi>\d{1,3})\s*세\s*(?P<upper>이하|미만)?"
    r"|(?:만\s*)?(?P<single>\d{1,3})\s*세\s*(?P<cmp>이상|초과|이하|미만)"
    r"|만\s*(?P<exact>\d{1,3})\s*세\s*중(?=\s|[가-힣])"
)
_APPLICANT = re.compile(r"대상자?|수급권자|수급자|어르신|노인|청소년|청년|주민|학생|사람|거주자|^\s*(?:의\s*)?자")
_DEPENDENT = re.compile(r"자녀|손자녀|보호자|부모|배우자|가구원|대리\s*신청")
_AGE_EXCEPTION = re.compile(
    r"나이\s*(?:무관|제한\s*없)|연령\s*(?:무관|제한\s*없|예외)|예외\s*(?:가능|적용|지원)|예외적으로"
    r"|\d+\s*세\s*(?:미만|초과)(?:인\s*사람)?도"
    r"|(?:또는|혹은|및|,|/)\s*(?:등록\s*)?(?:장애인|한부모|일반인|보호자)|다음\s*중\s*하나"
)
_EXCLUDED_GROUP = re.compile(r"(?:제외\s*대상|지원\s*제외|제외\s*조건)\s*[:：]?\s*$")
_EXCLUDED_AFTER_AGE = re.compile(r"^\s*(?:은|는|을|를|인\s*자는)?\s*(?:지원\s*)?(?:제외|불가)")


def _bounds(text):
    intervals = []
    for m in _AGE.finditer(text):
        prefix, suffix = text[max(0, m.start() - 28):m.start()], text[m.end():m.end() + 33]
        if (_DEPENDENT.search(prefix[-16:] + suffix[:12]) or _EXCLUDED_GROUP.search(prefix)
                or _EXCLUDED_AFTER_AGE.search(suffix)):
            return []
        if not _APPLICANT.search(suffix) and not _APPLICANT.search(prefix):
            return []
        if m["lo"]:
            lo, hi = int(m["lo"]), int(m["hi"]) - (m["upper"] == "미만")
        elif m["exact"]:
            lo = hi = int(m["exact"])
        else:
            n = int(m["single"])
            lo, hi = {"이상": (n, 200), "초과": (n + 1, 200), "이하": (0, n), "미만": (0, n - 1)}[m["cmp"]]
        if not 0 <= lo <= hi <= 200:
            return []
        intervals.append((lo, hi))
    return intervals


def corroborated_age_conflict(svc, age):
    """Return an exclusion only when both fields put the user on the same side of every bound."""
    def field(name):
        value = svc.get(name)
        return " ".join(value.split()) if isinstance(value, str) else ""

    target, criteria = field("target_text"), field("criteria_text")
    if _AGE_EXCEPTION.search(f"{target} {criteria}"):
        return None
    target_ranges, criteria_ranges = _bounds(target), _bounds(criteria)
    if not target_ranges or not criteria_ranges:
        return None
    bounds = target_ranges + criteria_ranges
    lower, upper = min(lo for lo, _ in bounds), max(hi for _, hi in bounds)
    if age < lower:
        return f"지원대상·선정기준 원문이 모두 {lower}세 이상을 명시"
    if age > upper:
        return f"지원대상·선정기준 원문이 모두 {upper}세 이하를 명시"
    return None
