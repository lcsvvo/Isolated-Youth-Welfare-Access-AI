"""Check explicit enrollment and applicant conflicts before recommendation reranking."""

import re

from .filter import EXCLUDE


_OTHER_PERSON = re.compile(
    r"(?:자녀|아들|딸|동생|친구|조카|학생|부모님)(?:은|는|이|가|의|에게|을|를)(?=\s|학교|대학)"
    r"|(?:자녀|아들|딸|동생|친구|조카|학생|부모님)을?\s*위해"
    r"|(?:자녀|아들|딸|동생|친구|조카)\s+(?:대학|학교|고등학교|중학교)"
)
_COLLEGE_NOT_ENROLLED = re.compile(
    r"대학(?:교)?(?:에|을|은|도|는)?\s*(?:(?:아직|결국|그냥|아예)\s*)*"
    r"(?:못\s*갔|안\s*갔|안\s*다니|다니지\s*않|포기했|입학(?:을)?\s*못|입학하지\s*못|가(?:는|기)\s*(?:거|걸)?\s*실패)"
    r"|대학(?:교)?\s*(?:입학\s*)?준비(?:를)?\s*접"
)
_LEFT_SCHOOL = re.compile(
    r"(?:대학교|고등학교|중학교|초등학교|학교|대학)(?:를|을)?\s*(?:그만\s*(?:뒀|둔|두고)|자퇴했|자퇴하고|자퇴한)"
)
_CURRENT_ENROLLMENT = re.compile(
    r"복학|재입학|편입|(?:예전|과거)(?:에|엔)|어릴\s*때|한때"
    r"|(?:대학(?:교|원)?|학교).{0,8}(?:재학\s*중|다니고\s*있|다녀|다닙|입학했|입학해)"
    r"|(?:현재|지금|이번\s*학기).{0,18}(?:재학\s*중|입학했)|입학하면|입학\s*후"
)
_COLLEGE_TARGET = re.compile(
    r"대학(?:교|원)?(?:의)?\s*(?:신규\s*)?(?:입학자|신입생|재학생|재학\s*중)"
    r"|대학(?:교|원)?에\s*(?:입학한|재학\s*중인)"
    r"|대학(?:\(원\))?생|(?:국내\s*)?고등교육기관.{0,12}(?:학부생|대학원생)"
)
_SCHOOL_TARGET = re.compile(
    r"(?:초등학교|중학교|고등학교)\s*\d\s*학년(?:\s*재학생)?"
    r"|(?:초\s*[·ㆍ/]?\s*중\s*[·ㆍ/]?\s*고등학생)|(?:초등|중학|고등)학교\s*재학생"
)
_OTHER_EDUCATION_PATH = re.compile(
    r"또는|혹은|이거나|중\s*하나|입학\s*예정|합격자|비재학생|미재학"
    r"|학교\s*밖|휴학생|재학\s*(?:여부|조건).{0,8}(?:무관|불문|없)|(?:학생|재학생)이\s*아닌"
    r"|(?:학생|재학생)(?:을)?\s*제외|(?:대학생|학생|신입생)\s*자녀"
    r"|미진학자|구직자|취업\s*준비생|졸업생|일반인|(?:대학생|재학생)(?:(?:인|일)\s*경우|\s*등)"
)
_DOCUMENT_OR_HOUSEHOLD_ALTERNATIVE = re.compile(
    r"부\s*(?:또는|혹은)\s*모|감면\s*(?:또는|혹은)\s*면제"
    r"|(?:재학증명서|등록금\s*납부확인서)\s*(?:또는|혹은)\s*"
    r"(?:(?:대학\s*)?등록금\s*납부확인서|재학증명서)"
)
_INSTITUTION_TARGET = re.compile(
    r"시설\s*[·ㆍ,]\s*법인\s*(?:또는|및|[·ㆍ,])\s*단체"
    r"|(?:재학\s*중인\s*)대학(?:원)?을\s*지원"
    r"|독거\s*노인\s*\d+\s*명\s*이상.{0,16}공동으로\s*생활하는\s*시설"
)
_PERSONAL_APPLICATION = re.compile(
    r"개인\s*(?:또는|및|이나|도\s*가능|지원\s*가능)|(?:본인|개인)과\s*(?:시설|기관|단체)"
    r"|(?:개인|본인|학생|이용자)(?:도|이|은)?\s+[^.;]{0,24}(?:신청|접수)"
)
_ON_BEHALF_OF_INSTITUTION = re.compile(
    r"(?:센터|시설|법인|단체|복지관|공동체)(?:를|을)?\s*운영"
    r"|(?:기관|시설|법인|단체|학교|대학|공동체)\s*(?:명의|운영비|대상\s*공모)"
    r"|(?:센터|학교|기관|단체|대학)\s*(?:담당자|대표|운영자)"
)


def _text(value):
    return " ".join(value.split()) if isinstance(value, str) else ""


def institution_support_requested(query):
    return bool(_ON_BEHALF_OF_INSTITUTION.search(query))


def filter_applicant_conflicts(eligible, services, query, excluded=None):
    """Keep candidates with alternate paths or ambiguous user and service conditions."""
    query = _text(query)
    if not query:
        return eligible

    other_person = bool(_OTHER_PERSON.search(query))
    current_enrollment = bool(_CURRENT_ENROLLMENT.search(query))
    left_school = not (other_person or current_enrollment) and bool(_LEFT_SCHOOL.search(query))
    no_college = left_school or (
        not (other_person or current_enrollment) and bool(_COLLEGE_NOT_ENROLLED.search(query))
    )
    individual = not institution_support_requested(query)
    excluded_count = len(excluded) if excluded is not None else 0
    kept = []
    for row in eligible:
        svc = services.loc[row["servId"]]
        target = _text(svc.get("target_text"))
        criteria = _text(svc.get("criteria_text"))
        apply = _text(svc.get("apply_text"))
        conflicts = {}
        if individual and _INSTITUTION_TARGET.search(target) and not _PERSONAL_APPLICATION.search(
            f"{target} {criteria} {apply}"
        ):
            conflicts["신청 주체"] = "지원 대상이 시설·기관·단체로 명시됨"
        # A choice of proof or which parent meets a condition does not provide a separate enrollment route.
        conditions = _DOCUMENT_OR_HOUSEHOLD_ALTERNATIVE.sub(" ", f"{target} {criteria}")
        if (no_college or left_school) and not _OTHER_EDUCATION_PATH.search(conditions):
            if no_college and _COLLEGE_TARGET.search(target):
                conflicts["학적"] = "사용자 요청에 대학 미진학·학업 중단 명시, 서비스는 대학 입학·재학 필요"
            elif left_school and _SCHOOL_TARGET.search(target):
                conflicts["학적"] = "사용자 요청에 학업 중단 명시, 서비스는 재학·해당 학년 필요"

        if not conflicts:
            kept.append(row)
        elif excluded is not None:
            excluded.append({
                "servId": row["servId"], "rank": row["rank"],
                "checks": {**row["checks"], **dict.fromkeys(conflicts, EXCLUDE)},
                "check_reasons": {**row["check_reasons"], **conflicts},
            })
    if excluded is not None and len(excluded) > excluded_count:
        excluded.sort(key=lambda row: row["rank"])
    return kept
