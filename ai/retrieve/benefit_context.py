"""Identify requested benefits and their education scope for live reranking."""

import re


BENEFIT_DOMAINS = {
    "노동권 상담·구제": {"노동(일하는 중)"},
    "월세·주거비": {"주거", "경제"},
    "이사비": {"주거"},
    "임차보증금": {"주거", "경제"},
    "교육비": {"교육훈련"},
    "시험 응시료": {"교육훈련"},
    "학습·직업훈련": {"교육훈련"},
    "진로 상담·설계": {"미래설계"},
    "구직 상담·알선": {"미취업(구직)"},
}
SUBJECT_ALIASES = (
    (re.compile(r"(?<![A-Za-z0-9])(?:Auto\s*)?CAD(?![A-Za-z0-9])|캐드", re.I),
     ("CAD", "캐드", "AutoCAD", "오토캐드")),
    (re.compile(r"컴퓨터\s*활용능력|컴활"), ("컴활", "컴퓨터활용능력")),
)
SUBJECTS = tuple(pattern for pattern, _ in SUBJECT_ALIASES) + tuple(
    re.compile(pattern, re.I) for pattern in (
        r"정보\s*처리",
        r"컴퓨터(?!\s*활용능력)|정보기술",
        r"코딩|프로그래밍",
        r"제과",
        r"제빵",
        r"베이킹",
        r"바리스타|커피\s*(?:추출|교육)",
        r"한식|양식|중식|일식|요리|조리(?!원)",
        r"헤어디자인|네일아트|미용",
        r"전기\s*(?:기사|기능사|실습|기술)",
        r"용접",
        r"그래픽\s*디자인|시각디자인|포토샵",
        r"간호조무사",
        r"영어|토익|토플|오픽",
        r"일본어|JLPT",
        r"중국어|HSK",
    )
)
EDU_MODES = {
    "교육비": (
        r"교육비|학원비|수강료|수강비|학습비|교재비|장학금|훈련비",
        r"(?:교육비|학원비|수강료|수강비|학습비|교재비|장학금|훈련비).{0,22}(?:지원|환급|보조|지급|감면)|"
        r"(?:지원|환급|보조|지급|감면).{0,12}(?:수강료|학원비|교육비|교재비)",
    ),
    "시험 응시료": (
        r"응시\s*(?:료|비용)|시험\s*(?:비|비용)",
        r"(?:응시\s*(?:료|비용)|시험\s*(?:비|비용)).{0,22}(?:지원|환급|보조|지급|감면)|"
        r"(?:지원|환급|보조|지급|감면).{0,12}응시\s*(?:료|비용)",
    ),
    "학습·직업훈련": (
        r"배우(?:고|는|려|길)|배워|배울|공부(?:하고|할|하고자)|실습(?!\s*비)|"
        r"(?:수업|강의|강좌|훈련)(?!\s*(?:의\s*)?(?:비|료|비용|수강료))|교육\s*(?:과정|프로그램)",
        r"(?:교육|훈련|강의|강좌|실습|수업)\s*(?:을|를)?\s*(?:제공|지도|운영|실시)|"
        r"(?:교육|훈련|강의|강좌|실습|수업)\s*(?:프로그램|과정)(?!\s*(?:의\s*)?(?:수강료|비용|비\b))",
    ),
}
EDU_MODES = {name: (re.compile(request), re.compile(offer)) for name, (request, offer) in EDU_MODES.items()}
DIRECT_LEARNING = re.compile(r"배우고\s*싶|배워\s*보고\s*싶|배울\s*수|"
                             r"(?:강의|수업|실습|훈련).{0,12}(?:듣고\s*싶|참여하고\s*싶|받고\s*싶|할\s*수)")
DECLINED = re.compile(r"필요\s*(?:없|하지\s*않)|원\s*치\s*않|원하지\s*않|싶(?:지|진)\s*않|"
                      r"이미\s*(?:받|지원받)|관심\s*없|기\s*싫")
REQUEST = re.compile(r"싶(?:어|은|고|다|으면)|시퍼|필요|원(?:해|하|합니다)|알려\s*(?:줘|주)|"
                     r"추천\s*(?:해|받)|어디(?:서|부터)|어떻게|알아\s*(?:보|볼)|있을까|있나|있는지|"
                     r"도움\s*(?:받|을\s*받)|지원\s*(?:을\s*)?받|도와\s*(?:줘|주)|찾고\s*(?:있|싶)")
HARDSHIP = re.compile(r"(?:월세|집세|주거비|임대료|보증금|공과금|생활비|임금|월급).{0,18}"
                      r"(?:밀[렸렷려린]|체납|부족|부조캐|모자라|못\s*(?:내|받)|없)|체불|밀린\s*(?:월세|임금|월급)")
TRANSITION = re.compile(r"(나가(?:는데|는디)|타는데|다니는데|살고\s*있는데|그만두고|쉬다가|했는데|"
                        r"받고\s*있지만|받았지만|받고\s*있는데|"
                        r"필요\s*없(?:고|는데)|원치\s*않고|원하지\s*않고|싶지\s*않고|싫고|말고)\s+")
OFFER_DENIED = re.compile(r"지원\s*(?:하지\s*않|제외|대상\s*아님|불가)|환급\s*(?:불가|없)|"
                           r"제공\s*(?:하지\s*않|불가)|미지원")
CLAUSE_BREAK = re.compile(r"[,，.!?。！？;\r\n]+")
APPROVED_TRAINING_FUNDING = re.compile(
    r"(?:정부|고용노동부)\s*(?:가\s*)?(?:승인|인정)(?:한)?\s*(?:직업\s*)?훈련"
    r"(?:과정)?\s*(?:비|비용).{0,40}(?:지원|보조|지급|감면|환급)"
)


def request_clauses(text):
    return CLAUSE_BREAK.split(TRANSITION.sub(r"\1\n", text))


def requested_benefits(query, needs, answers, benefit_types):
    """Map positive request clauses to confirmed domains and any named subject."""
    domains = {need["domain"] for need in needs}
    requested = {}
    patterns = {**benefit_types, **EDU_MODES}
    for text, is_answer in [(query, False), *((answer, True) for answer in answers)]:
        for clause in request_clauses(text):
            if DECLINED.search(clause):
                continue
            explicit = is_answer or bool(REQUEST.search(clause))
            if not explicit and not HARDSHIP.search(clause):
                continue
            asks_for_cost = any(EDU_MODES[name][0].search(clause) for name in ("교육비", "시험 응시료"))
            for name, (pattern, _) in patterns.items():
                if not domains.intersection(BENEFIT_DOMAINS[name]) or not pattern.search(clause):
                    continue
                if name in EDU_MODES:
                    subjects = {subject for subject in SUBJECTS if subject.search(clause)}
                    explicit_mode = name != "학습·직업훈련" or not asks_for_cost or DIRECT_LEARNING.search(clause)
                    if explicit and subjects and explicit_mode:
                        requested.setdefault(name, set()).update(subjects)
                else:
                    requested.setdefault(name, set())
    return requested


def accepts_training_funding(query, answers):
    """A direct training request can use a funding route unless the user declines it."""
    costs = EDU_MODES["교육비"][0]
    return not any(costs.search(clause) and DECLINED.search(clause)
                   for text in (query, *answers) for clause in request_clauses(text))


def offer_clauses(passage):
    return [" ".join(clause.split()) for clause in re.split(r"[.;。\r\n]+", passage)
            if not OFFER_DENIED.search(clause)]


def offers_approved_training_funding(passage):
    return any(APPROVED_TRAINING_FUNDING.search(clause) for clause in offer_clauses(passage))


def count_supported_benefits(passage, requested, benefit_types):
    """Require a positive offer and, for education, the subject in the same clause."""
    clauses = offer_clauses(passage)
    patterns = {**benefit_types, **EDU_MODES}
    return sum(
        any(patterns[name][1].search(clause) and (not subjects or any(subject.search(clause) for subject in subjects))
            for clause in clauses)
        for name, subjects in requested.items()
    )
