"""Rank eligible candidates by explicitly requested benefits in service source text."""

import re

from .bm25 import build_texts
from .benefit_context import (accepts_training_funding, count_supported_benefits,
                              offers_approved_training_funding, requested_benefits)

WINDOW = 12
TYPE_BONUS = 5

# Each pair describes an explicitly requested action and evidence of providing it.
BENEFITS = {
    "노동권 상담·구제": (
        r"밀린\s*(?:임금|월급)|(?:임금|월급).{0,16}(?:못\s*받|안\s*(?:받|주))|체불|최저(?:임금|시급)|주휴수당|휴게시간|야근",
        r"체불임금|임금.?체불|대지급금|노동권익|노무\s*상담|노동\s*(?:상담|법률)|근로기준법",
    ),
    "월세·주거비": (
        r"월세|집세|주거비|임대료", r"(?:월세|주거비|임차료|임대료).{0,24}(?:지원|보조|감면|지급|원)",
    ),
    "이사비": (
        r"이사(?:비|가|해|할|를)|이주비", r"(?:이사\s*(?:비|비용)|이주비).{0,24}(?:지원|보조|지급|실비|원)",
    ),
    "임차보증금": (
        r"보증금|전세금", r"(?:보증금|전세금).{0,24}(?:지원|보조|대출|융자|지급)",
    ),
    "교육비": (
        r"교육비|학원비|수강료|장학금|훈련비",
        r"교육비|학원비|수강료|수강비|장학금|훈련비|(?:학원|학습지).{0,5}비용",
    ),
    "학습·직업훈련": (
        r"배우|배워|직업훈련|검정고시|기술을|공부.{0,8}다시|다시.{0,8}공부",
        r"(?:직업교육훈련|직업훈련|직무훈련|기술교육|실습교육)|"
        r"(?:교육|훈련|강의|강좌|학습|검정고시|수업)\s*(?:프로그램|과정|제공|지도|운영)",
    ),
    "진로 상담·설계": (
        r"진로|미래.{0,10}(?:생각|고민)|다른\s*길|기술을\s*배울지",
        r"진로.{0,8}(?:상담|지도|설계|코칭|검사|프로그램)|진학\s*(?:상담|지도|코치)",
    ),
    "구직 상담·알선": (
        r"취업|구직|일\s*구하|회사.{0,6}붙|일자리|채용",
        r"취업\s*(?:상담|알선|연계)|구직\s*(?:상담|알선|지원)|구인\s*(?:정보|구직)|채용정보",
    ),
}
BENEFITS = {name: (re.compile(request), re.compile(service)) for name, (request, service) in BENEFITS.items()}



def rank_by_benefit_type(eligible, query, services, *, confirmed_needs=None, need_answers=(), top_n=5):
    """Reorder up to 12 eligible candidates; omit confirmed needs to replay archived scores."""
    if confirmed_needs is None:
        requested = {name for name, (request, _) in BENEFITS.items() if request.search(query)}
    else:
        requested = requested_benefits(query, confirmed_needs, need_answers, BENEFITS)
    if not requested or not eligible:
        return eligible, "|".join(sorted(requested))

    window = eligible[:WINDOW]
    ids = [row["servId"] for row in window]
    field = "benefit_text"
    if confirmed_needs is not None and "benefit_text_raw" in services:
        field = "benefit_text_raw"
    texts = build_texts(services.loc[ids], fields=((field, 1),))
    passages = dict(zip(ids, texts))

    def score(row):
        passage = passages[row["servId"]]
        if confirmed_needs is None:
            normalized = " ".join(passage.split())
            matched = sum(bool(BENEFITS[name][1].search(normalized)) for name in requested)
        else:
            matched = count_supported_benefits(passage, requested, BENEFITS)
        return row["score"] - min(TYPE_BONUS, 3 * matched)

    ranked = sorted(window, key=score)
    if confirmed_needs is not None and "학습·직업훈련" in requested and accepts_training_funding(query, need_answers):
        training = {"학습·직업훈련": requested["학습·직업훈련"]}
        if not any(count_supported_benefits(passage, training, BENEFITS) for passage in passages.values()):
            # A verified funding route can move within the displayed results. It does
            # not establish that the requested course exists or add a new candidate.
            ranked[:top_n] = sorted(ranked[:top_n], key=lambda row: (
                score(row) - 2 * offers_approved_training_funding(passages[row["servId"]])))
    return ranked + eligible[WINDOW:], "|".join(sorted(requested))
