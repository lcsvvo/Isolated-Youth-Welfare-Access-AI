"""Positive service evidence for an already confirmed relationship request."""
import re
from .benefit_context import DECLINED, offer_clauses, request_clauses

_REQUEST = re.compile(r'대화|소통|교류|말\s*(?:을\s*)?(?:주고받|나누)|얘기|이야기|친구|안부|모임|사람.{0,10}(?:만나|지내)|외로|공허')
_OFFER = re.compile(
    r'소통(?:하고|할|하|을|\s*공간)|대화\s*(?:할|하|공간|모임)|교류\s*(?:활동|프로그램|공간|지원)'
    r'|(?:자조|소)?모임\s*(?:운영|활동|지원|프로그램)|사회적\s*관계\s*(?:회복|형성)\s*지원'
    r'|(?:커뮤니티|동아리|멘토링)\s*(?:지원|활동|운영|프로그램|연계)')
_DENIED = re.compile(r'(?:운영|개설)\s*하지\s*않|미운영')


def relation_support_evidence(query, needs, service, benefit, answers=()):
    if '관계참여' not in {n['domain'] for n in needs}:
        return []
    if not any(_REQUEST.search(c) and not DECLINED.search(c)
               for text in (query, *answers) for c in request_clauses(text)):
        return []
    evidence = []
    for field, passage in [('benefit_text', benefit), ('summary_text', service.get('summary_text'))]:
        if not isinstance(passage, str):
            continue
        for clause in offer_clauses(passage):
            if _OFFER.search(clause) and not _DENIED.search(clause):
                evidence.append({'field': field, 'passage': clause})
    return evidence
