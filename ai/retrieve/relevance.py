"""Select up to five recommendations with evidence from the request and service text."""

from .activity_scope import offers_leisure_activity, recovery_target_matches, requests_leisure_activity
from .benefit import BENEFITS, WINDOW
from .benefit_context import count_supported_benefits, requested_benefits
from .bm25 import build_texts
from .funding_scope import funding_matches_request, infrastructure_matches_request
from .recipient_scope import (matches_psychological_care, missing_recipient_requirements, offers_living_cost,
                              recipient_evidence, requests_living_cost)
from .social_support import relation_support_evidence
from rules.filter import CHECK, domain_overlap


MAX_RECOMMENDATIONS = 5
PRIMARY_RANK = 20
CORROBORATING_RANK = 75
RETRIEVED_DEPTH = 100


def select_relevant(eligible, query, services, needs, ranks, limit=MAX_RECOMMENDATIONS, *, need_answers=(), decisions=None):
    """Require a supported benefit plus a type, domain, or strong cross-search match."""
    limit = min(MAX_RECOMMENDATIONS, limit)
    if not eligible or limit <= 0:
        return []

    requested = requested_benefits(query, needs, need_answers, BENEFITS)
    recipient = recipient_evidence(query, need_answers)
    domains = {need["domain"] for need in needs}
    living = "경제" in domains and requests_living_cost(query, need_answers)
    leisure = domains == {"일상·여가"} and requests_leisure_activity(query, need_answers)
    bm25, dense = ranks.get("bm25", []), ranks.get("dense", [])
    agreed = ((set(bm25[:PRIMARY_RANK]) & set(dense[:CORROBORATING_RANK]))
              | (set(dense[:PRIMARY_RANK]) & set(bm25[:CORROBORATING_RANK])))
    retrieved = set(bm25[:RETRIEVED_DEPTH]) | set(dense[:RETRIEVED_DEPTH])
    window = eligible[:WINDOW]
    shortlist = services.loc[[row["servId"] for row in window]]
    field = "benefit_text_raw" if "benefit_text_raw" in shortlist else "benefit_text"
    benefits = build_texts(shortlist, fields=((field, 1),))
    selected, conditional = [], []
    for row, benefit in zip(window, benefits):
        if not benefit.strip():
            if decisions is not None:
                decisions.append({'servId': row['servId'], 'stage': 'relevance_candidate', 'status': 'excluded', 'reason': 'missing_benefit_text',
                    'direct_support': False, 'relation_support_evidence': []})
            continue
        sid = row["servId"]
        service = shortlist.loc[sid].to_dict()
        direct = count_supported_benefits(benefit, requested, BENEFITS)
        social_evidence = relation_support_evidence(query, needs, service, benefit, need_answers)
        direct = bool(direct) or bool(social_evidence)
        def record(status, reason):
            if decisions is not None:
                decisions.append({'servId': sid, 'stage': 'relevance_candidate', 'status': status, 'reason': reason,
                    'direct_support': bool(direct), 'relation_support_evidence': social_evidence})
        if leisure:
            if not offers_leisure_activity(service, benefit):
                record('excluded', 'no_requested_leisure_activity')
                continue
            direct = True
        if not recovery_target_matches(service.get("target_text"), query, need_answers):
            record('excluded', 'targeted_recovery_request_mismatch')
            continue
        same_need = sid in retrieved and domain_overlap(service, needs)
        missing = missing_recipient_requirements(service, recipient)
        if missing:
            direct = (bool(direct) or (living and offers_living_cost(service, benefit))
                      or ("건강" in domains and matches_psychological_care(query, need_answers, benefit)))
            if not direct:
                record('excluded', 'missing_recipient_without_direct_support')
                continue
        if not (direct or same_need or sid in agreed):
            record('excluded', 'no_direct_domain_or_retrieval_evidence')
            continue
        if not funding_matches_request(service, benefit, query, need_answers, direct=bool(direct)):
            record('excluded', 'funding_recipient_or_purpose_mismatch')
            continue
        record('conditional' if missing else 'selected', 'direct_support' if direct else 'domain_or_retrieval_agreement')
        if missing:
            conditional.append({
                **row,
                "checks": {**row.get("checks", {}), "수혜 자격": CHECK},
                "check_reasons": {**row.get("check_reasons", {}),
                                  "수혜 자격": "·".join(missing) + " 해당 여부 확인 필요"},
            })
        else:
            selected.append(row)
        if len(selected) + len(conditional) == limit:
            break
    # Filter the capped set without refilling it with weaker fallback matches.
    passages = {row['servId']: passage for row, passage in zip(window, benefits)}
    result = []
    for row in selected + conditional:
        sid = row['servId']
        if infrastructure_matches_request(shortlist.loc[sid].to_dict(), passages[sid], query, need_answers):
            result.append(row)
        elif decisions is not None:
            decisions.append({'servId': sid, 'stage': 'final_funding_guard', 'status': 'excluded',
                'reason': 'infrastructure_only_benefit_without_personal_service',
                'source_passage': passages[sid], 'direct_support': False, 'relation_support_evidence': []})
    return result
