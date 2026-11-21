"""Build the Q1 recommendation addendum from the reviewed SSIS snapshot."""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from src.gap_index import q1_access as access, q1_final as final, q1_followup as followup, q1_policy as policy
from src.labeling import q1_human as human
from src.preprocess.keywords import APP_DOMAINS_V3, APP_DOMAIN_MAP_V3
from src.preprocess.ssis_master import app_domain_v3
from src.recommend.q3_search import file_sha256

ROOT = human.ROOT
MASTER = ROOT / "data/processed/services_master.csv"
OUT = ROOT / "outputs/03_q1_recommendation"
LABOR_FUTURE_REVIEW = ROOT / "logs/validation/q1_labor_future_review/labor_future_final_review.csv"
LABOR_FUTURE_DOMAINS = ("노동(일하는 중)", "미래설계")
# Same master blob on PR #42 (1697844) and main (ed28066).
REVIEWED_MASTER = "269005564d843cb53988422465e1ba56824341d8a26c3d1b655908ca6ef83d6e"
NO_CORE_MAPPING = tuple(d for d in APP_DOMAINS_V3
                        if d not in {d for _, d, level, *_ in APP_DOMAIN_MAP_V3 if level == "핵심"})
PROBES = {
    "주거": ("청년 월세 지원", "공공 임대주택 보증금"),
    "건강": ("마음건강 심리상담", "정신건강 진료 지원"),
    "일상·여가": ("일상생활 회복 훈련", "혼자 취미 문화 체육"),
    "관계참여": ("청년 커뮤니티 모임", "또래 멘토 상담"),
    "교육훈련": ("직업훈련 수강료 지원", "청년 자격증 응시료"),
    "미취업(구직)": ("청년 일경험 인턴", "구직 취업 상담"),
    "노동(일하는 중)": ("임금 체불 노동 권익", "직장 근로조건 노무상담"),
    "미래설계": ("청년 진로 탐색 설계", "미래 준비 진로 멘토링"),
    "경제": ("청년 생활비 생계 지원", "청년 자립수당 지원"),
}
RANK_PROFILES = (
    {"age": None, "region_sido": None, "activity_range": "집 안"},
    {"age": 25, "region_sido": "서울", "activity_range": "집 안"},
)


def status_for_service(row, labels, tier, decisions):
    """Keep confirmed positives, explicit non-main candidates, and unknowns distinct."""
    mapped = app_domain_v3(row, "|".join(sorted(labels)), tier == "candidate")
    core = set(mapped["app_domain"].split("|")) - {"N"}
    neutral = set(NO_CORE_MAPPING)
    neutral.update(d for q, d, level, *_ in APP_DOMAIN_MAP_V3
                   if level == "핵심" and decisions.get("h_" + q) == "불명")
    if tier in {"candidate", "unknown"}:
        neutral.add("경제")
    return {
        "q1_core_domains": [d for d in APP_DOMAINS_V3 if d in core],
        "q1_unjudged_domains": [d for d in APP_DOMAINS_V3 if d in neutral - core],
        "q1_partial_domains": (mapped["app_domain_partial"] or "").split("|") if mapped["app_domain_partial"] else [],
        "economy_tier": tier,
    }


def load_labor_future_core(path, master_ids):
    """Approved labor/future core domains per servId, read only from `운영_가점_영역` (partial rows are empty)."""
    path = Path(path)
    if not path.exists():
        raise ValueError(f"labor/future review file is missing: {path}")
    df = pd.read_csv(path, encoding="utf-8-sig", dtype=str).fillna("")
    if "servId" not in df or "운영_가점_영역" not in df:
        raise ValueError("labor/future review needs servId and 운영_가점_영역 columns")
    if df.servId.duplicated().any():
        raise ValueError("labor/future review has duplicate servId")
    unknown = set(df.servId) - set(master_ids)
    if unknown:
        raise ValueError(f"labor/future review servId not in master: {sorted(unknown)}")
    result = {}
    for sid, value in zip(df.servId, df["운영_가점_영역"]):
        domains = [d.strip() for d in value.split("|") if d.strip()]
        if set(domains) - set(LABOR_FUTURE_DOMAINS):
            raise ValueError(f"invalid labor/future domain for {sid}: {domains}")
        if domains:
            result[sid] = [d for d in APP_DOMAINS_V3 if d in domains]
    return result


def build(master, labor_future=None, *, apply_final_policy=False):
    gold, trace, log = human.build_gold()
    reviewed = gold.set_index("servId").to_dict("index")
    if not master.servId.is_unique or not set(reviewed) <= set(master.servId):
        raise ValueError("Q1 review IDs do not match the service master")
    auto, events = followup.revise(master)
    # Use the same reference input as the Q1 physical access study, without running or replacing its output tables.
    q1_core = final.overlay(auto, gold, trace)
    policy_rows = {}
    if apply_final_policy:
        adjusted, _, _ = policy.apply(q1_core)
        fields = ["servId", "need_area_q21", "q21_10_tier"]
        policy_rows = adjusted[fields + ["q21_07_beneficiary", "q1_direct_care", "q1_open_space_universal",
                                          "q1_recovery_training_cost", "q1_age_exception_only"]].merge(
            q1_core[fields], on="servId", suffixes=("", "_before"), validate="one_to_one"
        ).set_index("servId").to_dict("index")
        log["classification_policy"] = policy.VERSION
    home, _ = access.read_verification()
    physical = access.final_verification(access.home_verification(q1_core, home),
                                         access.nonhome_verification(q1_core, access.read_nonhome_verification()),
                                         access.read_physical_review()).set_index("servId").fillna("").to_dict("index")
    auto_by_id = auto.set_index("servId").to_dict("index")
    trace_by_id = {sid: g.set_index("label").to_dict("index") for sid, g in trace.groupby("servId")}
    evid_cols = ("verdict", "strict_status", "apply_method", "support_delivery", "outing_required",
                 "eligibility_requirements", "review_evidence", "review_source", "review_reason",
                 "evidence_text", "official_source", "detail_url", "operation_status", "operation_note", "operation_source")
    services = {}
    for _, row in master.iterrows():
        sid = row.servId
        decisions = reviewed.get(sid, {})
        labels = set(str(row.need_area_q21).split("|")) - {"N", "nan"}
        tier = str(row.q21_10_tier)
        # Only the existing direct recovery and education support distinctions; no general-care expansion.
        # Use the economy changes from the same reference (explicit essential costs vs earmarked payments).
        # Carry only q21_02/q21_07 (recovery and education distinctions) and q21_11: the reference adds housing
        # through the explicit_housing_support and room_payment rules, e.g. earmarked room payments that lose
        # economy would otherwise end up with no domain. q21_01/q21_03 differences are intentionally not carried.
        adjusted = auto_by_id[sid]
        reference = set(adjusted["need_area_q21"].split("|"))
        for q in ("q21_02", "q21_07", "q21_11"):
            labels.discard(q)
            labels.update(reference & {q})
        tier = adjusted["q21_10_tier"]
        # Q1 human decisions take precedence; do not promote a non-main candidate based on its existence.
        for field in human.AREA_LABELS:
            if field not in decisions:
                continue
            q = field.removeprefix("h_")
            labels.discard(q)
            if decisions[field] == human.YES:
                labels.add(q)
        if decisions.get("h_q21_10") == human.YES:
            tier = "main"
        elif decisions.get("h_q21_10") == "불명":
            tier = "unknown"
        elif decisions.get("h_q21_10") == human.NO:
            # Preserve an existing candidate even when the binary main label is 'no'.
            tier = "candidate" if row.q21_10_tier == "candidate" else (
                "excluded" if tier == "main" else tier)
        labels.discard("q21_10")
        if tier in {"main", "candidate"}:
            labels.add("q21_10")
        current = policy_rows.get(sid)
        if current is not None:
            old_labels = set(current["need_area_q21_before"].split("|")) - {"N"}
            new_labels = set(current["need_area_q21"].split("|")) - {"N"}
            labels.difference_update(old_labels - new_labels)
            labels.update(new_labels - old_labels)
            if current["q1_open_space_universal"]:
                labels.add("q21_01")
            new_tier = current["q21_10_tier"]
            if pd.notna(new_tier) and str(new_tier) != str(current["q21_10_tier_before"]):
                tier = new_tier
                labels.discard("q21_10")
                if tier in {"main", "candidate"}:
                    labels.add("q21_10")
        edu_neutral = (current is not None and "q21_07" in labels
                       and current["q21_07_beneficiary"] not in policy.SELF_BENEFICIARIES)
        rec = status_for_service(row, labels, tier, decisions)
        if edu_neutral and "교육훈련" not in rec["q1_unjudged_domains"]:
            rec["q1_core_domains"] = [d for d in rec["q1_core_domains"] if d != "교육훈련"]
            rec["q1_unjudged_domains"] = [d for d in APP_DOMAINS_V3
                                            if d == "교육훈련" or d in rec["q1_unjudged_domains"]]
        rec.update(q21_labels=sorted(labels), q21_human_reviewed=bool(decisions))
        if current is not None:
            for tag in ("q1_direct_care", "q1_recovery_training_cost", "q1_age_exception_only"):
                if current[tag]:
                    rec[tag] = True
        # Approved labor/future core candidates (AI-assisted). Q1 human review has no 노동/미래설계 fields
        # (q1_human.AREA_LABELS are q21_xx areas), so human decisions never conflict; the guard below only
        # skips a domain if decisions ever hold an explicit "아니오" under h_<domain>. Unjudged lists stay as is.
        added = [d for d in (labor_future or {}).get(sid, []) if decisions.get("h_" + d) != human.NO]
        if added:
            rec["q1_core_domains"] = [d for d in APP_DOMAINS_V3 if d in set(rec["q1_core_domains"]) | set(added)]
            rec["q1_core_sources"] = {d: "ai_assisted" for d in added}
        access_info = {}
        if decisions:
            review = trace_by_id[sid]
            for phase, field in (("application", human.APPLY), ("use", human.OUTING)):
                proof = review[field]
                access_info[phase] = {
                    "value": decisions[field], "source": proof["source"],
                    "evidence": proof["source_quote"] or proof["primary_evidence"] or proof["second_evidence"],
                    "url": proof["review_url"],
                }
        if sid in physical:
            access_info["physical_review"] = {k: physical[sid][k] for k in evid_cols}
        if access_info:
            rec["access"] = access_info
        services[sid] = rec
    event_cols = {"q21_02", "q21_07", "q21_10"}
    relevant = events.evidence.map(lambda s: any(x.get("action", "").partition(":")[2] in event_cols
                                                or x["rule"] in {"earmarked_payment", "explicit_allowance_candidate",
                                                                 "explicit_essential_cost"}
                                                for x in json.loads(s)))
    return services, events[relevant], log


def compare(master, overlay, bm25, attach, penalty, rules):
    services = master.set_index("servId", drop=False)
    after = attach(services, overlay)
    rows, probes, changes = [], [], []
    candidate_ids = set(master.loc[master.q21_10_tier.eq("candidate"), "servId"])
    unclassified_candidates = set(master.loc[master.q21_10_tier.eq("candidate") & master.app_domain.eq("N"), "servId"])
    for domain in APP_DOMAINS_V3:
        needs = [{"domain": domain}]
        before_core = services.app_domain.fillna("").str.split("|").map(lambda xs: domain in xs)
        after_core = [domain in r["q1_core_domains"] for r in overlay.values()]
        neutral = [domain in r["q1_unjudged_domains"] for r in overlay.values()]
        bpen = {sid for sid, s in services.iterrows() if penalty(s, needs)}
        apen = {sid for sid, s in after.iterrows() if penalty(s, needs)}
        rows.append({"domain": domain, "n_master": len(master), "core_before": int(before_core.sum()),
                     "core_after": sum(after_core), "neutral_after": sum(neutral),
                     "penalized_before": len(bpen), "penalized_after": len(apen),
                     "penalty_removed": len(bpen - apen), "penalty_added": len(apen - bpen),
                     "original_economy_candidates_penalized_before": len(bpen & candidate_ids) if domain == "경제" else "",
                     "original_economy_candidates_penalized_after": len(apen & candidate_ids) if domain == "경제" else "",
                     "original_economy_unclassified_penalized_before": len(bpen & unclassified_candidates) if domain == "경제" else "",
                     "original_economy_unclassified_penalized_after": len(apen & unclassified_candidates) if domain == "경제" else ""})
        for sid in sorted(bpen ^ apen):
            changes.append({"servId": sid, "domain": domain, "penalty_before": 5 if sid in bpen else 0,
                            "penalty_after": 5 if sid in apen else 0, "app_domain_before": services.at[sid, "app_domain"],
                            "core_after": domain in overlay[sid]["q1_core_domains"],
                            "unjudged_after": domain in overlay[sid]["q1_unjudged_domains"],
                            "economy_tier_before": services.at[sid, "q21_10_tier"],
                            "economy_tier_after": overlay[sid]["economy_tier"]})
        for query in PROBES[domain]:
            ids = bm25.search(query, 100)
            for i, profile in enumerate(RANK_PROFILES, 1):
                before_rank = [r["servId"] for r in rules(ids, services, {"needs": needs, **profile})]
                after_rank = [r["servId"] for r in rules(ids, after, {"needs": needs, **profile})]
                probes.append({"domain": domain, "query": query, "profile": i, "bm25_top100": len(ids),
                               "bm25_top10_changed": False, "eligible_count": len(before_rank),
                               "penalized_before": len(set(before_rank) & bpen), "penalized_after": len(set(after_rank) & apen),
                               "top5_order_changed": before_rank[:5] != after_rank[:5],
                               "top5_common": len(set(before_rank[:5]) & set(after_rank[:5])),
                               "top5_before": "|".join(before_rank[:5]), "top5_after": "|".join(after_rank[:5])})
                if set(before_rank) != set(after_rank):
                    raise ValueError("Q1 rank addendum changed the eligibility decisions")
    return pd.DataFrame(rows), pd.DataFrame(probes), pd.DataFrame(changes)


def main(out=OUT):
    master_hash = file_sha256(MASTER)
    if master_hash != REVIEWED_MASTER:
        raise ValueError("Revalidate Q1 evidence before applying it to a different master snapshot")
    sys.path.insert(0, str(ROOT / "ai"))
    from core.data import load_services
    from core.review_data import attach_q1
    from retrieve.bm25 import BM25Index, build_texts
    from rules.filter import apply_rules, domain_penalty

    master = load_services(MASTER)
    labor_future = load_labor_future_core(LABOR_FUTURE_REVIEW, master.servId)
    overlay, events, trace_log = build(master, labor_future, apply_final_policy=True)
    texts = build_texts(master)
    if texts != build_texts(attach_q1(master.set_index("servId", drop=False), overlay)):
        raise ValueError("Q1: rank metadata changed the search documents")
    bm25 = BM25Index(master.servId, texts, cache_path=ROOT / "private/q1_recommendation_bm25.pkl")
    summary, probes, changes = compare(master, overlay, bm25, attach_q1, domain_penalty, apply_rules)
    out.mkdir(parents=True, exist_ok=True)
    tables = out / "tables"
    tables.mkdir(exist_ok=True)
    for name, df in {"domain_comparison": summary, "rank_comparison": probes, "penalty_changes": changes,
                     "reference_rule_events": events}.items():
        df.to_csv(tables / f"{name}.csv", **human.CSV)
    sources = list(access.VERIFY.glob("*.csv"))
    review_path = out / "q1_review.json"
    baseline_path = out / "q1_review_before_final_policy.json"
    if review_path.exists() and not baseline_path.exists():
        baseline_path.write_bytes(review_path.read_bytes())
    review_path.write_text(json.dumps({
        "schema": "q1-review-v1", "master_sha256": master_hash,
        "q1_gold_build": trace_log, "labor_future_review_sha256": file_sha256(LABOR_FUTURE_REVIEW), "access_inputs_sha256": {p.name: file_sha256(p) for p in sorted(sources)},
        "services": overlay,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    main(parser.parse_args().out)
