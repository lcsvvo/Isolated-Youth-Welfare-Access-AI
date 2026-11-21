"""Compare retrieval and ranking changes on the separate 30-card development pool."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

import pandas as pd
import yaml

from ai.eval import q3_top5_audit as audit
from ai.eval import independent_retrieval as q3
from ai.scripts import rrf_weight_compare as rw
from eval.stage2_query_review import dev_cases
from retrieve.bm25 import BM25Index, build_texts
from rules.filter import apply_rules
from rules.normalize import build_sgg_map, use_sgg_map, valid_sgg_for


BENEFIT_CANDIDATE_DEPTH = 20
BENEFIT_RERANK_DEPTH = 10
BENEFIT_MAX_BONUS = 5
LEARNING_NEEDS = {"교육훈련", "미래설계"}
EDUCATION_COST = re.compile(r"교육비|학원비|학원[·ㆍ/\s]*(?:학습지)?\s*비용|수강료|수강비|훈련비|장학금")
DIRECT_LEARNING_SUPPORT = re.compile(
    r"(?:교육|훈련|강좌|수업|상담|멘토링|학습)\s*(?:프로그램|과정|서비스)?\s*(?:제공|운영|실시|진행)"
)
FINANCIAL_REQUEST = re.compile(r"비용|교육비|학원비|수강료|장학금|돈|무료|저렴|비싸")


def benefit_rerank(kept, benefit_order):
    """Add a bounded bonus for the strongest benefit-text matches among eligible candidates."""
    eligible = {row["servId"] for row in kept}
    best = [sid for sid in benefit_order if sid in eligible][:BENEFIT_RERANK_DEPTH]
    bonus = {sid: BENEFIT_MAX_BONUS * (BENEFIT_RERANK_DEPTH - i) / BENEFIT_RERANK_DEPTH
             for i, sid in enumerate(best)}
    return sorted(kept, key=lambda row: row["score"] - bonus.get(row["servId"], 0))


def write_review_form(path, pairs, services, texts):
    if not pairs:
        return
    arms = {card: {"unjudged": sorted(ids)} for card, ids in pairs.items()}
    _, form = q3.build_pool(arms, services, texts, depth=max(map(len, pairs.values())))
    form.assign(**dict.fromkeys(q3.independent.RELEVANCE_COLUMNS[1:], "")).to_csv(path, **q3.CSV)


def hold_cost_only_swaps(before, trial, services, query, needs):
    """For a learning request, hold the original slots when a new service only pays tuition."""
    if not any(n["domain"] in LEARNING_NEEDS for n in needs) or FINANCIAL_REQUEST.search(query):
        return trial
    blocked = set()
    for sid in set(trial) - set(before):
        benefit = str(services.at[sid, "benefit_text"])
        if EDUCATION_COST.search(benefit) and not DIRECT_LEARNING_SUPPORT.search(benefit):
            blocked.add(sid)
    displaced = iter(sid for sid in before if sid not in trial)
    return [next(displaced) if sid in blocked else sid for sid in trial]


def reviewed_change(arms, human_review, grades):
    """Merge the judged swaps by blind key. Shared unknown slots cancel in the change."""
    expected = {}
    for card, (before, trial, _) in arms.items():
        for sid in (set(before) ^ set(trial)) - grades[card].keys():
            key = hashlib.sha256(f"q3-v1:{card}:{sid}".encode()).hexdigest()[:20]
            expected[key] = (card, sid)
    reviews = q3.independent.read_csv(human_review)
    if (not reviews.pool_id.is_unique or set(reviews.pool_id) != set(expected)
            or not reviews.judge_type.eq("human").all() or not reviews.relevance.isin(["0", "1", "2"]).all()):
        raise ValueError("Expected human 0/1/2 judgments for every changed unjudged pair")
    for p, card, grade in reviews[["pool_id", "card_id", "relevance"]].itertuples(index=False):
        expected_card, sid = expected[p]
        if card != expected_card:
            raise ValueError("Review card does not match the swap")
        grades[card][sid] = int(grade)
    rows = []
    for card, (before, trial, gated) in arms.items():
        marks = grades[card]
        hits = [sum(marks.get(sid) == 2 for sid in selected) for selected in (before, trial, gated)]
        rows.append({
            "query_id": card, "cost_only_swaps_blocked": len(set(trial) - set(gated)),
            "baseline_known_hits": hits[0], "trial_known_hits": hits[1], "gated_known_hits": hits[2],
            "gated_delta_hits": hits[2] - hits[0],
        })
    return pd.DataFrame(rows)


def dev_rerank_cases(root, services, master, meta):
    """Yield frozen retrievals, rule views and checked baseline orders for development."""
    archive = root / "outputs/03_q1_recommendation/tables/final_policy_top5.csv"
    baseline = pd.read_csv(archive, keep_default_na=False)
    baseline = baseline.loc[baseline.operational_bonus].set_index("query_id").top5_final.to_dict()
    with use_sgg_map(build_sgg_map(services, master)):
        for source in meta["sources"]:
            colab = root / source["colab"].replace("\\", "/")
            _, saved, _ = rw.load_colab(colab)
            cfg = yaml.safe_load((colab / "runs/config_core.yaml").read_text(encoding="utf-8"))
            query_meta = q3.independent.read_csv(colab / "tables/queries.csv")
            access = query_meta[query_meta.profile.eq("core")].set_index("query_id").access.to_dict()
            corpus = rw.read_jsonl(root / "ai/data/dev_corpus.jsonl")
            if extra := source.get("extra_corpus"):
                corpus += rw.read_jsonl(root / extra.replace("\\", "/"))
            cases = {f"dev:{c['case_id']}": c for c in dev_cases(corpus)}
            for card in source["queries"]:
                ranks = {m: saved[card, m] for m in ("bm25", "dense", "fused")}
                ranks["rrf"] = ranks.pop("fused")
                case = cases[card]
                view = q3.rules_view(case["session"]["confirmed_profile"], cfg, json.loads(access[card]))
                view["region_sgg"] = valid_sgg_for(view.get("region_sido"), view.get("region_sgg"))
                view["today"] = meta["as_of"]
                kept = apply_rules(ranks["rrf"], services, view)
                top5 = [r["servId"] for r in kept[:audit.K]]
                if "|".join(top5) != baseline.get(card):
                    raise ValueError(f"Development baseline does not replay for {card}")
                yield card, case["text"], view, ranks, kept


def run(root, out, *, benefit_text=False, reviewed_precision=None):
    combined = root / "outputs/11_rrf_weights/combined30/tables/review.csv"
    meta = audit.read_json(root / "outputs/12_q1_bonus/run1/runs/metadata.json")
    if q3.file_sha256(combined) != meta["review_sha256"]:
        raise ValueError("Development review differs from the archived run")
    judged = q3.independent.read_csv(combined)
    grades = {card: dict(zip(g.servId, g.relevance_0_1_2.astype(int)))
              for card, g in judged.groupby("query_id")}
    if len(grades) != 30 or not all(card.startswith("dev:") for card in grades):
        raise ValueError("Expected the separate 30 development queries")

    q1 = root / "outputs/03_q1_recommendation"
    master = root / "data/processed/services_master.csv"
    services = q3.RecommendationReview(master, q1_path=q1 / "q1_review.json").services
    benefit_index = None
    if benefit_text:
        documents = build_texts(services, fields=(("benefit_text", 1),))
        benefit_index = BM25Index(services.index, documents)
    rankings, rows, texts, review_arms = {}, [], {}, {}
    recall_review, precision_review = {}, {}
    for card, text, view, ranks, kept in dev_rerank_cases(root, services, master, meta):
        needs = view["needs"]
        if card not in grades or card in rankings:
            raise ValueError(f"Unexpected or duplicate development query: {card}")
        rankings[card] = ranks
        a = [r["servId"] for r in kept[:audit.K]]
        marks = grades[card]
        recall_stats = {}
        if benefit_index is not None:
            texts[card] = text
            benefit_order = benefit_index.search(text, k=len(services))
            added = set(benefit_order[:BENEFIT_CANDIDATE_DEPTH]) - set(ranks["rrf"])
            recall_review[card] = added - marks.keys()
            recall_stats = {
                "added_candidate_pairs": len(added),
                "added_known_relevant": sum(marks.get(sid) == 2 for sid in added),
                "added_unjudged": len(recall_review[card]),
            }
            trial = benefit_rerank(kept, benefit_order)
        else:
            # Remove only the soft domain mismatch penalty for the original comparison.
            def without_domain_penalty(row):
                svc = services.loc[row["servId"]].to_dict()
                penalty, preference = q3._domain_rank(svc, needs)
                return row["score"] - penalty, -preference, row["rank"]

            trial = sorted(kept, key=without_domain_penalty)
        b = [r["servId"] for r in trial[:audit.K]]
        if reviewed_precision is not None:
            gated = hold_cost_only_swaps(a, b, services, text, needs)
            review_arms[card] = (a, b, gated)
        pa, pb = [{sid for sid in ids if marks.get(sid) == 2} for ids in (a, b)]
        ua, ub = [set(ids) - marks.keys() for ids in (a, b)]
        precision_review[card] = ua ^ ub
        delta = len(pb) - len(pa)
        rows.append({
            "query_id": card, "membership_changed": set(a) != set(b),
            "baseline_known_hits": len(pa), "trial_known_hits": len(pb),
            "baseline_unjudged": len(ua), "trial_unjudged": len(ub),
            "delta_hits_lower": delta - len(ua - ub), "delta_hits_upper": delta + len(ub - ua),
            **recall_stats,
        })
    if rankings.keys() != grades.keys():
        raise ValueError("Not all 30 reviewed queries had saved ranks")
    gated_result = reviewed_change(review_arms, reviewed_precision, grades) if reviewed_precision is not None else None
    tables = out / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    if gated_result is not None:
        gated_result.to_csv(tables / "dev_benefit_cost_gate.csv", **q3.CSV)
        outcomes = Counter(gated_result.gated_delta_hits.map(lambda n: (n > 0) - (n < 0)))
        print(gated_result.loc[gated_result.cost_only_swaps_blocked.gt(0)].to_string(index=False))
        print(f"Gated comparison: {gated_result.gated_delta_hits.sum() / (len(gated_result) * audit.K):+.6f}; "
              f"improved {outcomes[1]}, worsened {outcomes[-1]}, unchanged {outcomes[0]}")
        return
    if not benefit_text:
        candidates = audit.coverage(rankings, grades, depths=(20, 100), include_rescue=True)
        candidates.to_csv(tables / "dev_candidate_coverage.csv", **q3.CSV)
        print(candidates.to_string(index=False))
    precision = pd.DataFrame(rows)
    label = "dev_benefit" if benefit_text else "dev_precision"
    precision.to_csv(tables / f"{label}_by_query.csv", **q3.CSV)
    sums = precision.sum(numeric_only=True)
    slots = len(rows) * audit.K
    summary = {
        "queries": len(rows), "changed_membership": int(sums.membership_changed),
        "baseline_p5_lower": sums.baseline_known_hits / slots,
        "baseline_p5_upper": (sums.baseline_known_hits + sums.baseline_unjudged) / slots,
        "trial_p5_lower": sums.trial_known_hits / slots,
        "trial_p5_upper": (sums.trial_known_hits + sums.trial_unjudged) / slots,
        "delta_p5_lower": sums.delta_hits_lower / slots, "delta_p5_upper": sums.delta_hits_upper / slots,
    }
    if benefit_text:
        summary.update({key: int(sums[key]) for key in recall_stats})
        summary["changed_top5_unjudged"] = sum(map(len, precision_review.values()))
        review = out / "review"
        review.mkdir(exist_ok=True)
        write_review_form(review / "dev_benefit_precision.csv", precision_review, services, texts)
        write_review_form(review / "dev_benefit_recall.csv", recall_review, services, texts)
    pd.DataFrame([summary]).to_csv(tables / f"{label}_summary.csv", **q3.CSV)
    print(summary)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--benefit-text", action="store_true", help="Score matches to benefit_text on the development pool")
    p.add_argument("--reviewed-precision", type=Path, help="Compare a cost-only swap guard using the 10 reviewed pairs")
    a = p.parse_args()
    run(a.data_root, a.out, benefit_text=a.benefit_text or a.reviewed_precision is not None,
        reviewed_precision=a.reviewed_precision)
