"""Audit frozen Q3 rankings and prepare a blinded review of newly pooled Top-20 pairs."""

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml

from ai.eval import independent_retrieval as q3
from rules.filter import EXCLUDE, apply_rules

SEARCH = ("bm25", "dense", "rrf")
DEPTHS = (5, 20, 100)
K = q3.K


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def candidate_set(arms, methods, depth):
    return set().union(*(arms[method][:depth] for method in methods))


def coverage(rankings, grades, *, depths=DEPTHS, include_rescue=False):
    """Micro recall within the supplied judgments; unjudged pairs stay separate."""
    positive = {card: {sid for sid, grade in marks.items() if grade == 2}
                for card, marks in grades.items()}
    all_relevant = sum(map(len, positive.values()))
    rows = []
    specs = [(method, {method: depth}) for method in SEARCH for depth in depths]
    specs += [("union", dict.fromkeys(SEARCH, depth)) for depth in depths]
    specs += [(method, {method: K}) for method in ("rules_q1off", "recommendation")
             if all(method in ranked for ranked in rankings.values())]
    if include_rescue:
        specs.append(("rrf100_plus_bm25_dense20", {"rrf": 100, "bm25": 20, "dense": 20}))
    for name, cutoffs in specs:
        candidates = {card: {sid for arm, depth in cutoffs.items() for sid in r[arm][:depth]}
                      for card, r in rankings.items()}
        hits = [len(ids & positive[card]) for card, ids in candidates.items()]
        total = sum(map(len, candidates.values()))
        judged = sum(len(ids & grades[card].keys()) for card, ids in candidates.items())
        rows.append({
            "candidate_set": name, "depth": max(cutoffs.values()), "cards": len(rankings),
            "candidate_pairs": total, "judged_pairs": judged, "unjudged_pairs": total - judged,
            "known_relevant_pairs": sum(hits), "all_known_relevant_pairs": all_relevant,
            "pooled_recall_micro": sum(hits) / all_relevant if all_relevant else None,
            "cards_with_known_relevant": sum(n > 0 for n in hits),
            "pooled_oracle_p_at_5": sum(min(K, n) for n in hits) / (K * len(rankings)),
        })
    return pd.DataFrame(rows)


def audit(run, subset_path, reviews, config, master, q1_review, out):
    bundle = read_json(run / "restricted/rankings.json")
    subset = read_json(subset_path)
    judgments = q3.load_relevance(reviews)
    scores = q3.score(bundle, judgments, subset=subset)
    _, judged_ids = q3.subset_rankings(bundle, subset)
    rankings = {card: bundle["rankings"][card] for card in subset["cards"]}
    grades = {card: {} for card in rankings}
    for p, v in judgments.loc[judgments.pool_id.isin(judged_ids), ["pool_id", "relevance"]].itertuples(index=False):
        card, sid = bundle["pool_key"][p]
        grades[card][sid] = int(v)

    for name, path in {"config": config, "master": master, "q1_review": q1_review}.items():
        if q3.file_sha256(path) != bundle["inputs_sha256"].get(name):
            raise ValueError(f"{name} differs from the frozen Q3 inputs")
    cfg = yaml.safe_load(config.read_text(encoding="utf-8"))
    services = q3.RecommendationReview(master, q1_path=q1_review).services
    date = q3.parse_reference_date(bundle["as_of"])
    details, cards = [], []
    for card, arms in rankings.items():
        positions = {arm: {sid: i for i, sid in enumerate(ids, 1)} for arm, ids in arms.items()}
        profile = bundle["profile_by_card"][card]
        view = q3.rules_view(profile, cfg, profile.get("access"))
        view["today"] = date
        exclusions = []
        kept = apply_rules(arms["rrf"], services, view, excluded=exclusions)
        kept_ids = {r["servId"]: i for i, r in enumerate(kept, 1)}
        excluded = {r["servId"]: r for r in exclusions}
        matches = [r["servId"] for r in kept[:K]] == arms["recommendation"][:K]
        positive = {sid for sid, v in grades[card].items() if v == 2}
        final = positive & positions["recommendation"].keys()
        missing = positive - final
        top20 = candidate_set(arms, SEARCH, 20)
        cards.append({
            "card_id": card, "known_relevant": len(positive), "final_top5_relevant": len(final),
            "final_p_at_5": len(final) / K, "pooled_oracle_top5_relevant": min(K, len(positive)),
            "known_relevant_outside_rrf100": len(positive - positions["rrf"].keys()),
            "missed_known_relevant": len(missing),
            "missed_from_rrf_top5": len(missing & set(arms["rrf"][:K])),
            "union_top20_unjudged": len(top20 - grades[card].keys()),
            "final_top5_replayed": matches,
        })
        for sid in sorted(missing):
            row = excluded.get(sid) if matches else None
            in_rrf = sid in positions["rrf"]
            status = ("outside_rrf_candidates" if not in_rrf else "replay_mismatch" if not matches
                      else "excluded_by_rules" if row else "below_final_top5")
            penalty = q3._domain_rank(services.loc[sid].to_dict(), view["needs"])[0] if matches and sid in kept_ids else None
            details.append({
                "card_id": card, "servId": sid,
                **{f"{arm}_rank": positions[arm].get(sid) for arm in SEARCH},
                "ranks_after_rules": kept_ids.get(sid) if matches else None,
                "diagnosis": status, "domain_mismatch_penalty": penalty,
                "exclusion_checks": "|".join(n for n, v in row["checks"].items() if v == EXCLUDE) if row else "",
            })

    source = q3.independent.read_csv(run / "review/relevance_review.csv")
    texts = source.drop_duplicates("card_id").set_index("card_id").input_text.to_dict()
    # Remove judged pairs before building the blind form. Reuse the original deterministic pool keys.
    new = {card: {"unjudged": sorted(candidate_set(r, SEARCH, 20) - grades[card].keys())}
           for card, r in rankings.items()}
    _, form = q3.build_pool(new, services, texts, depth=20 * len(SEARCH))
    form = form.assign(**dict.fromkeys(q3.independent.RELEVANCE_COLUMNS[1:], ""))
    owners = scores.drop_duplicates("card_id").set_index("card_id").reviewer
    form["assigned_reviewer"] = form.card_id.map(owners)

    tables = out / "tables"
    tables.mkdir(parents=True, exist_ok=False)
    (out / "review").mkdir()
    coverage(rankings, grades).to_csv(tables / "candidate_coverage.csv", **q3.CSV)
    pd.DataFrame(cards).to_csv(tables / "by_card.csv", **q3.CSV)
    missed = pd.DataFrame(details, columns=[
        "card_id", "servId", "bm25_rank", "dense_rank", "rrf_rank", "ranks_after_rules",
        "diagnosis", "domain_mismatch_penalty", "exclusion_checks",
    ]).convert_dtypes()
    missed.to_csv(tables / "missed_relevant.csv", **q3.CSV)
    form.to_csv(out / "review/relevance_top20_new.csv", **q3.CSV)
    summary = scores.groupby("arm").agg(
        cards=("card_id", "size"), precision_at_5=("precision_at_k", "mean"),
        ndcg_at_5=("ndcg_at_k", "mean"), relevant_in_top5=("n_relevant_in_topk", "sum"),
    )
    print(summary.to_string())
    n_relevant = sum(v == 2 for marks in grades.values() for v in marks.values())
    print(f"Judged: {len(judged_ids)}; known relevant: {n_relevant}; "
          f"new Top-20 review pairs: {len(form)}; final Top-5 replay: {sum(c['final_top5_replayed'] for c in cards)}/{len(cards)}")
    print(missed.groupby(["diagnosis", "exclusion_checks"], dropna=False).size().to_string())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for arg in ("run", "subset", "master", "q1-review", "out"):
        p.add_argument(f"--{arg}", type=Path, required=True)
    p.add_argument("--config", type=Path, default=q3.DEFAULT_CONFIG)
    p.add_argument("--relevance", type=Path, nargs="+", required=True)
    args = p.parse_args()
    audit(args.run, args.subset, args.relevance, args.config, args.master, args.q1_review, args.out)


if __name__ == "__main__":
    main()
