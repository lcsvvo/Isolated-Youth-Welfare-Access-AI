"""Measure the frozen benefit-type reranker against the archived Q3 final recommendation."""

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
import yaml

from ai.eval import independent_retrieval as q3
from ai.eval import q3_top5_audit as audit
from ai.eval import q3_top5_dev as dev
from ai.eval.q3_benefit_experiment import load_supplement
from retrieve.benefit import WINDOW, rank_by_benefit_type
from rules.filter import apply_rules


def run(archive, subset_path, review_files, config, master, q1_review, out, additional_review):
    rank_file = archive / "restricted/rankings.json"
    bundle = audit.read_json(rank_file)
    subset = audit.read_json(subset_path)
    old_review = q3.load_relevance(review_files)
    baseline = q3.score(bundle, old_review, subset=subset)
    _, judged_ids = q3.subset_rankings(bundle, subset)
    grades = {card: {} for card in subset["cards"]}
    for row in old_review.loc[old_review.pool_id.isin(judged_ids)].itertuples():
        card, sid = bundle["pool_key"][row.pool_id]
        grades[card][sid] = int(row.relevance)

    inputs = {name: q3.file_sha256(path)
              for name, path in {"config": config, "master": master, "q1_review": q1_review}.items()}
    if any(hexdigest != bundle["inputs_sha256"].get(name) for name, hexdigest in inputs.items()):
        raise ValueError("Configuration, service master and Q1 labels must match the archived Q3 run")
    cfg = yaml.safe_load(config.read_text(encoding="utf-8"))
    services = q3.RecommendationReview(master, q1_path=q1_review).services
    date = q3.parse_reference_date(bundle["as_of"])
    source = q3.independent.read_csv(archive / "review/relevance_review.csv")
    texts = source.drop_duplicates("card_id").set_index("card_id").input_text.to_dict()

    reviewed_new = load_supplement(additional_review) if additional_review else {}
    rows, queue, frozen = [], {}, {}
    for card, marks in grades.items():
        arms = bundle["rankings"][card]
        before = arms["recommendation"][:audit.K]
        profile = bundle["profile_by_card"][card]
        view = q3.rules_view(profile, cfg, profile.get("access"))
        view["today"] = date
        kept = apply_rules(arms["rrf"], services, view)
        replayed = [r["servId"] for r in kept[:audit.K]] == before
        # The archived baseline remains the comparison on cards whose rule replay differs.
        after = before
        if replayed:
            ordered, _ = rank_by_benefit_type(kept, texts[card], services)
            after = [r["servId"] for r in ordered[:audit.K]]
        frozen[card] = {"baseline": before, "benefit_type": after}

        for sid in set(after) - marks.keys():
            pool_id = hashlib.sha256(f"q3-v1:{card}:{sid}".encode()).hexdigest()[:20]
            if pool_id in reviewed_new:
                reviewed_card, grade = reviewed_new.pop(pool_id)
                if reviewed_card != card:
                    raise ValueError(f"Supplemental review card does not match {pool_id}")
                marks[sid] = grade
        queue[card] = set(after) - marks.keys()
        rows.append(dict(
            card_id=card, baseline_replayed=replayed, top5_changed=set(before) != set(after),
            baseline_hits=sum(marks.get(sid) == 2 for sid in before),
            new_known_hits=sum(marks.get(sid) == 2 for sid in after),
            new_unjudged=len(queue[card]),
        ))
    if reviewed_new:
        raise ValueError("Supplemental file contains judgments outside the frozen trial's new recommendations")

    by_card = pd.DataFrame(rows)
    n = len(by_card)
    slots = n * audit.K
    totals = by_card.sum(numeric_only=True)
    old_p5 = totals.baseline_hits / slots
    archived_hits = baseline.loc[baseline.arm.eq("recommendation"), "n_relevant_in_topk"].sum()
    if totals.baseline_hits != archived_hits:
        raise ValueError("Archived baseline differs from the Q3 evaluator")
    low, high = totals.new_known_hits, totals.new_known_hits + totals.new_unjudged
    summary = {
        "cards": n, "baseline_replayed": int(totals.baseline_replayed), "changed_top5": int(totals.top5_changed),
        "baseline_hits": int(totals.baseline_hits), "new_known_hits": int(low),
        "new_unjudged": int(totals.new_unjudged), "baseline_p_at_5": old_p5,
        "new_p_at_5_lower": low / slots, "new_p_at_5_upper": high / slots,
        "delta_p_at_5_lower": (low - totals.baseline_hits) / slots,
        "delta_p_at_5_upper": (high - totals.baseline_hits) / slots,
    }

    out.mkdir(parents=True, exist_ok=False)
    (out / "tables").mkdir()
    (out / "restricted").mkdir()
    by_card.to_csv(out / "tables/by_card.csv", **q3.CSV)
    pd.DataFrame([summary]).to_csv(out / "tables/summary.csv", **q3.CSV)
    (out / "restricted/rankings.json").write_text(json.dumps({
        "source_sha256": q3.file_sha256(rank_file), "window": WINDOW, "rankings": frozen,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if any(queue.values()):
        (out / "review").mkdir()
        dev.write_review_form(out / "review/independent_q3_new_top5.csv", queue, services, texts)
    print(summary)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for option in ("run", "subset", "master", "q1-review", "out"):
        cli.add_argument(f"--{option}", type=Path, required=True)
    cli.add_argument("--relevance", type=Path, nargs="+", required=True)
    cli.add_argument("--config", type=Path, default=q3.DEFAULT_CONFIG)
    cli.add_argument("--additional-review", type=Path)
    a = cli.parse_args()
    run(a.run, a.subset, a.relevance, a.config, a.master, a.q1_review, a.out, a.additional_review)
