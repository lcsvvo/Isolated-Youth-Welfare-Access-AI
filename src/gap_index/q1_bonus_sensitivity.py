"""Sensitivity of Q1 rank results to the confirmed main-need preference (bonus).

The default (bonus 1) in ai/rules/filter.py is not changed. The bonus is varied only inside a
context manager that patches `rules.filter._domain_rank` and restores it afterwards. The penalty
(5) and the tie-break (confirmed match first) are kept.

Definitions (also written to the metadata json):
- Top-5 composition change (구성 변화): the Top-5 service sets differ.
- Top-5 order change (순서 변화): the Top-5 lists (order included) differ; includes composition changes.
- Top-1 change: the first-ranked service differs.
- Eligible set change: the set of services that pass the eligibility rules differs.
This is a sensitivity analysis; no human relevance judgement is used.
"""

import argparse
import json
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path

import pandas as pd

from src.labeling import q1_human as human
from src.preprocess.keywords import APP_DOMAINS_V3
from src.recommend.q3_search import file_sha256

ROOT = human.ROOT
OUT = ROOT / "outputs/03_q1_recommendation/bonus_sensitivity_20261116"
BONUSES = (0, 0.5, 1, 2)
DEFINITIONS = {
    "top5_set_changed": "Top-5 서비스 집합이 다름",
    "top5_order_changed": "Top-5 목록(순서 포함)이 다름. 구성 변화 포함",
    "top1_changed": "1위 서비스가 다름",
    "eligible_set_changed": "자격 판정을 통과한 서비스 집합이 다름",
}


@contextmanager
def patched_bonus(filter_module, bonus):
    """Temporarily replace the confirmed-match preference; penalty and tie-break stay as is."""
    original = filter_module._domain_rank

    def domain_rank(svc, needs):
        penalty, preference = original(svc, needs)
        return penalty, (bonus if preference else 0)

    filter_module._domain_rank = domain_rank
    try:
        yield
    finally:
        filter_module._domain_rank = original


def case_row(domain, query, profile, results):
    """results: {bonus: ranked servId list (eligible, ordered)}."""
    row = {"domain": domain, "query": query, "profile": profile}
    for b, ranked in results.items():
        row[f"eligible_count_b{b}"] = len(ranked)
        row[f"top5_b{b}"] = "|".join(ranked[:5])
        row[f"eligible_ids_b{b}"] = "|".join(sorted(ranked))
    return row


def summarize(cases, bonuses=BONUSES, bases=(0, 1)):
    """cases: DataFrame of case_row. Pairwise counts of each bonus against each base bonus."""
    out = []
    split = lambda col: cases[col].map(lambda s: s.split("|") if s else [])
    for base in bases:
        a = split(f"top5_b{base}")
        for b in bonuses:
            if b == base:
                continue
            c = split(f"top5_b{b}")
            out.append({
                "base_bonus": base, "bonus": b, "n_cases": len(cases),
                "top5_set_changed": sum(set(x) != set(y) for x, y in zip(a, c)),
                "top5_order_changed": sum(x != y for x, y in zip(a, c)),
                "top1_changed": sum(x[:1] != y[:1] for x, y in zip(a, c)),
                "eligible_set_changed": int((cases[f"eligible_ids_b{base}"] != cases[f"eligible_ids_b{b}"]).sum()),
            })
    return pd.DataFrame(out)


def git_head():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main(out=OUT):
    from src.gap_index import q1_recommendation as q1
    master_hash = file_sha256(q1.MASTER)
    if master_hash != q1.REVIEWED_MASTER:
        raise ValueError("Revalidate Q1 evidence before applying it to a different master snapshot")
    sys.path.insert(0, str(ROOT / "ai"))
    import rules.filter as rules
    from core.data import load_services
    from core.review_data import attach_q1
    from retrieve.bm25 import BM25Index, build_texts

    master = load_services(q1.MASTER)
    overlay, _, _ = q1.build(master, q1.load_labor_future_core(q1.LABOR_FUTURE_REVIEW, master.servId))
    texts = build_texts(master)
    after = attach_q1(master.set_index("servId", drop=False), overlay)
    if texts != build_texts(after):
        raise ValueError("Q1: rank metadata changed the search documents")
    (ROOT / "private").mkdir(exist_ok=True)
    bm25 = BM25Index(master.servId, texts, cache_path=ROOT / "private/q1_recommendation_bm25.pkl")
    rows = []
    for domain in APP_DOMAINS_V3:
        needs = [{"domain": domain}]
        for query in q1.PROBES[domain]:
            ids = bm25.search(query, 100)
            for i, profile in enumerate(q1.RANK_PROFILES, 1):
                results = {}
                for b in BONUSES:
                    with patched_bonus(rules, b):
                        ranked = rules.apply_rules(ids, after, {"needs": needs, **profile})
                    results[b] = [r["servId"] for r in ranked]
                rows.append(case_row(domain, query, i, results))
    cases = pd.DataFrame(rows)
    summary = summarize(cases)
    out.mkdir(parents=True, exist_ok=True)
    tables = out / "tables"
    tables.mkdir(exist_ok=True)
    cases.drop(columns=[c for c in cases if c.startswith("eligible_ids_")]).to_csv(
        tables / "bonus_top5.csv", **human.CSV)
    summary.to_csv(tables / "bonus_summary.csv", **human.CSV)
    (out / "bonus_sensitivity.json").write_text(json.dumps({
        "schema": "q1-bonus-sensitivity-v1", "code_commit": git_head(), "master_sha256": master_hash,
        "bonuses": list(BONUSES), "default_bonus": 1, "penalty": 5, "n_cases": len(cases),
        "candidates": "BM25 top-100 per PROBES query; RANK_PROFILES x2",
        "definitions": DEFINITIONS,
        "note": "Sensitivity analysis only; no human relevance judgement; filter.py defaults unchanged.",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    main(parser.parse_args().out)
