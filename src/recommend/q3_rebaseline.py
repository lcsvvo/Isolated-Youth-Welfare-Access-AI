"""Rerun Q3 keyword diagnostics with the existing human intent cohorts."""

import argparse
import json
from pathlib import Path

import pandas as pd

from src.recommend import q3_eval as ev, q3_run as run, q3_search as search, q3_tie_audit as ties

OUT = run.ROOT / "outputs/05_q3_rebaseline"
INTENT = run.OUT / "review/q3_intent_review_final.csv"
LEGACY = run.OUT / "tables"
READ = dict(encoding="utf-8-sig", keep_default_na=False)


def cohorts(pairs, raw_intent):
    intent = ev.load_human_intent(raw_intent)
    if len(pairs) != 30 or set(intent.pair_id) != set(pairs.pair_id):
        raise ValueError("Q3: 30 input pairs and their human intent reviews are required")
    keys = ["pair_id", "daily_query", "admin_query"]
    if not pairs[keys].sort_values("pair_id").reset_index(drop=True).equals(
            intent[keys].sort_values("pair_id").reset_index(drop=True)):
        raise ValueError("Q3: the human intent review refers to different query text")
    same = intent.intent_equivalence.eq("동일 의도")
    primary = set(intent.loc[same & intent.comparison_class.eq("직접 비교"), "pair_id"])
    secondary = set(intent.loc[same, "pair_id"])
    if len(primary) != 24 or len(secondary) != 27:
        raise ValueError("Q3: expected 24 direct same-intent and 27 same-intent pairs")
    return {"primary_24": primary, "secondary_27": secondary, "legacy_all_30": set(pairs.pair_id)}


def summarize(stats, overlaps, tie, groups):
    hit_keys = ["cond_id", "query_type"]
    tie = tie.merge(stats[["query_id", "cond_id", "pair_id", "query_type"]],
                    on=["query_id", "cond_id"], validate="one_to_one")
    output = []
    for cohort, ids in groups.items():
        s, ov, t = (d[d.pair_id.isin(ids)] for d in (stats, overlaps, tie))
        output.append(s.groupby(hit_keys, as_index=False).agg(
            n_queries=("query_id", "count"), mean_hits=("n_hits", "mean"), median_hits=("n_hits", "median"),
            min_hits=("n_hits", "min"), max_hits=("n_hits", "max"), zero_queries=("zero_result", "sum"),
            score_tie_queries=("score_tie_crosses_cutoff", "sum"),
            id_decides_queries=("servid_order_decides", "sum")).assign(cohort=cohort))
        intersections = ov.assign(zero_common=ov.n_common.eq(0),
                                  jaccard=pd.to_numeric(ov.jaccard, errors="coerce"))
        overlap = intersections.groupby("cond_id", as_index=False).agg(
            n_pairs=("pair_id", "count"), mean_common=("n_common", "mean"), median_common=("n_common", "median"),
            zero_common_pairs=("zero_common", "sum"), mean_jaccard=("jaccard", "mean")).assign(cohort=cohort)
        t = t.groupby(hit_keys, as_index=False).agg(tie_changed_queries=("membership_changed", "sum"),
                                                   mean_replaced=("n_replaced", "mean"),
                                                   max_replaced=("n_replaced", "max")).assign(cohort=cohort)
        output[-1] = output[-1].merge(t, on=["cohort", *hit_keys], validate="one_to_one"), overlap
    return tuple(pd.concat(x, ignore_index=True) for x in zip(*output))


def compare(before, after, keys):
    joined = before.merge(after, on=keys, suffixes=("_legacy", "_current"), validate="one_to_one")
    if len(joined) != len(before) or len(joined) != len(after):
        raise ValueError("Q3: legacy and current conditions or cohorts differ")
    for name in before.columns.difference(keys):
        joined[f"{name}_delta"] = joined[f"{name}_current"] - joined[f"{name}_legacy"]
    return joined


def main(master_path=run.MASTER, out=OUT):
    pairs = pd.read_csv(run.QUERIES, **READ, dtype=str)
    intent = pd.read_csv(INTENT, **READ, dtype=str)
    groups = cohorts(pairs, intent)
    master = search.load_master(master_path)
    queries = run.load_queries()
    conditions = search.PRIMARY_CONDITIONS + search.SENSITIVITY_CONDITIONS
    _, stats, rankings = run.run_search(search.KeywordIndex(master), queries, conditions)
    overlap, tie = run.pair_overlap(queries, rankings, conditions), ties.audit(rankings)
    filenames = ["q3_query_stats.csv", "q3_pair_overlap.csv", "q3_tie_sensitivity.csv"]
    previous = [pd.read_csv(LEGACY / name, **READ) for name in filenames]
    for now, old, key in zip((stats, overlap, tie), previous,
                             (["query_id", "cond_id"], ["pair_id", "cond_id"], ["query_id", "cond_id"])):
        if (old.duplicated(key).any() or now.duplicated(key).any()
                or set(map(tuple, now[key].values)) != set(map(tuple, old[key].values))):
            raise ValueError("Q3: legacy row keys differ from current diagnostics")
    hit, pairs_summary = summarize(stats, overlap, tie, groups)
    old_hit, old_pairs = summarize(*previous, groups)
    tables = out / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    for name, frame in {
        "cohort_query_comparison": compare(old_hit, hit, ["cohort", "cond_id", "query_type"]),
        "cohort_overlap_comparison": compare(old_pairs, pairs_summary, ["cohort", "cond_id"]),
        "query_counts": stats, "pair_overlap": overlap, "tie_sensitivity": tie,
        "intent_cohort": intent[["pair_id", "intent_equivalence", "comparison_class"]].assign(
            primary=intent.pair_id.isin(groups["primary_24"]), secondary=intent.pair_id.isin(groups["secondary_27"])),
    }.items():
        frame.to_csv(tables / f"{name}.csv", **run.CSV)
    meta = {
        "baseline": search.BASELINE_VERSION, "master_rows": len(master), "master_sha256": search.file_sha256(master_path),
        "query_sha256": search.file_sha256(run.QUERIES), "intent_sha256": search.file_sha256(INTENT),
        "legacy_tables_sha256": {n: search.file_sha256(LEGACY / n) for n in filenames},
        "cohorts": {k: sorted(v) for k, v in groups.items()}, "conditions": [c.cond_id for c in conditions],
        "scope": "Development diagnostics only; no human relevance metrics or ranked service IDs exported.",
    }
    (out / "baseline_manifest.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{len(master)} services; {[(key, len(ids)) for key, ids in groups.items()]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master", type=Path, default=run.MASTER)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    main(args.master, args.out)
