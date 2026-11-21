"""Compare BM25, Dense, RRF and recommendation ranks after independent human review."""

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
AI = ROOT / "ai"
for folder in (ROOT, AI):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

from confirm.session import rules_view
from core.models import Embedder
from core.review_data import RecommendationReview, configured_addendum_path
from retrieve import bm25, dense, dense_texts
from retrieve.fusion import check_params, search_and_fuse
from retrieve.rrf import DEFAULT_TOP_N
from rules.filter import _domain_rank, apply_rules
from rules.period import parse_reference_date
from src.recommend import q3_eval
from src.recommend import q3_independent as independent
from src.recommend.q3_run import CSV
from src.recommend.q3_search import file_sha256

# rules_q1off: same rules as recommendation but the Q1 main-need preference is 0 (recommendation uses +1).
ARMS = ("bm25", "dense", "rrf", "rules_q1off", "recommendation")
K = 5
POOL_DEPTH = 5  # judged depth; Precision@5·nDCG@5 need only each arm's Top-5
SCHEMAS = ("q3-independent-ranks-v1", "q3-independent-ranks-v2")
DEFAULT_CONFIG = AI / "configs/config.yaml"
# Pre-registered paired comparisons (a, b): d = b - a. Effects are read in this order.
COMPARISONS = (
    ("bm25", "dense", "검색 모델: Dense - BM25"),
    ("bm25", "rrf", "검색 결합: RRF - BM25"),
    ("dense", "rrf", "검색 결합: RRF - Dense"),
    ("rrf", "rules_q1off", "규칙 효과: 규칙(Q1 0점) - RRF"),
    ("rules_q1off", "recommendation", "Q1 가점 효과: +1점 - 0점"),
    ("rrf", "recommendation", "최종 추천 - RRF"),
    ("bm25", "recommendation", "최종 추천 - BM25"),
)


def approved_cards(card_path, review_path, domains):
    cards, review = independent.load_assignment(card_path, review_path)
    frame = cards.merge(review, on="card_id", validate="one_to_one")
    checks = {
        "human authorship": frame.authorship.eq("human") & frame.author.str.strip().ne(""),
        "independent review": (frame.reviewer.str.strip().ne("") & frame.reviewer.ne(frame.author)
                               & frame.judge_type.eq("human") & frame.review_evidence.str.strip().ne("")
                               & frame.reviewed_at.str.strip().ne("")),
        "approval": frame.decision.eq("approved") & frame.semantic_overlap.eq("no"),
        "expected action": frame.expected_action.isin({"search", "clarify", "stop"}),
    }
    for reason, allowed in checks.items():
        if not allowed.all():
            raise ValueError(f"Q3: {reason} incomplete for {int((~allowed).sum())} cards")
    audit = independent.audit_text(cards, independent.development_bank())
    if not audit.status.eq("human_semantic_review").all():
        raise ValueError("Q3: unresolved empty or near-duplicate card text")
    for r in frame[frame.expected_action.eq("search")].itertuples():
        needs = r.needs.split("|")
        if len(set(needs)) != len(needs) or not set(needs) <= set(domains):
            raise ValueError(f"Q3: invalid confirmed need for {r.card_id}")
    return frame


def design_cards(domains):
    """Use the persona design labels (fixed before writing) without a separate card review.

    Text checks still apply: human authorship, assigned reviewer differs from the writer, and the overlap audit.
    """
    cards, review = independent.persona_forms(domains)
    audit = independent.audit_text(cards, independent.development_bank())
    if not audit.status.eq("human_semantic_review").all():
        raise ValueError("Q3: unresolved empty or near-duplicate card text")
    frame = cards.merge(review, on="card_id", validate="one_to_one")
    for r in frame[frame.expected_action.eq("search")].itertuples():
        needs = r.needs.split("|")
        if not r.needs or len(set(needs)) != len(needs) or not set(needs) <= set(domains):
            raise ValueError(f"Q3: invalid design need for {r.card_id}")
    return frame


def _configured_as_of(cfg):
    return cfg.get("evaluation", {}).get("q3", {}).get("as_of")


def rank_all(query, profile, bm, de, services, cfg, *, as_of=None):
    if not query.strip() or not profile["needs"]:
        raise ValueError("Q3: ranked arms need a query and confirmed needs")
    if as_of is None:
        # Direct callers with a retrieval-only config use the bundled Q3 date.
        date_cfg = cfg if "evaluation" in cfg else yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8"))
        as_of = _configured_as_of(date_cfg)
    reference_date = parse_reference_date(as_of)
    retrieval = cfg["retrieval"]
    params = check_params(retrieval["top_k_each"], retrieval["rrf_k"], retrieval.get("rrf_top_n", DEFAULT_TOP_N))
    retrievers = {"bm25": bm, "dense": de}
    found = search_and_fuse(retrievers, dict.fromkeys(retrievers, query), params, known_ids=services.index)
    if found["errors"]:
        failures = ", ".join(f"{e['module']}: {e['type']}" for e in found["errors"])
        raise ValueError(f"Q3: retrieval comparison interrupted ({failures})")
    return {"bm25": list(dict.fromkeys(found["bm25"])), "dense": list(dict.fromkeys(found["dense"])),
            "rrf": found["fused"], **rule_arms(found["fused"], profile, services, cfg, reference_date)}


def rule_arms(fused, profile, services, cfg, reference_date):
    """Rule-based arms from a fused RRF list. With no needs the pipeline asks instead of recommending."""
    top_n = cfg["retrieval"]["final_top_n"]
    if not profile["needs"]:
        return {"rules_q1off": [], "recommendation": []}
    rules = rules_view(profile, cfg, profile.get("access"))
    rules["today"] = reference_date
    judged = apply_rules(fused, services, rules)
    # Remove only the Q1 main-need preference; ties then follow the RRF rank instead of the preference.
    preference = {row["servId"]: _domain_rank(services.loc[row["servId"]].to_dict(), rules.get("needs", []))[1]
                  for row in judged}
    q1off = sorted(judged, key=lambda row: (row["score"] + preference[row["servId"]], row["rank"]))
    return {"rules_q1off": [row["servId"] for row in q1off[:top_n]],
            "recommendation": [row["servId"] for row in judged[:top_n]]}


def card_profile(needs, age, region_sido, activity_range, access_json, card_id):
    """Profile from reviewed or extracted labels. The first need is primary; later needs are supplementary."""
    access = json.loads(access_json or "[]")
    if not isinstance(access, list) or not all(isinstance(a, dict) for a in access):
        raise ValueError(f"Q3: access must be a list of dictionaries ({card_id})")
    domains = [d for d in (needs or "").split("|") if d]
    return {"needs": [{"domain": d, "level": "main" if i == 0 else "sub"} for i, d in enumerate(domains)],
            "age": age or None, "region_sido": region_sido or None,
            "activity_range": activity_range or None, "access": access}


RULE_CODE = ("ai/rules", "ai/confirm", "ai/core/review_data.py", "ai/configs")


def code_commit():
    import subprocess
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def rule_code_changed(since):
    """True/False when git can tell whether rule code changed after `since`; None when unknown."""
    import subprocess
    if not since:
        return None
    done = subprocess.run(["git", "diff", "--quiet", since, "--", *RULE_CODE], cwd=ROOT, capture_output=True)
    return {0: False, 1: True}.get(done.returncode)


PROFILE_COLUMNS = ("card_id", "needs", "age", "region_sido", "activity_range", "access_json")
RELEVANCE_KEEP = ("pool_id", "relevance", "evidence", "reviewer", "reviewed_at", "judge_type")


def rerank(args):
    """Recompute the rule arms of a frozen run with other profiles (e.g. model-extracted needs).

    BM25, Dense and RRF stay as frozen, so no GPU is needed. Pool ids use the same hash, so earlier
    judgments carry over and only services that newly enter a Top-5 need review.
    """
    bundle = json.loads((args.run / "restricted/rankings.json").read_text(encoding="utf-8"))
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    master = args.master or AI / cfg["paths"]["services"]
    expected = bundle["inputs_sha256"]
    for name, path in {"master": master, "config": args.config, "q1_review": args.q1_review}.items():
        if expected.get(name) != file_sha256(path):
            raise ValueError(f"Q3: {name} differs from the frozen run; rerank needs the same inputs")
    domains = yaml.safe_load((AI / cfg["paths"]["domains_file"]).read_text(encoding="utf-8"))["domains"]
    table = independent.read_csv(args.profiles)
    if not set(PROFILE_COLUMNS) <= set(table) or not table.card_id.is_unique:
        raise ValueError(f"Q3: profiles need unique card_id and columns {PROFILE_COLUMNS}")
    missing = set(bundle["rankings"]) - set(table.card_id)
    if missing:
        raise ValueError(f"Q3: profiles missing for {len(missing)} ranked cards")
    profiles = {r.card_id: card_profile(r.needs, r.age, r.region_sido, r.activity_range, r.access_json, r.card_id)
                for r in table.itertuples() if r.card_id in bundle["rankings"]}
    for card, profile in profiles.items():
        bad = {n["domain"] for n in profile["needs"]} - set(domains)
        if bad:
            raise ValueError(f"Q3: unknown need {sorted(bad)} for {card}")
    if args.out.exists():
        raise ValueError("Q3: use a new run directory to protect prior ranks and reviews")
    services = RecommendationReview(master, q1_path=args.q1_review).services
    reference_date = parse_reference_date(bundle["as_of"])
    rankings = {card: {**{a: arms[a] for a in ("bm25", "dense", "rrf")},
                       **rule_arms(arms["rrf"], profiles[card], services, cfg, reference_date)}
                for card, arms in bundle["rankings"].items()}
    texts = dict(zip(*[independent.read_csv(args.run / "review/relevance_review.csv")[c] for c in ("card_id", "input_text")]))
    key, sheet = build_pool(rankings, services, texts, depth=bundle["pool_depth"])
    new_rows = sheet[~sheet.pool_id.isin(bundle["pool_key"])]
    args.out.mkdir(parents=True)
    (args.out / "review").mkdir()
    (args.out / "restricted").mkdir()
    new_rows.assign(**dict.fromkeys(independent.RELEVANCE_COLUMNS[1:], "")).to_csv(
        args.out / "review/relevance_review_new.csv", **CSV)
    (args.out / "restricted/rankings.json").write_text(json.dumps({
        **bundle, "rankings": rankings, "pool_key": key, "profile_by_card": profiles,
        "profile_condition": f"profiles from {args.profiles.name} (rule arms recomputed; retrieval frozen)",
        "derived_from": {"run": str(args.run), "rankings_sha256": file_sha256(args.run / "restricted/rankings.json")},
        "inputs_sha256": {**expected, "profiles": file_sha256(args.profiles)}, "code_commit": code_commit(),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    changed = rule_code_changed(bundle.get("code_commit"))
    if changed is not False:
        print("Q3 warning: rule code may differ from the source run. Compare extracted profiles only with a "
              "rerank of the design labels made with this same code, not with the source run.")
    no_needs = sum(not p["needs"] for p in profiles.values())
    print(f"{len(rankings)} cards reranked; {no_needs} without needs (no recommendation); "
          f"{len(new_rows)} new pool rows to judge")


def build_pool(rankings, services, card_text, depth=POOL_DEPTH):
    """Return reviewer rows in stable shuffled order; retain the key separately."""
    key, blinded = {}, []
    source_fields = ("servNm", "summary_text", "target_text", "criteria_text", "benefit_text", "apply_text", "detail_url")
    for card, arms in sorted(rankings.items()):
        ids = set().union(*(set(r[:depth]) for r in arms.values()))
        pairs = sorted((hashlib.sha256(f"q3-v1:{card}:{sid}".encode()).hexdigest()[:20], sid) for sid in ids)
        for blind_id, sid in pairs:
            if blind_id in key:
                raise ValueError("Q3: duplicate pool key")
            key[blind_id] = [card, sid]
            service = services.loc[sid].fillna("")
            blinded.append({"pool_id": blind_id, "card_id": card, "input_text": card_text[card],
                            **{f: service.get(f, "") for f in source_fields}})
    return key, pd.DataFrame(blinded, columns=["pool_id", "card_id", "input_text", *source_fields])


def prepare_pool(args):
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    as_of = parse_reference_date(args.as_of if args.as_of is not None else _configured_as_of(cfg))
    paths = cfg["paths"]
    domains = yaml.safe_load((AI / paths["domains_file"]).read_text(encoding="utf-8"))["domains"]
    cards = design_cards(domains) if args.design_labels else approved_cards(args.cards, args.reviews, domains)
    master = args.master or AI / paths["services"]
    configured_q1 = AI / paths["q1_review"] if paths.get("q1_review") else None
    q1_path = configured_addendum_path(configured_q1, args.q1_review, custom_master=args.master is not None,
                                      optional=args.config.resolve() == DEFAULT_CONFIG)
    if q1_path is None or not Path(q1_path).exists():
        # Without the Q1 review every preference is 0, so rules_q1off and recommendation would be identical.
        raise ValueError("Q3: Q1 review JSON is required for the Q1 0 vs +1 comparison (use --q1-review)")
    data = RecommendationReview(master, q1_path=q1_path)
    services = data.services
    query_cards = cards[cards.expected_action.eq("search")]
    if query_cards.empty:
        raise ValueError("Q3: independent set has no approved search cards")
    profiles = {r.card_id: card_profile(r.needs, r.age, r.region_sido, r.activity_range, r.access_json, r.card_id)
                for r in query_cards.itertuples()}
    if args.out.exists():
        raise ValueError("Q3: use a new run directory to protect prior ranks and reviews")
    ids = list(services.index)
    cfg_retrieval = cfg["retrieval"]
    args.cache.mkdir(parents=True, exist_ok=True)  # private/ is not in git (fresh clones such as Colab)
    bm = bm25.BM25Index(ids, bm25.build_texts(services), cache_path=args.cache / "bm25_tokens.pkl")
    model = Embedder(cfg["models"]["embedder"], local_files_only=True)
    dprofile = cfg_retrieval.get("dense_profile", "baseline")
    de = dense.DenseIndex(model, ids, dense_texts.build_texts(services, dprofile),
                          cache_path=args.cache / "dense_index.npz")
    rankings = {r.card_id: rank_all(r.input_text, profiles[r.card_id], bm, de, services, cfg, as_of=as_of)
                for r in query_cards.itertuples()}
    key, sheet = build_pool(rankings, services, query_cards.set_index("card_id").input_text.to_dict(),
                            depth=args.pool_depth)
    args.out.mkdir(parents=True)
    blind, private = args.out / "review", args.out / "restricted"
    blind.mkdir()
    private.mkdir()
    for name, cols in (("relevance", independent.RELEVANCE_COLUMNS),
                       ("eligibility", independent.ELIGIBILITY_COLUMNS)):
        form = sheet.assign(**dict.fromkeys(cols[1:], ""))
        if name == "eligibility":
            for field in ("age", "region_sido", "activity_range", "access"):
                form[f"confirmed_{field}"] = form.card_id.map(
                    lambda card_id: json.dumps(profiles[card_id][field], ensure_ascii=False)
                    if field == "access" else profiles[card_id][field])
        form.to_csv(blind / f"{name}_review.csv", **CSV)
    (private / "rankings.json").write_text(json.dumps({
        "schema": SCHEMAS[-1], "eval_k": K, "pool_depth": args.pool_depth, "as_of": as_of.isoformat(),
        "arms": list(ARMS), "code_commit": code_commit(),
        "query_condition": "same approved human card text, no extraction/expansion",
        "profile_condition": "reviewed card_review.csv labels (oracle needs, not model extraction)",
        "q1_preference": {"rules_q1off": 0, "recommendation": 1},
        "card_labels": ("persona design labels (정답_* columns), no separate card review" if args.design_labels
                        else "approved card_review.csv"),
        "inputs_sha256": {name: file_sha256(path) for name, path in {
            "master": master, "config": args.config, "cards": args.cards, "reviews": args.reviews, "q1_review": q1_path,
            **({f"persona_{p.stem}": p for p in sorted(independent.PERSONAS.glob("persona_cards_*.xlsx"))}
               if args.design_labels else {}),
        }.items() if path is not None},
        "model": cfg["models"]["embedder"], "model_cache_key": model.cache_key, "dense_profile": dprofile,
        "retrieval_config": cfg_retrieval, "action_counts": cards.expected_action.value_counts().to_dict(),
        "profile_by_card": profiles, "rankings": rankings, "pool_key": key,
    }, ensure_ascii=False, indent=2), encoding="utf-8")


def reviewed_values(frame, keys, fields, *, all_keys, evidence_optional=None):
    """evidence_optional: boolean mask of rows that may leave evidence blank (e.g. relevance 0)."""
    need = {"pool_id", "reviewer", "reviewed_at", "judge_type", "evidence", *fields}
    if not need <= set(frame) or not frame.pool_id.is_unique:
        raise ValueError("Q3: judgment keys or fields missing/duplicated")
    ids = set(frame.pool_id)
    if not ids <= set(keys) or not set(all_keys) <= ids:
        raise ValueError("Q3: incomplete or foreign judgments")
    filled = {col: frame[col].str.strip().ne("") for col in need}
    if evidence_optional is not None:
        filled["evidence"] = filled["evidence"] | evidence_optional
    if not frame.judge_type.eq("human").all() or any(not ok.all() for ok in filled.values()):
        raise ValueError("Q3: final human judgment, evidence and reviewer required")
    return frame.set_index("pool_id")


def dcg(gains):
    """Graded DCG with gain 2^rel - 1 (same as the RRF weight review)."""
    return sum((2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(gains))


def rank_overlap(bundle, k=K):
    """Label-free Top-k differences between arms. This is not a quality measure."""
    rows = []
    for card, arms in sorted(bundle["rankings"].items()):
        present = [a for a in ARMS if a in arms]
        for i, a in enumerate(present):
            for b in present[i + 1:]:
                top_a, top_b = arms[a][:k], arms[b][:k]
                shared = set(top_a) & set(top_b)
                union = set(top_a) | set(top_b)
                rows.append({"card_id": card, "arm_a": a, "arm_b": b, "k": k, "n_shared": len(shared),
                             "jaccard": len(shared) / len(union) if union else None,
                             "same_top1": int(bool(top_a and top_b and top_a[0] == top_b[0])),
                             "same_order": int(top_a == top_b)})
    detail = pd.DataFrame(rows)
    summary = detail.groupby(["arm_a", "arm_b"], sort=False)[["n_shared", "jaccard", "same_top1", "same_order"]].mean()
    return detail, summary.assign(n_cards=detail.groupby(["arm_a", "arm_b"], sort=False).size())


def paired_arms(per_card, measures=("precision_at_k", "ndcg_at_k", "lenient_precision_at_k")):
    """Card-level paired differences for the pre-registered comparisons (bootstrap CI, sign test)."""
    wide = {m: per_card.pivot(index="card_id", columns="arm", values=m).astype(float) for m in measures}
    rows = []
    for a, b, label in COMPARISONS:
        for m in measures:
            if {a, b} <= set(wide[m]):
                rows.append({"measure": m, **q3_eval.paired_summary(wide[m][a].values, wide[m][b].values, label),
                             "arm_a": a, "arm_b": b})
    return pd.DataFrame(rows)


def reviewer_robustness(per_card, measures=("precision_at_k", "ndcg_at_k")):
    """Rater sensitivity: paired differences within each reviewer, and leaving one reviewer out."""
    if "reviewer" not in per_card or per_card.reviewer.nunique() < 2:
        return pd.DataFrame()
    rows = []
    for reviewer, part in per_card.groupby("reviewer"):
        rows.append(paired_arms(part, measures).assign(scope="only", reviewer=reviewer))
        rows.append(paired_arms(per_card[per_card.reviewer.ne(reviewer)], measures).assign(scope="without",
                                                                                            reviewer=reviewer))
    cols = ["scope", "reviewer", "measure", "comparison", "n_pairs", "mean_diff", "ci95_lo", "ci95_hi",
            "n_pos", "n_neg", "n_zero"]
    return pd.concat(rows, ignore_index=True)[cols]


def subset_rankings(bundle, subset=None):
    """Rankings and pool ids to judge. subset = {"cards": [...], "arms": [...]} limits both."""
    keys = bundle["pool_key"]
    if subset is None:  # full evaluation: every pooled row must be judged
        return bundle.get("rankings", {}), set(keys)
    lookup = {tuple(pair): blind_id for blind_id, pair in keys.items()}
    rankings = bundle["rankings"]
    unknown = set(subset["cards"]) - set(rankings) or set(subset["arms"]) - set(ARMS)
    if unknown:
        raise ValueError(f"Q3: unknown subset entries {sorted(unknown)}")
    rankings = {c: {a: rankings[c][a] for a in subset["arms"]} for c in subset["cards"]}
    needed = {lookup[(c, sid)] for c, arms in rankings.items() for r in arms.values() for sid in r[:K]}
    return rankings, needed


def select_subset(owners, cards, per_reviewer, arms, seed="q3-subset-v1"):
    """Fixed random choice of per_reviewer cards for each assigned reviewer, made before any judgment."""
    chosen = []
    for reviewer in sorted({r for _, r in owners.values()}):
        mine = sorted(c for c in cards if owners[c][1] == reviewer)
        chosen += sorted(mine, key=lambda c: hashlib.sha256(f"{seed}:{c}".encode()).hexdigest())[:per_reviewer]
    return {"cards": sorted(chosen), "arms": list(arms), "seed": seed, "per_reviewer": per_reviewer}


def score(bundle, relevance, eligibility=None, subset=None):
    keys = bundle["pool_key"]
    rankings, needed = subset_rankings(bundle, subset)
    relevance = relevance[relevance.pool_id.isin(needed)]  # rows outside the judged subset are ignored
    # Evidence is required for relevance 1 and 2; an irrelevant (0) row may leave it blank.
    rel = reviewed_values(relevance, keys, ["relevance"], all_keys=needed,
                          evidence_optional=relevance["relevance"].str.strip().eq("0"))
    if not rel.relevance.isin({"0", "1", "2"}).all():
        raise ValueError("Q3: relevance must be 0, 1 or 2")
    relevant = set(rel.index[rel.relevance.eq("2")])
    checks = ["age_status", "region_status", "access_status"]
    elig = None
    if eligibility is not None:
        if ("pool_id" not in eligibility or not eligibility.pool_id.is_unique
                or not set(eligibility.pool_id) <= set(keys)):
            raise ValueError("Q3: foreign or duplicate eligibility keys")
        # Eligibility is judged only for direct matches. The template also contains other pool rows.
        direct_rows = eligibility[eligibility.pool_id.isin(relevant)]
        elig = reviewed_values(direct_rows, keys, checks, all_keys=relevant)
        if not elig[checks].isin({"compatible", "incompatible", "uncertain"}).all().all():
            raise ValueError("Q3: invalid eligibility judgment")
    lookup = {tuple(pair): blind_id for blind_id, pair in keys.items()}
    relevant_counts = pd.Series([keys[x][0] for x in relevant], dtype=str).value_counts().to_dict()
    grade = rel.relevance.astype(int).to_dict()
    # One reviewer judges every arm of a card, so arm differences are compared within a reviewer.
    card_reviewer = (rel.assign(card=[keys[x][0] for x in rel.index]).groupby("card").reviewer
                     .agg(lambda s: "|".join(sorted(set(s.str.strip())))).to_dict())
    grades_by_card = {}
    for blind_id in needed:
        grades_by_card.setdefault(keys[blind_id][0], []).append(grade[blind_id])
    rows = []
    for card, arms in rankings.items():
        ideal = dcg(sorted(grades_by_card.get(card, []), reverse=True)[:K])
        for arm in ARMS:
            if arm not in arms:  # v1 bundles have no rules_q1off arm
                continue
            ids = [lookup[(card, sid)] for sid in arms[arm][:K]]
            found = set(ids) & relevant
            total = relevant_counts.get(card, 0)
            gains = [grade[x] for x in ids]
            row = {"card_id": card, "arm": arm, "reviewer": card_reviewer.get(card, ""), "k": K, "n_returned": len(ids),
                   "n_relevant_in_pool": total, "n_relevant_in_topk": len(found),
                   "precision_at_k": len(found) / K, "success_at_k": int(bool(found)),
                   "pooled_recall_at_k": len(found) / total if total else None,
                   "lenient_precision_at_k": sum(g >= 1 for g in gains) / K,
                   "ndcg_at_k": dcg(gains) / ideal if ideal else None}
            if elig is not None:
                flags = elig.loc[sorted(found), checks]
                incompatible = flags.eq("incompatible").any(axis=1)
                uncertain = flags.eq("uncertain").any(axis=1) & ~incompatible
                row.update(n_relevant_incompatible=int(incompatible.sum()),
                           n_relevant_uncertain_only=int(uncertain.sum()))
            rows.append(row)
    return pd.DataFrame(rows)


def split_review(sheet, owners, n_shared=2, seed="q3-split-v1"):
    """Split the blind sheet by card. owners: card_id -> (writer, assigned reviewer).

    Each reviewer judges their assigned cards. n_shared cards per reviewer are also judged by a
    second person who is neither the writer nor the first reviewer, for agreement only.
    """
    missing = set(sheet.card_id) - set(owners)
    if missing:
        raise ValueError(f"Q3: no reviewer for cards {sorted(missing)}")
    people = sorted({p for pair in owners.values() for p in pair})
    out = {p: [] for p in people}
    shared = []
    for reviewer in people:
        cards = sorted(c for c in set(sheet.card_id) if owners[c][1] == reviewer)
        for card in cards:
            out[reviewer].append(sheet[sheet.card_id.eq(card)].assign(assignment="primary"))
        ranked = sorted(cards, key=lambda c: hashlib.sha256(f"{seed}:{c}".encode()).hexdigest())
        shared += ranked[:n_shared]
    for card, person in assign_second_reviewers(shared, owners, people, seed).items():
        out[person].append(sheet[sheet.card_id.eq(card)].assign(assignment="agreement"))
    return {p: pd.concat(frames, ignore_index=True) for p, frames in out.items() if frames}


def assign_second_reviewers(cards, owners, people, seed):
    """Give each shared card to someone who is neither its writer nor its first reviewer, as evenly as possible.

    Exhaustive for small sets (the usual case); greedy beyond that.
    """
    import itertools
    options = [[p for p in people if p not in owners[c]] for c in cards]
    if any(not o for o in options):
        raise ValueError("Q3: no eligible second reviewer for a shared card")
    tiebreak = {(c, p): hashlib.sha256(f"{seed}:{c}:{p}".encode()).hexdigest() for c in cards for p in people}

    def cost(choice):
        load = pd.Series(choice).value_counts().reindex(people, fill_value=0)
        return (load.max() - load.min(), load.max(), [tiebreak[(c, p)] for c, p in zip(cards, choice)])

    if len(cards) <= 12:
        best = min(itertools.product(*options), key=cost)
    else:
        load, best = dict.fromkeys(people, 0), []
        for c, o in zip(cards, options):
            pick = min(o, key=lambda p: (load[p], tiebreak[(c, p)]))
            load[pick] += 1
            best.append(pick)
    return dict(zip(cards, best))


def merge_review(frames):
    """Combine split sheets: final rows feed scoring; agreement rows give Cohen's kappa.

    A final row is "primary" or "adjudicated" (a primary row later corrected; the sheet keeps one row per pool id).
    Any other assignment value stops the merge so no judgment is dropped silently.
    """
    allrows = pd.concat(frames, ignore_index=True)
    unknown = set(allrows.assignment) - {"primary", "adjudicated", "agreement"}
    if unknown:
        raise ValueError(f"Q3: unknown assignment values {sorted(unknown)}")
    primary = allrows[allrows.assignment.isin({"primary", "adjudicated"})].drop(columns="assignment")
    if not primary.pool_id.is_unique:
        raise ValueError("Q3: a pool row has more than one primary judgment")
    second = allrows[allrows.assignment.eq("agreement")].set_index("pool_id")
    second = second[["relevance", "reviewer"]].rename(columns={"relevance": "relevance_2", "reviewer": "reviewer_2"})
    pairs = primary.set_index("pool_id").join(second, how="inner")
    pairs = pairs[pairs.relevance.isin({"0", "1", "2"}) & pairs.relevance_2.isin({"0", "1", "2"})]
    agreement = {"n_rows": len(pairs), "n_cards": pairs.card_id.nunique()}
    if len(pairs):
        a, b = pairs.relevance.tolist(), pairs.relevance_2.tolist()
        agreement.update(exact=float((pairs.relevance == pairs.relevance_2).mean()),
                         kappa_0_1_2=q3_eval.cohen_kappa(a, b),
                         kappa_direct=q3_eval.cohen_kappa([x == "2" for x in a], [x == "2" for x in b]))
    return primary, pairs.reset_index(), agreement


def load_relevance(paths):
    """Concatenate judged sheets; keep only judgment columns; refuse a row judged in two sheets."""
    relevance = pd.concat([independent.read_csv(p) for p in paths], ignore_index=True)
    if set(RELEVANCE_KEEP) <= set(relevance):
        relevance = relevance[list(RELEVANCE_KEEP)]
    if relevance.pool_id.duplicated().any():
        raise ValueError("Q3: the same pool row is judged in more than one sheet")
    return relevance


def write_scores(dest, per_card, as_of, eligibility=False):
    dest.mkdir(exist_ok=True)
    per_card = per_card.assign(as_of=as_of)
    if eligibility:
        counts = ["n_relevant_in_topk", "n_relevant_incompatible", "n_relevant_uncertain_only"]
        per_card[["card_id", "arm", "k", "as_of", *counts]].to_csv(dest / "eligibility_by_card.csv", **CSV)
        per_card.groupby("arm")[counts].sum().assign(as_of=as_of).to_csv(
            dest / "eligibility_by_arm.csv", encoding="utf-8-sig")
        return
    measures = ["precision_at_k", "ndcg_at_k", "lenient_precision_at_k", "success_at_k", "pooled_recall_at_k"]
    measures = [m for m in measures if m in per_card]
    per_card.to_csv(dest / "by_card.csv", **CSV)
    grouped = per_card.groupby("arm", sort=False)
    avg = grouped[measures].mean()
    avg["n_search_cards"] = grouped.size()
    avg["n_cards_with_relevant"] = grouped.pooled_recall_at_k.count()
    avg.assign(as_of=as_of).to_csv(dest / "arms.csv", encoding="utf-8-sig")
    paired_arms(per_card).assign(as_of=as_of).to_csv(dest / "paired.csv", **CSV)
    if "reviewer" in per_card:
        by_reviewer = per_card.groupby(["reviewer", "arm"], sort=False)[["precision_at_k", "ndcg_at_k"]].mean()
        by_reviewer["n_cards"] = per_card.groupby(["reviewer", "arm"], sort=False).size()
        by_reviewer.assign(as_of=as_of).to_csv(dest / "arms_by_reviewer.csv", encoding="utf-8-sig")
        robust = reviewer_robustness(per_card)
        if not robust.empty:
            robust.assign(as_of=as_of).to_csv(dest / "paired_by_reviewer.csv", **CSV)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    cmd = p.add_subparsers(dest="command", required=True)
    pool = cmd.add_parser("pool")
    pool.add_argument("--cards", type=Path)
    pool.add_argument("--reviews", type=Path)
    pool.add_argument("--design-labels", action="store_true",
                      help="use the 80 persona cards and their design labels instead of --cards/--reviews")
    pool.add_argument("--out", type=Path, required=True, help="new private run directory")
    pool.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    pool.add_argument("--as-of", help="fixed YYYY-MM-DD date (default: config evaluation.q3.as_of)")
    pool.add_argument("--master", type=Path, help="override the service master from the configuration")
    pool.add_argument("--q1-review", type=Path)
    pool.add_argument("--cache", type=Path, default=ROOT / "private/q3_independent_cache")
    pool.add_argument("--pool-depth", type=int, default=POOL_DEPTH,
                      help=f"per-arm depth sent to human review (>= {K})")
    rr = cmd.add_parser("rerank", help="recompute rule arms of a frozen run with other profiles (no GPU)")
    rr.add_argument("--run", type=Path, required=True, help="frozen source run")
    rr.add_argument("--profiles", type=Path, required=True, help=f"CSV with columns {', '.join(PROFILE_COLUMNS)}")
    rr.add_argument("--out", type=Path, required=True, help="new run directory")
    rr.add_argument("--q1-review", type=Path, required=True)
    rr.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    rr.add_argument("--master", type=Path)
    overlap = cmd.add_parser("overlap", help="label-free Top-5 differences (not a quality measure)")
    overlap.add_argument("--run", type=Path, required=True)
    split = cmd.add_parser("split", help="split the blind relevance sheet by assigned reviewer")
    split.add_argument("--run", type=Path, required=True)
    split.add_argument("--shared", type=int, default=2, help="cards per reviewer judged twice for agreement")
    split.add_argument("--arms", nargs="+", choices=ARMS, help="judge only these arms' Top-5 (default: all)")
    split.add_argument("--cards-per-reviewer", type=int, help="fixed random cards per reviewer (default: all)")
    split.add_argument("--name", default="split", help="output folder under review/")
    merge = cmd.add_parser("merge", help="combine reviewer sheets and report agreement")
    merge.add_argument("--run", type=Path, required=True)
    merge.add_argument("--sheets", type=Path, nargs="+", required=True)
    metrics = cmd.add_parser("score")
    metrics.add_argument("--run", type=Path, required=True)
    metrics.add_argument("--relevance", type=Path, nargs="+", required=True,
                         help="one or more judged sheets (e.g. earlier judgments plus rerank's new rows)")
    metrics.add_argument("--eligibility", type=Path)
    metrics.add_argument("--subset", type=Path, help="subset.json written by split (judged cards and arms)")
    args = p.parse_args()
    if args.command == "pool":
        if args.pool_depth < K:
            p.error(f"--pool-depth must be at least {K}")
        if args.design_labels == bool(args.cards or args.reviews) or (not args.design_labels
                                                                       and not (args.cards and args.reviews)):
            p.error("use either --design-labels or both --cards and --reviews")
        prepare_pool(args)
    elif args.command == "split":
        sheet = independent.read_csv(args.run / "review/relevance_review.csv")
        domains_file = AI / yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8"))["paths"]["domains_file"]
        cards, review = independent.persona_forms(yaml.safe_load(domains_file.read_text(encoding="utf-8"))["domains"])
        owners = dict(zip(cards.card_id, zip(cards.author, review.assigned_reviewer)))
        dest = args.run / "review" / args.name
        dest.mkdir(exist_ok=False)
        if args.arms or args.cards_per_reviewer:
            bundle = json.loads((args.run / "restricted/rankings.json").read_text(encoding="utf-8"))
            subset = select_subset(owners, set(bundle["rankings"]), args.cards_per_reviewer or len(owners),
                                   args.arms or ARMS)
            _, needed = subset_rankings(bundle, subset)
            sheet = sheet[sheet.pool_id.isin(needed)]
            (dest / "subset.json").write_text(json.dumps(subset, ensure_ascii=False, indent=2), encoding="utf-8")
        for person, frame in split_review(sheet, owners, args.shared).items():
            frame.to_csv(dest / f"relevance_{person}.csv", **CSV)
            print(person, frame.groupby("assignment").card_id.nunique().to_dict(), len(frame), "rows")
    elif args.command == "merge":
        primary, pairs, agreement = merge_review([independent.read_csv(p) for p in args.sheets])
        dest = args.run / "review/merged"
        dest.mkdir(exist_ok=True)
        primary.to_csv(dest / "relevance_merged.csv", **CSV)
        pairs.to_csv(dest / "agreement_pairs.csv", **CSV)
        (dest / "agreement.json").write_text(json.dumps(agreement, ensure_ascii=False, indent=2), encoding="utf-8")
        print(agreement)
    elif args.command == "rerank":
        rerank(args)
    elif args.command == "overlap":
        bundle = json.loads((args.run / "restricted/rankings.json").read_text(encoding="utf-8"))
        detail, summary = rank_overlap(bundle)
        dest = args.run / "rank_overlap"
        dest.mkdir(exist_ok=True)
        detail.to_csv(dest / "by_card.csv", **CSV)
        summary.to_csv(dest / "summary.csv", encoding="utf-8-sig")
        print(summary.round(3).to_string())
    else:
        bundle = json.loads((args.run / "restricted/rankings.json").read_text(encoding="utf-8"))
        if bundle["schema"] not in SCHEMAS or bundle["eval_k"] != K:
            raise ValueError("Q3: unsupported frozen evaluation protocol")
        if "as_of" not in bundle:
            raise ValueError("Q3: the frozen rankings do not record an as_of date")
        as_of = parse_reference_date(bundle["as_of"]).isoformat()
        relevance = load_relevance(args.relevance)
        # Group the outputs under their reviewed relevance version so later corrections cannot mix stages.
        version = "".join(file_sha256(p) for p in args.relevance)
        dest = args.run / "metrics" / (file_sha256(args.relevance[0]) if len(args.relevance) == 1
                                       else hashlib.sha256(version.encode()).hexdigest())
        (args.run / "metrics").mkdir(exist_ok=True)
        subset = json.loads(args.subset.read_text(encoding="utf-8")) if args.subset else None
        if subset:
            dest = dest.with_name(f"{dest.name}_{file_sha256(args.subset)[:12]}")
        if args.eligibility is None:
            write_scores(dest, score(bundle, relevance, subset=subset), as_of)
        else:
            per_card = score(bundle, relevance, independent.read_csv(args.eligibility), subset=subset)
            eligibility_counts = ["n_relevant_incompatible", "n_relevant_uncertain_only"]
            write_scores(dest, per_card.drop(columns=eligibility_counts), as_of)
            write_scores(dest, per_card, as_of, eligibility=True)


if __name__ == "__main__":
    main()
