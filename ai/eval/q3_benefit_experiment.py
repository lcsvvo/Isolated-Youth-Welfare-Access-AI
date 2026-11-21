"""Compare explicit benefit types and a local Cross-Encoder on development queries."""

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from ai.eval import independent_retrieval as q3
from ai.eval import q3_top5_audit as audit
from ai.eval import q3_top5_dev as dev
from retrieve.benefit import WINDOW, rank_by_benefit_type
from retrieve.bm25 import build_texts


MAX_TOKENS = 256
BATCH_SIZE = 6


class LocalCrossEncoder:
    def __init__(self, model_dir):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device == "cpu":
            torch.set_num_threads(min(6, torch.get_num_threads()))
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, trust_remote_code=False)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_dir, local_files_only=True, trust_remote_code=False, use_safetensors=True,
        ).to(self.device).eval()

    def order(self, text, ids, passages):
        ids = [sid for sid in ids if passages[sid].strip()]
        scores = {}
        with self.torch.inference_mode():
            for start in range(0, len(ids), BATCH_SIZE):
                batch = ids[start:start + BATCH_SIZE]
                encoded = self.tokenizer(
                    [text] * len(batch), [passages[sid] for sid in batch],
                    padding=True, truncation="longest_first", max_length=MAX_TOKENS, return_tensors="pt",
                ).to(self.device)
                logits = self.model(**encoded).logits
                if logits.shape != (len(batch), 1) or not self.torch.isfinite(logits).all().item():
                    raise ValueError("Expected one finite relevance logit per query–benefit pair")
                scores.update(zip(batch, logits[:, 0].cpu().tolist()))
        return sorted(ids, key=scores.__getitem__, reverse=True)


def load_supplement(path):
    rows = q3.independent.read_csv(path)
    if (not rows.pool_id.is_unique or not rows.judge_type.eq("human").all()
            or not rows.relevance.isin(["0", "1", "2"]).all()):
        raise ValueError("Expected distinct human 0/1/2 judgments")
    return {r.pool_id: (r.card_id, int(r.relevance)) for r in rows.itertuples()}


def run(root, out, model_dir, reviewed_precision):
    master = root / "data/processed/services_master.csv"
    q1 = root / "outputs/03_q1_recommendation/q1_review.json"
    services = q3.RecommendationReview(master, q1_path=q1).services
    passages = dict(zip(services.index, build_texts(services, fields=(("benefit_text", 1),))))
    meta = audit.read_json(root / "outputs/12_q1_bonus/run1/runs/metadata.json")
    review_path = root / "outputs/11_rrf_weights/combined30/tables/review.csv"
    if q3.file_sha256(review_path) != meta["review_sha256"]:
        raise ValueError("Development labels differ from the archived run")
    judged = q3.independent.read_csv(review_path)
    grades = {card: dict(zip(g.servId, g.relevance_0_1_2.astype(int))) for card, g in judged.groupby("query_id")}
    extra = load_supplement(reviewed_precision)
    model = LocalCrossEncoder(model_dir)
    print(f"Cross-Encoder device: {model.device}", flush=True)
    rows, texts, queues = [], {}, {"benefit_type": {}, "cross_encoder": {}}
    for card, text, _, ranks, kept in dev.dev_rerank_cases(root, services, master, meta):
        marks = grades[card]
        for sid in ranks["rrf"]:
            key = hashlib.sha256(f"q3-v1:{card}:{sid}".encode()).hexdigest()[:20]
            if key in extra:
                judged_card, grade = extra.pop(key)
                if judged_card != card or sid in marks:
                    raise ValueError(f"Supplemental judgment disagrees with the development pool: {key}")
                marks[sid] = grade
        window = kept[:WINDOW]
        ids = [r["servId"] for r in window]
        before = set(ids[:audit.K])
        matched, types = rank_by_benefit_type(kept, text, services)
        semantic = dev.benefit_rerank(window, model.order(text, ids, passages))
        texts[card] = text
        for name, ordered in (("benefit_type", matched), ("cross_encoder", semantic)):
            after = {r["servId"] for r in ordered[:audit.K]}
            unknown_before, unknown_after = before - marks.keys(), after - marks.keys()
            removed, added = unknown_before - unknown_after, unknown_after - unknown_before
            queues[name][card] = removed | added
            old_hits = sum(marks.get(sid) == 2 for sid in before)
            new_hits = sum(marks.get(sid) == 2 for sid in after)
            delta = new_hits - old_hits
            rows.append(dict(
                query_id=card, method=name, request_types=types, top5_changed=before != after,
                base_known_hits=old_hits, trial_known_hits=new_hits,
                base_unjudged=len(unknown_before), trial_unjudged=len(unknown_after),
                changed_unjudged=len(removed | added),
                delta_lower_hits=delta - len(removed), delta_upper_hits=delta + len(added),
            ))
        if len(texts) % 10 == 0:
            print(f"Scored {len(texts)} dev queries", flush=True)
    if extra or texts.keys() != grades.keys() or len(texts) != 30:
        raise ValueError("Expected all 30 development queries and all supplemental judgments")
    out.mkdir(parents=True, exist_ok=False)
    tables, review = out / "tables", out / "review"
    tables.mkdir()
    review.mkdir()
    by_query = pd.DataFrame(rows)
    by_query.to_csv(tables / "by_query.csv", **q3.CSV)
    summary = []
    for method, group in by_query.groupby("method"):
        n = len(group)
        total = group.sum(numeric_only=True)
        summary.append(dict(
            method=method, queries=n, changed_queries=int(total.top5_changed),
            known_hits=int(total.trial_known_hits), baseline_known_hits=int(total.base_known_hits),
            changed_unjudged=int(total.changed_unjudged),
            delta_p5_lower=total.delta_lower_hits / (n * audit.K),
            delta_p5_upper=total.delta_upper_hits / (n * audit.K),
        ))
        if any(queues[method].values()):
            dev.write_review_form(review / f"{method}_changed_unjudged.csv", queues[method], services, texts)
    results = pd.DataFrame(summary)
    results.to_csv(tables / "summary.csv", **q3.CSV)
    (out / "experiment.json").write_text(json.dumps({
        "model_dir": str(model_dir), "model_sha256": q3.file_sha256(model_dir / "model.safetensors"),
        "device": model.device, "window": WINDOW, "max_tokens": MAX_TOKENS, "batch_size": BATCH_SIZE,
        "master_sha256": q3.file_sha256(master), "supplement_sha256": q3.file_sha256(reviewed_precision),
    }, indent=2) + "\n", encoding="utf-8")
    print(results.to_string(index=False))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for option in ("data-root", "model-dir", "reviewed-precision", "out"):
        cli.add_argument(f"--{option}", type=Path, required=True)
    args = cli.parse_args()
    run(args.data_root, args.out, args.model_dir, args.reviewed_precision)
