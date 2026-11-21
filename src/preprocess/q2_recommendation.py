"""Export a sourced Q2 addendum for the reviewed identical SSIS/Gov24 pairs."""

import argparse
import json
from pathlib import Path

import pandas as pd

from src.preprocess import q2_completeness as C, q2_items_v3 as V, q2_match as M

ROOT, MASTER = M.ROOT, M.MASTER
OUT = ROOT / "outputs/04_q2_recommendation"
FINAL = ROOT / "outputs/04_q2/tables/final"
LOG = ROOT / "logs/validation/q2_final"
# SSIS snapshot reused during the Q2 source/region recheck before the Q2 final tables.
REVIEWED_MASTER = "269005564d843cb53988422465e1ba56824341d8a26c3d1b655908ca6ef83d6e"
ITEMS = tuple(C.ITEMS6)
SSIS_FIELDS = {
    "age": ("target_text_raw", "criteria_text_raw"),
    "income": ("target_text_raw", "criteria_text_raw"),
    "channel": ("apply_text_raw", "apply_cat", "online_bokjiro"),
    "period": ("apply_text_raw",),
    "documents": ("apply_text_raw", "form_json"),
    "contact": ("apply_text_raw", "contact_json", "inq_num"),
}
GOV_FIELDS = {
    "age": ("지원대상", "선정기준"),
    "income": ("지원대상", "선정기준"),
    "channel": ("신청방법", "신청방법_분류", "온라인신청사이트URL", "접수기관"),
    "period": ("신청기한", "신청기한유형", "신청방법"),
    "documents": ("구비서류", "본인확인필요구비서류", "공무원확인구비서류", "신청방법"),
    "contact": ("문의처", "전화문의"),
}


def confirmed(adjudication, candidates, root=ROOT):
    if not adjudication.item_id.is_unique or len(adjudication) != 72:
        raise ValueError("Q2: unexpected adjudication keys")
    decisions = C.confirmed_pairs(adjudication, root=root)
    if len(decisions) != len(adjudication):
        raise ValueError("Q2: adjudication and assignment do not agree")
    post = C.post_review(root)
    check = post.merge(decisions, on=["servId", "gov_id"], suffixes=("_source", ""), validate="one_to_one")
    if len(check) != len(post) or not check.human_final.eq(check.human_final_source).all():
        raise ValueError("Q2: post-review decision has no matching original decision")
    reviewed = decisions[decisions.final.eq("동일")].copy()
    if not reviewed.final_source.isin({"단독", "일치", "조정"}).all():
        raise ValueError("Q2: final same-program decision lacks human review")
    pairs = reviewed.merge(candidates[["servId", "gov_id", "tier", "gov_shared", "servNm", "gov_name"]],
                           on=["servId", "gov_id"], validate="one_to_one")
    if len(pairs) != len(reviewed) or not pairs.tier.isin({"A", "B", "C", "U"}).all():
        raise ValueError("Q2: reviewed keys do not match the source candidate set")
    # The published 38 are the A/B/C analysis population; Tier U is outside that population.
    pairs = pairs[pairs.tier.isin({"A", "B", "C"})]
    if len(pairs) != 38 or not pairs.servId.is_unique or pairs.gov_id.eq("").any():
        raise ValueError("Q2: expected 38 reviewed pairs with unique SSIS IDs")
    return pairs.sort_values("servId")


def guidance_status(item, ssis, gov, verdict, raw_compare=""):
    if verdict == "실제 충돌":
        return "source_conflict"
    if verdict == "확인 불가" or (raw_compare == "상이 후보" and not verdict):
        return "source_check_required"
    if item == "period":
        return "freshness_check_required"
    if ((gov["explicitness"] == "명시" and ssis["explicitness"] not in {"명시", "해당없음"})
            or verdict == "상호 보완"):
        return "supplement_possible"
    if ssis["explicitness"] in {"명시", "해당없음"}:
        return "ssis_information_present"
    return "insufficient_evidence"


def fields(record, keys):
    return {k: record[k] for k in keys if isinstance(record[k], str) and record[k].strip()}


def unique_index(frame, keys):
    indexed = frame.set_index(keys)
    if not indexed.index.is_unique:
        raise ValueError(f"Q2: duplicate source keys ({keys})")
    return indexed


def side_evidence(record, keys, label, human_text, **provenance):
    return {
        **provenance, "explicitness": label["explicitness"],
        "rule_id": label["rule_id"], "rule_evidence": label["evidence"],
        "human_text_explicitness": human_text,
        "source_fields": fields(record, keys),
    }


def main(out=OUT):
    master_hash = V.sha256(MASTER)
    if master_hash != REVIEWED_MASTER:
        raise ValueError("Revalidate Q2 identity and evidence on the new SSIS snapshot")
    items_path = M.PROCESSED / "q2_items_v3_1.csv"
    match_path = M.PROCESSED / "q2_match.csv"
    adj_path = FINAL / "adjudication_match.csv"
    review_path = LOG / "content_conflict_review.csv"
    samples_path = ROOT / "logs/validation/q2_item_rules/item_gold_sample.csv"
    adj_items_path = FINAL / "adjudication_item.csv"
    pairs = confirmed(V.read(adj_path), V.read(match_path))
    qmeta = json.loads((M.PROCESSED / "q2_match.meta.json").read_text(encoding="utf-8"))
    for path in (M.GOV_LIST, M.GOV_DETAIL):
        if V.sha256(path) != qmeta["inputs_sha256"][path.relative_to(ROOT).as_posix()]:
            raise ValueError("Q2: Gov24 evidence snapshot differs from the reviewed matching input")

    master = unique_index(V.read(MASTER), "servId")
    gov = unique_index(M.load_gov24(), "서비스ID")
    details = unique_index(V.read(M.GOV_DETAIL), "서비스ID")
    gov = gov.join(details[["본인확인필요구비서류", "공무원확인구비서류"]], validate="one_to_one").fillna("")
    if (not pairs.servId.map(master.servNm).eq(pairs.servNm).all()
            or not pairs.gov_id.map(gov["서비스명"]).eq(pairs.gov_name).all()):
        raise ValueError("Q2: source names or IDs differ from reviewed matching")
    exact_keys = ["servId", "gov_id", "item", "side"]
    v31 = V.read(items_path)
    v31 = v31[v31.scope.eq("all") & v31.item.isin(ITEMS)].copy()
    v31["explicitness"] = V.label(v31)
    v31 = unique_index(v31, exact_keys)
    samples = V.read(samples_path)
    adj_items = V.read(adj_items_path).query("status == 'confirmed'")
    text_checks = unique_index(samples.merge(adj_items[["item_id", "final"]], left_on="sample_id", right_on="item_id",
                                             validate="one_to_one"), exact_keys)["final"]
    compare = unique_index(V.read(review_path), exact_keys[:3])
    gov_collection = M.GOV_LIST.with_name("_collection.json")
    collection = json.loads(gov_collection.read_text(encoding="utf-8-sig"))

    reports, records = [], {}
    for r in pairs.itertuples(index=False):
        s, g = master.loc[r.servId], gov.loc[r.gov_id]
        detail = {}
        for item in ITEMS:
            key = (r.servId, r.gov_id, item)
            ssis = side_evidence(s, SSIS_FIELDS[item], v31.loc[(*key, "ssis")],
                                 text_checks.get((*key, "ssis"), ""), url=s.detail_url,
                                 collected_at=s.collected_at, source_last_modified=s.last_mod)
            gov24 = side_evidence(g, GOV_FIELDS[item], v31.loc[(*key, "gov")],
                                  text_checks.get((*key, "gov"), ""), url=g["상세조회URL"],
                                  collected_at=collection["collected_at"], source_last_modified=g["수정일시"])
            review = compare.loc[key] if key in compare.index else {}
            verdict = review.get("final_verdict", "")
            status = guidance_status(item, ssis, gov24, verdict, review.get("compare", ""))
            info = {"status": status, "freshness": "latest_not_verified", "auto_fill": False,
                    "review": {c: review.get(c, "") for c in (*C.REVIEW_COLS[:-1], "basis")},
                    "ssis": ssis, "gov24": gov24}
            detail[item] = info
            reports.append({"servId": r.servId, "gov_id": r.gov_id, "item": item,
                            "status": status, "freshness": info["freshness"], "auto_fill": False,
                            **info["review"], **{f"{side}_{field}": (
                                json.dumps(value, ensure_ascii=False) if isinstance(value, dict) else value)
                                for side in ("ssis", "gov24") for field, value in info[side].items()}})
        records[r.servId] = {"gov_id": r.gov_id, "match_final_source": r.final_source,
                              "same_program_confirmed": True, "items": detail}

    tables = out / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    pairs.to_csv(tables / "confirmed_ssis_gov24.csv", index=False, encoding="utf-8-sig")
    frame = pd.DataFrame(reports)
    frame.to_csv(tables / "field_evidence.csv", index=False, encoding="utf-8-sig")
    frame.groupby(["item", "status"]).size().rename("n").reset_index().to_csv(
        tables / "field_status_counts.csv", index=False, encoding="utf-8-sig")
    inputs = [items_path, match_path, adj_path, review_path, samples_path, adj_items_path,
              LOG / "match_post_review.csv", ROOT / "logs/validation/q2_match/review_assignment.csv",
              M.GOV_LIST, M.GOV_DETAIL, gov_collection]
    (out / "q2_info.json").write_text(json.dumps({
        "schema": "q2-info-v1", "master_sha256": master_hash,
        "inputs_sha256": {p.relative_to(ROOT).as_posix(): V.sha256(p) for p in inputs},
        "gov24_collection_metadata": collection, "services": records,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{len(records)} confirmed pairs; {len(frame)} field rows")
    print(frame.status.value_counts().to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    main(parser.parse_args().out)
