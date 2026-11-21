"""Prepare independent Q3 card slots and audit candidate text against development exposure."""

import argparse
import json
from pathlib import Path

import pandas as pd

from src.preprocess.xlsx_reader import read_sheet
from src.recommend import q3_query_check as qc
from src.recommend.q3_run import CSV, ROOT, QUERIES

OUT = ROOT / "outputs/05_q3_independent"
PERSONAS = ROOT / "data/raw/user_input_corpus/persona_cards"
DEVELOPMENT_QUERIES = ROOT / "outputs/06_stage2_query/tables"
CARD_COLUMNS = ("card_id", "scenario", "writer_slot", "input_text", "author", "authorship")
REVIEW_COLUMNS = ("card_id", "expected_action", "semantic_overlap", "decision", "review_evidence",
                  "reviewer", "reviewed_at", "judge_type", "needs", "age", "region_sido",
                  "activity_range", "access_json")
RELEVANCE_COLUMNS = ("pool_id", "relevance", "evidence", "reviewer", "reviewed_at", "judge_type")
ELIGIBILITY_COLUMNS = ("pool_id", "age_status", "region_status", "access_status",
                       "evidence", "reviewer", "reviewed_at", "judge_type")


def read_csv(path):
    return pd.read_csv(path, dtype=str, encoding="utf-8-sig", keep_default_na=False)


def persona_rows():
    files = sorted(PERSONAS.glob("persona_cards_*.xlsx"))
    if len(files) != 4:
        raise ValueError("Q3: expected four persona sheets")
    frames = [read_sheet(p, "작성카드").fillna("").assign(writer_slot=f"R{i}") for i, p in enumerate(files, 1)]
    frame = pd.concat(frames, ignore_index=True)
    frame = frame[frame["카드ID"].ne("")]
    if not frame["카드ID"].is_unique:
        raise ValueError("Q3: duplicate persona card ID")
    return frame


def card_slots():
    p = persona_rows()
    hold = p[p["세트"].eq("평가(사람 작성)")]
    slots = hold.rename(columns={"카드ID": "card_id", "유형": "scenario"})
    if len(slots) != 80 or not slots.groupby("writer_slot").size().eq(20).all():
        raise ValueError("Q3: expected 80 independent slots (20 per writer)")
    return slots[["card_id", "scenario", "writer_slot"]].sort_values("card_id")


def development_bank():
    bank = {}

    def add(source, ident, values):
        for value in values:
            if isinstance(value, str) and qc._clean(value):
                bank.setdefault(qc._clean(value), (source, str(ident), value))

    for ident, value in qc.load_corpus().fillna("").itertuples(index=False, name=None):
        add("corpus_v2", ident, (value,))
    p = persona_rows()
    for ident, value in p.loc[p["세트"].str.startswith("개발"), ["카드ID", "작성글"]].itertuples(index=False, name=None):
        add("persona_dev", ident, (value,))
    for path in (QUERIES.with_name("q3_query_pairs_v1.csv"), QUERIES):
        for r in read_csv(path).itertuples():
            add(path.stem, r.pair_id, (r.daily_query, r.admin_query))
    for name in ("dev_corpus", "dev_cards", "dense_handoff_queries"):
        path = ROOT / "ai/data" / f"{name}.jsonl"
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            if line.strip():
                r = json.loads(line)
                add(name, r["id"], (r.get("text"), r.get("bm25"), r.get("dense")))
    for path in DEVELOPMENT_QUERIES.glob("query_check_*.csv"):
        frame = read_csv(path)
        if not {"case_id", "bm25_query", "dense_query"} <= set(frame):
            continue
        for r in frame.itertuples():
            add(path.stem, r.case_id, (r.original_text, r.bm25_query, r.dense_query))
    return list(bank.values())


def audit_text(cards, bank):
    """Lexical gate; semantic sameness still needs a blinded human review."""
    if not cards.card_id.is_unique:
        raise ValueError("Q3: duplicate card ID")
    rows = []
    authored = [(r.card_id, r.input_text) for r in cards.itertuples() if qc._clean(r.input_text)]
    for r in cards.itertuples():
        best = {"card_id": r.card_id, "status": "awaiting_text", "dev_source": "", "dev_id": "",
                "dev_similarity": None, "card_neighbor": "", "card_similarity": None}
        if qc._clean(r.input_text):
            matches = [(qc.similarity(r.input_text, text), source, ident) for source, ident, text in bank]
            dev_sim, source, ident = max(matches, default=(0, "", ""))
            card_sim, card_id = max(((qc.similarity(r.input_text, t), cid) for cid, t in authored if cid != r.card_id),
                                    default=(0, ""))
            best.update(status="rewrite_required" if max(dev_sim, card_sim) >= qc.SIM_MAX else "human_semantic_review",
                        dev_source=source, dev_id=ident, dev_similarity=round(dev_sim, 6),
                        card_neighbor=card_id, card_similarity=round(card_sim, 6))
        rows.append(best)
    return pd.DataFrame(rows)


def load_assignment(cards_path, reviews_path):
    cards, reviews = read_csv(cards_path), read_csv(reviews_path)
    required_slots = card_slots().set_index("card_id")
    key = ["scenario", "writer_slot"]
    if not set(CARD_COLUMNS) <= set(cards) or not set(REVIEW_COLUMNS) <= set(reviews):
        raise ValueError("Q3: missing card or review columns")
    current_slots = cards.set_index("card_id")[key]
    if (not current_slots.index.is_unique or not reviews.card_id.is_unique
            or not current_slots.sort_index().equals(required_slots[key].sort_index())
            or set(reviews.card_id) != set(cards.card_id)):
        raise ValueError("Q3: card IDs, scenario, or writer assignment do not match the 80 slots")
    return cards, reviews


def prepare_replacements(cards_path, reviews_path, card_ids, out):
    """Open the same scenario slots for new drafts and reviews in a new directory."""
    cards, reviews = load_assignment(cards_path, reviews_path)
    requested = set(card_ids)
    replaceable = set(reviews.loc[reviews.decision.isin({"revise", "exclude"}), "card_id"])
    if not requested or not requested <= replaceable:
        raise ValueError("Q3: replacements require an existing revise or exclude decision")
    cards.loc[cards.card_id.isin(requested), ["input_text", "author", "authorship"]] = ""
    reviews.loc[reviews.card_id.isin(requested), list(REVIEW_COLUMNS[1:])] = ""
    out.mkdir(parents=True, exist_ok=False)
    cards.to_csv(out / "cards_to_write.csv", **CSV)
    reviews.to_csv(out / "card_review.csv", **CSV)
    print(f"Q3: reopened {len(requested)} slot(s) for a new draft and independent review")


EVAL_SET = "평가(사람 작성)"
ACTION_MAP = {"추천": "search", "추천(자격 확인 필요 표시)": "search", "추천(활동 범위 과대판정 금지)": "search",
              "되묻기": "clarify", "추천 중단·상담 안내": "stop"}
ACCESS_MAP = {"없음": [], "방문 부담(선호)": [("방문", "부담")], "방문 불가(필수 제한)": [("방문", "불가")],
              "전화 부담(선호)": [("전화", "부담")]}
UNKNOWN = "판단 불가"


def _access_json(label):
    items = [{"polarity": "avoid", "channel": ch, "level": lv, "scope": None, "mode": None,
              "pattern_id": None, "source": "persona_design"} for ch, lv in ACCESS_MAP[label]]
    return json.dumps(items, ensure_ascii=False)


def persona_forms(domains):
    """Fill the card and review forms from the 80 human-written cards and their design labels.

    The labels were fixed before writing (정답_* columns). Age and region enter the profile only when the
    text mentions them. Review decision fields stay blank for the assigned cross-reviewer.
    """
    p = persona_rows()
    hold = p[p["세트"].eq(EVAL_SET)].sort_values("카드ID")
    slots = card_slots().set_index("card_id")
    unknown = {
        "기대행동": set(hold["기대행동"]) - set(ACTION_MAP),
        "정답_제약": set(hold["정답_제약"]) - set(ACCESS_MAP),
        "정답_활동범위": set(hold["정답_활동범위"]) - {"방 안", "집 안", "집 근처", "필요하면 외출", UNKNOWN},
        "정답_영역": {d.strip() for v in hold["정답_영역"] for d in v.split(";")} - set(domains) - {UNKNOWN},
    }
    if any(unknown.values()):
        raise ValueError(f"Q3: unmapped persona labels {({k: v for k, v in unknown.items() if v})}")
    if hold["작성글"].str.strip().eq("").any() or hold["작성자"].str.strip().eq("").any():
        raise ValueError("Q3: persona card without text or writer")
    if hold["검토자"].eq(hold["작성자"]).any():
        raise ValueError("Q3: assigned reviewer must differ from the writer")

    def needs(value):
        return "|".join(d.strip() for d in value.split(";") if d.strip() and d.strip() != UNKNOWN)

    cards = pd.DataFrame({
        "card_id": hold["카드ID"], "scenario": hold["카드ID"].map(slots["scenario"]),
        "writer_slot": hold["카드ID"].map(slots["writer_slot"]),
        "input_text": hold["작성글"].str.strip(), "author": hold["작성자"], "authorship": "human",
    })
    review = pd.DataFrame({
        "card_id": hold["카드ID"], "expected_action": hold["기대행동"].map(ACTION_MAP),
        "semantic_overlap": "", "decision": "", "review_evidence": "", "reviewer": "", "reviewed_at": "",
        "judge_type": "", "needs": hold["정답_영역"].map(needs),
        "age": hold["나이"].where(hold["정답_나이언급"].eq("Y"), "").astype(str),
        "region_sido": hold["시도"].where(hold["정답_지역언급"].eq("Y"), ""),
        "activity_range": hold["정답_활동범위"], "access_json": hold["정답_제약"].map(_access_json),
        "assigned_reviewer": hold["검토자"], "design_action": hold["기대행동"], "design_crisis": hold["정답_위기"],
        "design_age": hold["나이"].astype(str), "design_region_sido": hold["시도"],
        "prefill_source": "persona_design_labels",
    })
    return cards.reset_index(drop=True), review.reset_index(drop=True)


def prepare_from_personas(out, domains):
    """Write pre-filled forms to a new directory and audit the real card text."""
    cards, review = persona_forms(domains)
    out.mkdir(parents=True, exist_ok=False)
    cards.to_csv(out / "cards_to_write.csv", **CSV)
    review.to_csv(out / "card_review.csv", **CSV)
    audit = audit_text(cards, development_bank())
    audit.to_csv(out / "overlap_audit.csv", **CSV)
    print(f"{len(cards)} persona cards; actions {review.expected_action.value_counts().to_dict()}; "
          f"audit {audit.status.value_counts().to_dict()}")
    return cards, review, audit


def prepare(out=OUT):
    review = out / "review"
    slots = card_slots()
    cards = slots.assign(input_text="", author="", authorship="")
    checks = slots[["card_id"]].reindex(columns=REVIEW_COLUMNS, fill_value="")
    review.mkdir(parents=True, exist_ok=True)
    for name, frame in {
        "cards_to_write": cards, "card_review": checks,
        "relevance_review_schema": pd.DataFrame(columns=RELEVANCE_COLUMNS),
        "eligibility_review_schema": pd.DataFrame(columns=ELIGIBILITY_COLUMNS),
    }.items():
        path = review / f"{name}.csv"
        if not path.exists():
            frame.to_csv(path, **CSV)
    bank = development_bank()
    audit = audit_text(read_csv(review / "cards_to_write.csv"), bank)
    tables = out / "tables"
    tables.mkdir(exist_ok=True)
    audit.to_csv(tables / "overlap_audit.csv", **CSV)
    print(f"{len(cards)} writing slots; {len(bank)} distinct dev texts/queries; {audit.status.value_counts().to_dict()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--cards", type=Path)
    parser.add_argument("--reviews", type=Path)
    parser.add_argument("--replace", nargs="+", metavar="CARD_ID")
    parser.add_argument("--from-personas", action="store_true",
                        help="fill forms from the 80 persona cards into a new --out directory")
    args = parser.parse_args()
    if args.from_personas:
        if not args.out or args.cards or args.reviews or args.replace:
            parser.error("--from-personas takes only a new --out directory")
        import yaml
        domains_file = ROOT / "ai/configs/domains.yaml"
        prepare_from_personas(args.out, yaml.safe_load(domains_file.read_text(encoding="utf-8"))["domains"])
    elif args.replace:
        if not (args.cards and args.reviews and args.out):
            parser.error("--replace requires --cards, --reviews and a new --out directory")
        prepare_replacements(args.cards, args.reviews, args.replace, args.out)
    else:
        if args.cards or args.reviews:
            parser.error("--cards and --reviews are used with --replace")
        prepare(args.out or OUT)
