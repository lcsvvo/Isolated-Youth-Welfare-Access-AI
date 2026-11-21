"""Q1 주 욕구 우대 가점 크기 비교(개발 질의). 독립 평가 카드 80장은 쓰지 않음.

저장소 루트에서:
  python ai/scripts/q1_bonus_compare.py run --out outputs/12_q1_bonus/<run> \
      --source <Colab 결과 폴더> [--source <노동 Colab 폴더>::ai/data/labor_eval_queries.jsonl]
  python ai/scripts/q1_bonus_compare.py score --run outputs/12_q1_bonus/<run> --review <판정 csv> [<추가 판정 csv> ...]

run: RRF 가중치 비교(combined30)에서 사람이 판정한 개발 질의 30개를 Colab 저장 순위로 재생하고,
     RRF(1:1) → 규칙 → Q1 가점 0·1·2·3점 → Top-5를 계산함. 가점 외 설정은 운영과 같음.
     기존 판정에 없는 (질의, 서비스) 쌍만 review_pool_new.csv로 냄(가점 정보 없음).
score: 가점별 nDCG@5·P@5와 운영값(+1점) 대비 승·패를 계산하고 사전 등록 기준으로 권고를 출력함. config는 수정하지 않음.

사전 등록 판정 기준(결과 보기 전에 고정):
  +1점 외 가점이 nDCG@5 평균 +0.02 이상이고 질의별 승 > 패(대 +1점)일 때만 변경 권고. 아니면 +1점 유지.
  RRF 가중치 비교와 같은 기준. 개발 질의 기반 잠정 판단.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

AI = Path(__file__).resolve().parents[1]
ROOT = AI.parent
sys.path.insert(0, str(AI))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import rrf_weight_compare as rw  # noqa: E402
from rules.filter import _domain_rank, apply_rules  # noqa: E402

BONUSES = (0, 1, 2, 3)
BASE = 1   # 운영값
K = 5
MIN_GAIN, TOL = 0.02, 1e-9
DECISION_RULE = ("+1점 외 가점이 nDCG@5 평균 +0.02 이상, 질의별 승 > 패(대 +1점)일 때만 변경 권고. "
                 "아니면 +1점 유지. 개발 질의 기반 잠정 판단.")
DEV_REVIEW = ROOT / "outputs/11_rrf_weights/combined30/tables/review.csv"
CSV = rw.CSV


def order_with_bonus(judged, services, needs, bonus):
    """apply_rules 결과를 Q1 가점 크기만 바꿔 다시 정렬. bonus=1이면 운영 순서와 같음."""
    pref = {r["servId"]: _domain_rank(services.loc[r["servId"]].to_dict(), needs)[1] for r in judged}

    def key(r):
        p = pref[r["servId"]]
        # 운영 정렬은 (score, -우대). score에는 이미 -1×우대가 들어 있으므로 되돌린 뒤 bonus를 적용함
        return (r["score"] + p - bonus * p, -p if bonus > 0 else 0, r["rank"])
    return [r["servId"] for r in sorted(judged, key=key)]


def parse_source(text):
    """'Colab 폴더' 또는 'Colab 폴더::추가 질의 JSONL'(드라이브 문자와 겹치지 않게 '::' 사용)."""
    colab, _, extra = text.partition("::")
    return Path(colab), (Path(extra) if extra else None)


def run_cmd(args):
    review = pd.read_csv(args.review, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    wanted = sorted(set(review.query_id))
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "tables").mkdir()
    (args.out / "runs").mkdir()
    rows, info, found_q, repro = [], {}, set(), {"current_code_vs_colab_top5_diff": []}
    sources = []
    for spec in args.source:
        colab, extra = parse_source(spec)
        queries, ranks, colab_cards = rw.load_colab(colab.resolve())
        todo = [q for q in wanted if q in queries and q not in found_q]
        if not todo:
            continue
        p = rw.build_pipeline(args.out, args.as_of)
        if "q1_core_domains" not in p.services.columns:
            raise SystemExit("Q1 검수 JSON이 로드되지 않음")
        p.cfg["retrieval"].pop("rrf_weights", None)   # 1:1(운영)
        ids = list(p.services.index)
        p.bm25, p.dense = (rw.Replay(ids, m, ranks, queries) for m in ("bm25", "dense"))
        from eval.stage2_query_review import dev_cases
        records = rw.read_jsonl(rw.CORPUS) + (rw.read_jsonl(extra) if extra else [])
        cases = {f"dev:{c['case_id']}": c for c in dev_cases(records)}
        for qid in todo:
            for m in (p.bm25, p.dense):
                m.current = qid
            out = p.search(copy.deepcopy(cases[qid]["session"]))
            if out.get("retrieval_status") == "error" or "rules_view" not in out:
                raise SystemExit(f"검색 실패 {qid}")
            view = out["rules_view"]
            judged = apply_rules(out["ranks"]["fused"], p.services, view, None)
            needs = view.get("needs", [])
            base = order_with_bonus(judged, p.services, needs, BASE)[:K]
            if base != [c["servId"] for c in out["cards"]]:
                raise SystemExit(f"+1점 재현 실패 {qid}: 가점 재정렬이 파이프라인 Top-5와 다름")
            if base != colab_cards.get(qid, []):
                repro["current_code_vs_colab_top5_diff"].append(qid)
            info[qid] = {"text": cases[qid]["text"], "needs": needs}
            for bonus in BONUSES:
                for rank, sid in enumerate(order_with_bonus(judged, p.services, needs, bonus)[:K], 1):
                    rows.append({"bonus": bonus, "query_id": qid, "rank": rank, "servId": sid,
                                 "servNm": p.services.loc[sid, "servNm"],
                                 "q1_main_preference": _domain_rank(p.services.loc[sid].to_dict(), needs)[1]})
            found_q.add(qid)
        sources.append({"colab": str(colab), "extra_corpus": str(extra) if extra else None,
                        "queries": sorted(set(todo) & found_q),
                        "input_sha256": {f"colab/{x}": rw.sha(colab / "tables" / x)
                                         for x in ("rankings.csv", "queries.csv", "top5_cards.csv")}})
        services = p.services
    missing = sorted(set(wanted) - found_q)
    if missing:
        raise SystemExit(f"판정 질의가 Colab 결과에 없음: {missing}")
    top5 = pd.DataFrame(rows)
    top5.to_csv(args.out / "tables/top5_by_bonus.csv", **CSV)
    judged_pairs = set(zip(review.query_id, review.servId))
    new = top5[[(q, s) not in judged_pairs for q, s in zip(top5.query_id, top5.servId)]]
    new = new.drop_duplicates(["query_id", "servId"]).sort_values(["query_id", "servId"])
    pool = [{"pool_id": f"Q1-{i:04d}", "query_id": r.query_id, "user_text": info[r.query_id]["text"],
             "needs": "|".join(f"{n['domain']}:{n.get('level', 'main')}" for n in info[r.query_id]["needs"]),
             "servId": r.servId, "servNm": r.servNm,
             **{c: services.loc[r.servId].get(c, "") for c in rw.POOL_TEXT},
             "relevance_0_1_2": "", "reason": "", "reviewer": ""}
            for i, r in enumerate(new.itertuples(), 1)]
    pd.DataFrame(pool).to_csv(args.out / "tables/review_pool_new.csv", **CSV)
    changed = top5.groupby("query_id").apply(
        lambda g: len({tuple(x.servId) for _, x in g.groupby("bonus")}) > 1, include_groups=False)
    meta = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "as_of": args.as_of, "bonuses": list(BONUSES),
            "base_bonus": BASE, "rrf_weights": "1:1", "decision_rule": DECISION_RULE,
            "review_source": str(args.review), "review_sha256": rw.sha(args.review), "sources": sources,
            "input_sha256": {str(x.relative_to(ROOT)): rw.sha(x) for x in (rw.CONFIG, rw.Q1, rw.Q2, rw.CORPUS)},
            "code_commit": _commit(), "counts": {"queries": len(found_q), "queries_changed_by_bonus": int(changed.sum()),
                                                  "new_pairs_to_review": len(pool)},
            "reproduction": repro,
            "note": "+1점 재정렬이 현재 코드 파이프라인 Top-5와 일치함을 확인. Colab 저장 Top-5와 다른 질의는 이후 규칙 코드 변경 때문일 수 있음"}
    (args.out / "runs/metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"질의 {len(found_q)}개, 가점에 따라 Top-5가 달라지는 질의 {int(changed.sum())}개, 새 판정 쌍 {len(pool)}개")
    print(f"현재 코드 Top-5가 Colab 저장본과 다른 질의 {len(repro['current_code_vs_colab_top5_diff'])}개")
    return 0


def _commit():
    import subprocess
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def ndcg(rels, pool_rels):
    gain = lambda r: 2 ** r - 1   # noqa: E731
    dcg = sum(gain(r) / math.log2(i + 2) for i, r in enumerate(rels[:K]))
    ideal = sum(gain(r) / math.log2(i + 2) for i, r in enumerate(sorted(pool_rels, reverse=True)[:K]))
    return dcg / ideal if ideal else None


def score(top5, review):
    """가점별 질의 점수. 판정이 비어 있는 Top-5 쌍이 있으면 중단(무관으로 간주하지 않음)."""
    rel = review[review.relevance_0_1_2.isin(["0", "1", "2"])]
    if rel.duplicated(["query_id", "servId"]).any():
        raise ValueError("같은 질의·서비스 판정이 여러 개 있음")
    grade = {(q, s): int(v) for q, s, v in zip(rel.query_id, rel.servId, rel.relevance_0_1_2)}
    missing = {(q, s) for q, s in zip(top5.query_id, top5.servId)} - set(grade)
    if missing:
        raise ValueError(f"판정이 없는 Top-5 쌍 {len(missing)}개. review_pool_new.csv를 판정해 함께 넣어야 함")
    pool = {}
    for (q, _s), g in grade.items():
        pool.setdefault(q, []).append(g)
    rows = []
    for (bonus, qid), g in top5.sort_values("rank").groupby(["bonus", "query_id"]):
        rs = [grade[(qid, s)] for s in g.servId]
        rows.append({"bonus": bonus, "query_id": qid, "ndcg5": ndcg(rs, pool[qid]),
                     "p5": sum(r == 2 for r in rs) / K})
    return pd.DataFrame(rows)


def summarize(per_query):
    ok = per_query[per_query.ndcg5.notna()]
    base = ok[ok.bonus == BASE].set_index("query_id").ndcg5
    out = []
    for bonus, s in ok.groupby("bonus"):
        s = s.set_index("query_id")
        d = (s.ndcg5 - base.reindex(s.index)).dropna()
        out.append({"bonus": bonus, "queries": len(s), "mean_ndcg5": s.ndcg5.mean(), "mean_p5": s.p5.mean(),
                    "diff_vs_base": d.mean(), "wins_vs_base": int((d > TOL).sum()),
                    "losses_vs_base": int((d < -TOL).sum()), "ties_vs_base": int((d.abs() <= TOL).sum()),
                    "undefined_excluded": int(per_query[per_query.bonus == bonus].ndcg5.isna().sum())})
    return pd.DataFrame(out)


def recommend(summary):
    cand = summary[(summary.bonus != BASE) & (summary.diff_vs_base >= MIN_GAIN - TOL)
                   & (summary.wins_vs_base > summary.losses_vs_base)]
    if cand.empty:
        return f"+{BASE}점 유지 (사전 기준 충족 가점 없음)"
    best = cand.sort_values("mean_ndcg5", ascending=False).iloc[0]
    return f"{best.bonus:g}점으로 변경 검토 (nDCG@5 {best.diff_vs_base:+.3f}, 승 {best.wins_vs_base} / 패 {best.losses_vs_base})"


def score_cmd(args):
    top5 = pd.read_csv(args.run / "tables/top5_by_bonus.csv", dtype={"servId": str}, encoding="utf-8-sig")
    review = pd.concat([pd.read_csv(p, dtype=str, keep_default_na=False, encoding="utf-8-sig") for p in args.review],
                       ignore_index=True)
    per_query = score(top5, review)
    summary = summarize(per_query)
    per_query.to_csv(args.run / "tables/bonus_scores_per_query.csv", **CSV)
    summary.to_csv(args.run / "tables/bonus_scores.csv", **CSV)
    print(summary.round(3).to_string(index=False))
    print("권고:", recommend(summary))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--source", action="append", required=True,
                   help="Colab 결과 폴더. 추가 질의 JSONL이 필요하면 '폴더::JSONL'")
    r.add_argument("--review", type=Path, default=DEV_REVIEW, help="기존 개발 질의 사람 판정(질의 목록도 여기서 정함)")
    r.add_argument("--as-of", default=rw.DEFAULT_AS_OF)
    r.add_argument("--out", type=Path, required=True)
    s = sub.add_parser("score")
    s.add_argument("--run", type=Path, required=True)
    s.add_argument("--review", type=Path, nargs="+", required=True)
    args = ap.parse_args(argv)
    return run_cmd(args) if args.cmd == "run" else score_cmd(args)


if __name__ == "__main__":
    raise SystemExit(main())
