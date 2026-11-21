"""RRF 가중치(BM25:Dense) 비교. Colab에서 저장한 BGE-M3 core 실제 순위를 그대로 재생해 GPU 없이 실행.

저장소 루트에서:
  python ai/scripts/rrf_weight_compare.py run --colab <Colab 결과 폴더>
  python ai/scripts/rrf_weight_compare.py score --run outputs/11_rrf_weights/<run> --review <사람 검수 완료 review_pool.csv>

run: Colab 순위(tables/rankings.csv의 dense·bm25 Top-100)를 재생 검색기로 Pipeline에 넣고,
     가중치별로 RRF → 규칙 → Q1 우대 → Top-5를 다시 계산함. 모델·GPU 불필요(MockLLM 설명문).
     1:1은 운영 설정(가중치 없음) 그대로이며 Colab fused·Top-5와 일치해야 함(재현 확인, 불일치 시 종료 코드 1).
     가중치별 Top-5가 다른 질의에서 주 욕구 영역 고정 배분 표본(SAMPLE_QUOTA) 또는 --pool-queries 지정 질의
     → review_pool.csv·xlsx(가중치 정보 없음), 대응표 review_key.csv.
     ai_relevance_draft·ai_reason은 AI 초안 열이며 score는 사람 판정 relevance_0_1_2만 씀.
score: 사람이 채운 relevance_0_1_2(0/1/2)로 가중치별 nDCG@5, P@5(관련성 2 비율)를 계산하고 권고를 출력함.
       config는 수정하지 않음.

사전 등록 판정 기준(결과 보기 전에 고정):
  1:1 외 조합이 nDCG@5 평균 +0.02 이상, 질의별 승 > 패(대 1:1)일 때만 변경 권고. 아니면 1:1 유지.
  4번째 기준 없음. 개발 질의 기반 잠정 판단.
출력: outputs/11_rrf_weights/<run>/tables/*.csv, runs/metadata.json (Git에 올리지 않음)
선행: 가중치 지원은 Pipeline이 config retrieval.rrf_weights({"bm25": a, "dense": b})를 RRF에 넘기는 코드가 있어야 함.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

AI = Path(__file__).resolve().parents[1]
ROOT = AI.parent
sys.path.insert(0, str(AI))

import pipeline as P  # noqa: E402
from rules.filter import _domain_rank  # noqa: E402

CONFIG = AI / "configs" / "config.yaml"
CORPUS = AI / "data" / "dev_corpus.jsonl"
Q1 = ROOT / "outputs/03_q1_recommendation/q1_review.json"
Q2 = ROOT / "outputs/04_q2_recommendation/q2_info.json"
PROFILE = "core"
DEFAULT_AS_OF = "2026-11-16"   # Colab 실행일
GRID = [(1, 1), (1, 1.5), (1, 2), (1.5, 1), (2, 1)]   # 극단값 1:3·3:1 제외(검수 부담)
SAMPLE_SEED = 20261116
# 영역별 개발 평가셋(주 욕구 기준 고정 배분). 실제 이용자 분포가 아니라 영역별 검색 품질을 고루 보기 위한 할당.
# 노동(일하는 중)은 개발 질의가 없어 별도 추가 전까지 0
SAMPLE_QUOTA = {"주거": 4, "경제": 4, "미취업(구직)": 4, "교육훈련": 4, "미래설계": 4, "건강": 3, "관계참여": 3,
                "일상·여가": 1, "노동(일하는 중)": 0}
BASE = "1:1"
MIN_GAIN, TOL = 0.02, 1e-9
DECISION_RULE = ("1:1 외 조합이 nDCG@5 평균 +0.02 이상, 질의별 승 > 패(대 1:1)일 때만 변경 권고. "
                 "아니면 1:1 유지. 4번째 기준 없음. 개발 질의 기반 잠정 판단.")
CSV = {"index": False, "encoding": "utf-8-sig"}
POOL_TEXT = ["summary_text", "benefit_text", "target_text", "apply_text"]


def sha(path) -> str | None:
    path = Path(path)
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def label(a, b) -> str:
    return f"{a:g}:{b:g}"


def read_jsonl(path):
    return [json.loads(x) for x in Path(path).read_text(encoding="utf-8-sig").splitlines() if x.strip()]


class Replay:
    """Colab에서 저장한 순위를 질의별로 돌려주는 검색기. 질의 문장이 저장본과 다르면 중단."""

    def __init__(self, ids, method, ranks, queries):
        self.ids, self.method, self.ranks, self.queries = ids, method, ranks, queries
        self.current = None

    def search(self, query, k=100):
        want = self.queries[self.current][self.method]
        if query != want:
            raise SystemExit(f"질의 불일치 {self.current} {self.method}: 실행 질의가 Colab 저장본과 다름")
        return list(self.ranks[(self.current, self.method)][:k])


def load_colab(colab: Path):
    t = colab / "tables"
    q = pd.read_csv(t / "queries.csv", dtype=str, keep_default_na=False, encoding="utf-8-sig")
    q = q[(q.profile == PROFILE) & q.query_id.str.startswith("dev:")]
    queries = {r.query_id: {"bm25": r.bm25_query, "dense": r.dense_query} for r in q.itertuples()}
    r = pd.read_csv(t / "rankings.csv", dtype={"servId": str}, encoding="utf-8-sig")
    r = r[(r.profile == PROFILE) & r.query_id.str.startswith("dev:")].sort_values(["query_id", "method", "rank"])
    ranks = {k: g.servId.tolist() for k, g in r.groupby(["query_id", "method"])}
    c = pd.read_csv(t / "top5_cards.csv", dtype={"servId": str}, encoding="utf-8-sig")
    c = c[(c.profile == PROFILE) & c.query_id.str.startswith("dev:")].sort_values(["query_id", "rank"])
    cards = {k: g.servId.tolist() for k, g in c.groupby("query_id")}
    return queries, ranks, cards


def build_pipeline(run_dir, as_of):
    """기본 config 그대로. 단 mock Dense가 실제 캐시를 만들지 않게 캐시 경로만 실행 폴더로 바꾼 사본을 씀."""
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    cfg["paths"]["dense_cache"] = str((run_dir / "cache" / "service_emb_{profile}.npz").resolve())
    path = run_dir / "runs" / "config_replay.yaml"
    path.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return P.Pipeline(config_path=path, mock=True, q1_review_path=Q1, q2_info_path=Q2, as_of=as_of)


def weights_supported() -> bool:
    import inspect
    from retrieve import rrf
    return "weights" in inspect.signature(rrf.rrf_detail).parameters


def run_cmd(args):
    colab = args.colab.resolve()
    tag = datetime.now().strftime("%Y%m%d_%H%M")
    run_dir = args.out or ROOT / "outputs" / "11_rrf_weights" / tag
    (run_dir / "tables").mkdir(parents=True, exist_ok=True)
    (run_dir / "runs").mkdir(parents=True, exist_ok=True)

    queries, ranks, colab_cards = load_colab(colab)
    print(f"Colab {PROFILE} 개발 질의 {len(queries)}개")
    p = build_pipeline(run_dir, args.as_of)
    if p.cfg["retrieval"].get("dense_profile") != PROFILE:
        raise SystemExit(f"config dense_profile이 {PROFILE}이 아님")
    if "q1_core_domains" not in p.services.columns or p.review.q1 is None:
        raise SystemExit("Q1 검수 JSON이 로드되지 않음(q1_core_domains 없음)")
    if p.review.q2 is None:
        raise SystemExit("Q2 JSON이 로드되지 않음")
    ids = list(p.services.index)
    p.bm25, p.dense = (Replay(ids, m, ranks, queries) for m in ("bm25", "dense"))

    from eval.stage2_query_review import dev_cases
    from query.build import build_queries
    records = read_jsonl(CORPUS) + (read_jsonl(args.extra_corpus) if args.extra_corpus else [])
    if len({r["id"] for r in records}) != len(records):
        raise SystemExit("개발 말뭉치와 추가 질의 id 중복")
    cases = {f"dev:{c['case_id']}": c for c in dev_cases(records)}
    missing = sorted(set(queries) - set(cases))
    if missing:
        raise SystemExit(f"Colab 질의가 개발 세트에 없음: {missing[:5]}")
    info = {}
    for qid in sorted(queries):
        case = cases[qid]
        try:
            q = build_queries(case["session"], p.terms, p.cfg)
        except ValueError as e:
            raise SystemExit(f"build_queries 실패 {qid}: {e}")
        if (q["bm25"], q["dense"]) != (queries[qid]["bm25"], queries[qid]["dense"]):
            raise SystemExit(f"build_queries 결과가 Colab queries.csv와 다름: {qid}")
        needs = case["session"]["confirmed_profile"]["needs"]
        info[qid] = {"text": case["text"], "needs": needs, "access": json.dumps(q["access"], ensure_ascii=False)}
    not_run = sorted(set(cases) - set(queries))
    print(f"질의 문장 일치 {len(info)}개 (Colab에 없는 개발 케이스 {len(not_run)}개는 제외)")

    top5, results, errors = [], {}, {}
    for a, b in GRID:
        w = label(a, b)
        if (a, b) == (1, 1):
            p.cfg["retrieval"].pop("rrf_weights", None)
        else:
            p.cfg["retrieval"]["rrf_weights"] = {"bm25": a, "dense": b}
        try:
            for qid in sorted(queries):
                for m in (p.bm25, p.dense):
                    m.current = qid
                out = p.search(copy.deepcopy(cases[qid]["session"]))
                if "ranks" not in out or out.get("retrieval_status") == "error":
                    raise SystemExit(f"검색 실패 {w} {qid}: {out.get('retrieval_status')} {out.get('retrieval_errors')}")
                rrf = {r["servId"]: r for r in out["ranks"]["rrf"]}
                results[(w, qid)] = {"cards": [c["servId"] for c in out["cards"]], "fused": out["ranks"]["fused"],
                                     "n_excluded": len(out["excluded"])}
                needs = info[qid]["needs"]
                for rank, c in enumerate(out["cards"], 1):
                    sid, svc = c["servId"], p.services.loc[c["servId"]].to_dict()
                    d = rrf.get(sid, {})
                    top5.append({"weight": w, "query_id": qid, "rank": rank, "servId": sid, "servNm": c["servNm"],
                                 "bm25_rank": (d.get("ranks") or {}).get("bm25"),
                                 "dense_rank": (d.get("ranks") or {}).get("dense"),
                                 "rrf_score": d.get("rrf_score"), "q1_main_preference": _domain_rank(svc, needs)[1],
                                 "checks": json.dumps(c["checks"], ensure_ascii=False)})
        except TypeError as e:
            errors[w] = f"TypeError: {e}"
            print(f"[가중치 미지원] {w}: {errors[w]}")
    p.cfg["retrieval"].pop("rrf_weights", None)

    # 재현 확인(1:1 = 운영)
    mism = []
    for qid in sorted(queries):
        r = results[(BASE, qid)]
        if r["fused"] != ranks[(qid, "fused")]:
            mism.append((qid, "fused"))
        if r["cards"] != colab_cards.get(qid, []):
            mism.append((qid, "top5"))
    repro = {"queries": len(queries), "mismatches": len(mism), "detail": mism[:20], "ok": not mism}
    print(f"재현 확인(1:1): 질의 {len(queries)}개, 불일치 {len(mism)}건")

    weights = [label(a, b) for a, b in GRID if label(a, b) not in errors]
    others = [w for w in weights if w != BASE]
    same_all = bool(others) and all(results[(w, q)]["cards"] == results[(BASE, q)]["cards"]
                                    and results[(w, q)]["fused"] == results[(BASE, q)]["fused"]
                                    for w in others for q in queries)
    supported = weights_supported()
    if errors or same_all or not supported:
        print("경고: 가중치 지원이 확인되지 않음 "
              f"(rrf_detail weights 인자 {supported}, 비 1:1 전부 1:1과 동일 {same_all}, 오류 {sorted(errors)}). "
              "Pipeline·RRF 가중치 코드 반영 후 다시 실행해야 비교 가능")

    t = run_dir / "tables"
    t5 = pd.DataFrame(top5)
    t5.to_csv(t / "top5_by_weight.csv", **CSV)
    chg = []
    for w in weights:
        for qid in sorted(queries):
            x, y = results[(w, qid)]["cards"], results[(BASE, qid)]["cards"]
            chg.append({"weight": w, "query_id": qid, "common_with_1to1": len(set(x) & set(y)), "same_order": x == y})
    chg = pd.DataFrame(chg)
    chg.to_csv(t / "change_vs_1to1.csv", **CSV)
    wc = chg.groupby("weight", sort=False).agg(queries=("query_id", "count"), mean_common=("common_with_1to1", "mean"),
                                               identical=("same_order", "sum")).reset_index()
    wc["changed"] = wc["queries"] - wc["identical"]
    wc = wc.rename(columns={"identical": "identical_count", "changed": "changed_count"})
    wc.to_csv(t / "weight_changes.csv", **CSV)
    print(wc.to_string(index=False))

    # 검수 표본: Top-5가 가중치 사이에서 다른 질의 중 영역 고정 배분(고정 seed) 또는 지정 질의
    diff_q = [q for q in sorted(queries) if len({tuple(results[(w, q)]["cards"]) for w in weights}) > 1]
    if args.pool_queries:   # 지정 질의만 검수 후보(예: 노동 추가 질의). 영역 배분·변동 여부와 무관하게 모두 포함
        want = [f"dev:{x}" if not x.startswith("dev:") else x for x in args.pool_queries.split(",")]
        missing_q = [q for q in want if q not in queries]
        if missing_q:
            raise SystemExit(f"--pool-queries가 Colab 결과에 없음: {missing_q}")
        sample = pd.DataFrame({"query_id": want})
        sample["main_domain"] = [main_domain(info[q]["needs"]) for q in want]
        sample["stratum"] = sample.main_domain
        sample["changed_across_weights"] = [q in diff_q for q in want]
    else:
        keep = set(pd.read_csv(args.keep, dtype=str).query_id) if args.keep else set()
        sample = domain_sample(diff_q, info, SAMPLE_QUOTA, SAMPLE_SEED, keep)
    sample.to_csv(t / "review_sample.csv", **CSV)
    # 검수표에는 가중치 정보 없음(눈가림). 질의 안에서 servId 순 정렬, 가중치 대응은 review_key.csv에 따로 둠
    pool_rows, key_rows, pid = [], [], 0
    for qid in sample.query_id:
        for sid in sorted(set(t5[t5.query_id == qid].servId)):
            pid += 1
            svc = p.services.loc[sid]
            in_w = [w for w in weights if sid in results[(w, qid)]["cards"]]
            pool_rows.append({"pool_id": f"P{pid:04d}", "query_id": qid, "user_text": info[qid]["text"],
                              "needs": "|".join(f"{n['domain']}:{n.get('level', 'main')}" for n in info[qid]["needs"]),
                              "access": info[qid]["access"], "servId": sid, "servNm": svc["servNm"],
                              **{c: svc.get(c, "") for c in POOL_TEXT},
                              "ai_relevance_draft": "", "ai_reason": "",
                              "relevance_0_1_2": "", "reason": "", "reviewer": ""})
            key_rows.append({"pool_id": f"P{pid:04d}", "query_id": qid, "servId": sid, "in_weights": "|".join(in_w)})
    pd.DataFrame(pool_rows).to_csv(t / "review_pool.csv", **CSV)
    write_review_xlsx(pd.DataFrame(pool_rows), t / "review_pool.xlsx")
    pd.DataFrame(key_rows).to_csv(t / "review_key.csv", **CSV)
    print(f"검수 표본: 질의 {len(sample)}개(변동 질의 {len(diff_q)}개 중), 서비스 쌍 {len(pool_rows)}개")
    print(sample.groupby("stratum").size().to_string())

    inputs = {f"colab/{x}": sha(colab / "tables" / x) for x in ("rankings.csv", "queries.csv", "top5_cards.csv")}
    inputs.update({str(x.relative_to(ROOT)): sha(x) for x in (CONFIG, Q1, Q2, CORPUS)})
    meta = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "colab_dir": str(colab),
            "colab_metadata": json.loads((colab / "runs" / "metadata.json").read_text(encoding="utf-8")),
            "input_sha256": inputs, "weights": weights, "weights_not_run": errors, "as_of": args.as_of,
            "profile": PROFILE, "retriever": "Colab 저장 순위 재생(BGE-M3 core, MockLLM 설명문)",
            "counts": {"dev_queries": len(queries), "changed_queries": len(diff_q), "review_pool_queries": len(sample),
                       "review_pool_pairs": len(pool_rows)},
            "sample": {"quota": SAMPLE_QUOTA, "seed": SAMPLE_SEED, "basis": "주 욕구 영역(첫 main) 고정 배분",
                       "kept_from": str(args.keep) if args.keep else None},
            "weights_supported_in_code": supported, "nonbase_identical_to_base_for_all": same_all,
            "reproduction": repro, "decision_rule": DECISION_RULE, "retrieval": p.cfg["retrieval"],
            "note": "개발 질의 기반 잠정 비교. 관련성은 score 단계에서 사람 검수로만 판정"}
    (run_dir / "runs" / "metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2, default=str),
                                                    encoding="utf-8")
    print(f"출력: {run_dir}")
    return 0 if repro["ok"] else 1


def write_review_xlsx(df, path):
    """검수용 엑셀. relevance_0_1_2는 0·1·2 드롭다운, ID는 텍스트 형식(지수 표기·수식 변환 방지)."""
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.worksheet.datavalidation import DataValidation
    wb = Workbook()
    ws = wb.active
    ws.title = "review"
    cols = list(df.columns)
    ws.append(cols)
    for row in df.itertuples(index=False):
        ws.append([str(v) for v in row])
    for cell in ws[1]:
        cell.font = Font(bold=True)
    for r in ws.iter_rows(min_row=2):
        for c in r:
            c.data_type = "s"          # 문자열 고정('-'·'=' 시작 문장도 수식 아님)
            c.number_format = "@"
            c.alignment = Alignment(wrap_text=True, vertical="top")
    n = ws.max_row
    rel = cols.index("relevance_0_1_2") + 1
    letter = ws.cell(row=1, column=rel).column_letter
    dv = DataValidation(type="list", formula1='"0,1,2"', allow_blank=True,
                        error="0, 1, 2 중 하나만 입력", errorTitle="판정값")
    ws.add_data_validation(dv)
    dv.add(f"{letter}2:{letter}{n}")
    fill = PatternFill("solid", fgColor="FFF2CC")
    for name in ("relevance_0_1_2", "reason", "reviewer"):
        col = ws.cell(row=1, column=cols.index(name) + 1).column_letter
        for i in range(1, n + 1):
            ws[f"{col}{i}"].fill = fill
    widths = {"user_text": 50, "servNm": 28, "summary_text": 40, "benefit_text": 50, "target_text": 40,
              "apply_text": 30, "reason": 30, "ai_reason": 40}
    for i, name in enumerate(cols, 1):
        ws.column_dimensions[ws.cell(row=1, column=i).column_letter].width = widths.get(name, 14)
    ws.freeze_panes = "A2"
    wb.save(path)


def main_domain(needs):
    main = [n["domain"] for n in needs if n.get("level", "main") == "main"]
    return main[0] if main else needs[0]["domain"]


def domain_sample(qids, info, quota, seed, keep=()):
    """주 욕구 영역별 고정 배분. keep(이전 표본)에 있는 질의를 먼저 쓰고 부족분만 고정 seed로 채움."""
    df = pd.DataFrame({"query_id": qids})
    df["main_domain"] = [main_domain(info[q]["needs"]) for q in qids]
    df["access"] = [info[q]["access"] != "[]" for q in qids]
    df["compound"] = [len(info[q]["needs"]) > 1 for q in qids]
    df["kept"] = df.query_id.isin(keep)
    picked, short = [], {}
    for dom, n in quota.items():
        g = df[df.main_domain == dom]
        if len(g) < n:
            short[dom] = (n, len(g))
        kept = g[g.kept]
        kept = kept.sample(n=min(n, len(kept)), random_state=seed) if len(kept) else kept
        rest = g[~g.query_id.isin(kept.query_id)]
        add = rest.sample(n=min(n - len(kept), len(rest)), random_state=seed) if len(rest) else rest
        picked += [kept, add]
    if short:
        print("배분 부족(영역: 필요, 가능):", short)
    out = pd.concat(picked).sort_values(["main_domain", "query_id"]).reset_index(drop=True)
    out["stratum"] = out.main_domain
    return out


# ── score ──────────────────────────────────────────────────────
def ndcg5(rels, pool_rels):
    gain = lambda r: 2 ** r - 1   # noqa: E731
    dcg = sum(gain(r) / math.log2(i + 2) for i, r in enumerate(rels[:5]))
    ideal = sum(gain(r) / math.log2(i + 2) for i, r in enumerate(sorted(pool_rels, reverse=True)[:5]))
    return None if ideal == 0 else dcg / ideal


def score_cmd(args):
    run = args.run
    t = run / "tables"
    pool = pd.read_csv(t / "review_pool.csv", dtype=str, keep_default_na=False, encoding="utf-8-sig")
    rev = pd.read_csv(args.review, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    errs = []
    for col in ("pool_id", "query_id", "servId", "relevance_0_1_2"):
        if col not in rev.columns:
            errs.append(f"열 없음: {col}")
    if errs:
        print("\n".join(errs))
        return 1
    if rev.duplicated("pool_id").any() or rev.duplicated(["query_id", "servId"]).any():
        errs.append("pool_id 또는 (query_id, servId) 중복")
    exp = {r.pool_id: (r.query_id, r.servId) for r in pool.itertuples()}
    got = {r.pool_id: (r.query_id, r.servId) for r in rev.itertuples()}
    if exp != got:
        errs.append(f"검수 파일 키 불일치: 누락 {len(set(exp) - set(got))}, 초과 {len(set(got) - set(exp))}, "
                    f"내용 다름 {sum(exp[k] != got[k] for k in set(exp) & set(got))}")
    bad = rev[~rev["relevance_0_1_2"].str.strip().isin(["0", "1", "2"])]
    errs += [f"{r.pool_id} {r.query_id} {r.servId}: relevance={r.relevance_0_1_2!r}" for r in bad.itertuples()]
    if errs:
        print(f"검수 파일 오류 {len(errs)}건")
        print("\n".join(errs[:50]))
        return 1
    rel = {(r.query_id, r.servId): int(r.relevance_0_1_2) for r in rev.itertuples()}
    top5 = pd.read_csv(t / "top5_by_weight.csv", dtype={"servId": str}, encoding="utf-8-sig")
    weights = list(dict.fromkeys(top5.weight))
    rows, undefined = [], set()
    for qid in sorted(set(pool.query_id)):
        pool_rels = [v for (q, _), v in rel.items() if q == qid]
        for w in weights:
            sub = top5[(top5.weight == w) & (top5.query_id == qid)].sort_values("rank")
            rs = [rel[(qid, s)] for s in sub.servId]
            nd = ndcg5(rs, pool_rels)
            if nd is None:
                undefined.add(qid)
            rows.append({"weight": w, "query_id": qid, "ndcg5": nd, "p5": sum(r == 2 for r in rs) / 5,
                         "zero": int(not any(rs)), "n_top5": len(rs)})
    pq = pd.DataFrame(rows)
    pq.to_csv(t / "weight_scores_per_query.csv", **CSV)
    ok = pq[pq.ndcg5.notna()]
    base = ok[ok.weight == BASE].set_index("query_id").ndcg5
    out = []
    for w in weights:
        s = ok[ok.weight == w].set_index("query_id")
        d = (s.ndcg5 - base.reindex(s.index)).dropna()
        out.append({"weight": w, "queries": len(s), "mean_ndcg5": s.ndcg5.mean(), "mean_p5": s.p5.mean(),
                    "zero_ratio": s.zero.mean(), "wins_vs_1to1": int((d > TOL).sum()),
                    "losses_vs_1to1": int((d < -TOL).sum()), "ties_vs_1to1": int((d.abs() <= TOL).sum()),
                    "undefined_excluded": len(undefined)})
    ws = pd.DataFrame(out)
    ws.to_csv(t / "weight_scores.csv", **CSV)
    print(ws.to_string(index=False))
    print(f"nDCG 정의 불가(이상적 0) 질의 {len(undefined)}개 제외")
    b = ws[ws.weight == BASE].iloc[0]
    cand = ws[(ws.weight != BASE) & (ws.mean_ndcg5 - b.mean_ndcg5 >= MIN_GAIN - TOL)
              & (ws.wins_vs_1to1 > ws.losses_vs_1to1)]
    if len(cand):
        best = cand.sort_values("mean_ndcg5", ascending=False).iloc[0]
        print(f"권고: {best.weight}로 변경 검토 (nDCG@5 +{best.mean_ndcg5 - b.mean_ndcg5:.3f}, "
              f"승 {best.wins_vs_1to1} > 패 {best.losses_vs_1to1}). 개발 질의 기반 잠정 판단")
    else:
        print("권고: 1:1 유지 (기준 미충족). 개발 질의 기반 잠정 판단")
    print("판정 기준:", DECISION_RULE)
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="가중치별 Top-5 재계산과 검수 풀 생성")
    r.add_argument("--colab", type=Path, required=True, help="Colab 결과 폴더(tables/, runs/)")
    r.add_argument("--as-of", default=DEFAULT_AS_OF, help=f"모집기간 기준일(기본 {DEFAULT_AS_OF}, Colab 실행일)")
    r.add_argument("--out", type=Path)
    r.add_argument("--keep", type=Path, help="이전 review_sample.csv. 영역 배분 안에서 이 질의를 우선 유지")
    r.add_argument("--extra-corpus", type=Path, help="별도 평가 질의 JSONL(Colab --extra-corpus와 같은 파일)")
    r.add_argument("--pool-queries", help="검수 후보를 만들 질의 id(쉼표 구분). 주면 영역 배분 대신 이 질의만")
    s = sub.add_parser("score", help="사람 검수 결과로 가중치별 점수 계산")
    s.add_argument("--run", type=Path, required=True)
    s.add_argument("--review", type=Path, required=True)
    args = ap.parse_args(argv)
    return run_cmd(args) if args.cmd == "run" else score_cmd(args)


if __name__ == "__main__":
    sys.exit(main())
