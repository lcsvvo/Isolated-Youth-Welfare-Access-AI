"""Dense 운영 구성(BGE-M3 core) 실제 모델 검증. baseline·core 캐시 생성·재사용·전환과 2단계 검색 연동 확인.

저장소 루트에서 (GPU 권장, Colab T4 기준):
  python ai/scripts/dense_ops_check.py --device cuda
  python ai/scripts/dense_ops_check.py --device cpu --profiles core   # 느림
  python ai/scripts/dense_ops_check.py --mock                         # 연결만 (MockEmbedder, 실제 검증 아님)

필요 파일: data/processed/services_master.csv(5,288건), outputs/03_q1_recommendation/q1_review.json,
          outputs/04_q2_recommendation/q2_info.json (config 기본 경로)
출력: outputs/10_dense_ops/<run>/tables/*.csv, runs/metadata.json (Git에 올리지 않음)

확인 내용
- 구성별 캐시(config paths.dense_cache의 {profile}) 생성 → 재사용 → 구성 전환 시 자기 캐시만 읽음
- 서비스 수·ID 중복·임베딩 개수·차원·유한값·단위 길이, 저장 지문 = 색인 지문
- 질의: 인계 4개(파일 문장 그대로) + 개발 말뭉치 사람 label 세션의 현재 build_queries 결과(#75 반영)
- 같은 질의·같은 DB에서 baseline·core Dense Top-100, BM25+Dense RRF, 규칙·Q1 우대, Top-5 카드 비교
- 검색 없음·한쪽 실패·전체 실패 처리
- 설명문 LLM은 기본 MockLLM(카드 생성 경로 확인용). --real-llm이면 config의 Kanana 사용
Q3 독립 평가 카드는 읽지 않음. 순위 차이 기술이며 관련성 정확도 평가가 아님.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import yaml

AI = Path(__file__).resolve().parents[1]
ROOT = AI.parent
sys.path.insert(0, str(AI))

import pipeline as P  # noqa: E402
from retrieve.dense_texts import PROFILES, build_texts  # noqa: E402
from rules.filter import _domain_rank  # noqa: E402

CONFIG = AI / "configs" / "config.yaml"
HANDOFF = AI / "data" / "dense_handoff_queries.jsonl"
CORPUS = AI / "data" / "dev_corpus.jsonl"
EVAL_REVISION = "5617a9f61b028005a4858fdac845db406aefb181"   # PR #71 사람 검수 실행의 BGE-M3 커밋
EVAL_MASTER_SHA = "269005564d843cb53988422465e1ba56824341d8a26c3d1b655908ca6ef83d6e"
EXPECTED_SERVICES = 5288
CSV = {"index": False, "encoding": "utf-8-sig"}


def sha(path) -> str | None:
    path = Path(path)
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def read_jsonl(path):
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8-sig").splitlines() if l.strip()]


# ── 모델 ───────────────────────────────────────────────────────
def load_embedder(args, cfg):
    if args.mock:
        from core.mock import MockEmbedder

        class Keyed(MockEmbedder):
            cache_key = "mock-embedder-v1"
        return Keyed(), {"model": "mock", "real_model": False}
    import os
    from core.models import Embedder
    cache_dir = os.environ.get("MODEL_CACHE_DIR") or cfg["paths"].get("cache_dir")
    t0 = perf_counter()
    emb = Embedder(cfg["models"]["embedder"], cache_dir, device=args.device, batch_size=args.batch_size,
                   max_length=args.max_length, revision=args.revision)
    config = getattr(getattr(emb.model[0], "auto_model", None), "config", None)
    commit = getattr(config, "_commit_hash", None)
    return emb, {"model": cfg["models"]["embedder"], "real_model": True, "commit": commit,
                 "matches_pr71_revision": commit == EVAL_REVISION, "requested_revision": args.revision,
                 "cache_key": emb.cache_key, "max_seq_length": emb.model.max_seq_length,
                 "device": str(emb.model.device), "load_seconds": perf_counter() - t0}


def load_llm(args, cfg):
    if args.real_llm and not args.mock:
        import os
        from core.models import LLM
        return LLM(cfg["models"]["llm"], os.environ.get("MODEL_CACHE_DIR") or cfg["paths"].get("cache_dir"))
    from core.mock import MockLLM
    return MockLLM()


def build_pipeline(profile, embedder, llm, run_dir, as_of, dense_cache=None):
    """config 복사본에서 dense_profile만 바꿔 Pipeline 생성. 상대 경로는 Pipeline이 ai/ 기준으로 해석.
    dense_cache를 주면 그 경로 템플릿 사용(mock 실행이 실제 캐시를 만들지 않게 함)."""
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    cfg["retrieval"]["dense_profile"] = profile
    if dense_cache:
        cfg["paths"]["dense_cache"] = str(dense_cache)
    path = run_dir / "runs" / f"config_{profile}.yaml"
    path.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
    original = P.load_models
    P.load_models = lambda cfg, mock: (embedder, llm)
    try:
        q1 = ROOT / "outputs/03_q1_recommendation/q1_review.json"
        q2 = ROOT / "outputs/04_q2_recommendation/q2_info.json"
        t0 = perf_counter()
        pipe = P.Pipeline(config_path=path, q1_review_path=q1 if q1.exists() else None,
                          q2_info_path=q2 if q2.exists() else None, as_of=as_of)
        return pipe, perf_counter() - t0
    finally:
        P.load_models = original


# ── 색인 점검 ──────────────────────────────────────────────────
def index_checks(profile, pipe, seconds):
    d, ids = pipe.dense, list(pipe.dense.ids)
    norms = np.linalg.norm(d.emb, axis=1)
    stored = None
    if d.cache_path and d.cache_path.exists():
        with np.load(d.cache_path, allow_pickle=False) as cached:
            stored = cached["fingerprint"].item()
    q1_col = pipe.services.get("q1_core_domains")
    return {"profile": profile, "fields": "|".join(PROFILES[profile]), "services": len(pipe.services),
            "expected_services": EXPECTED_SERVICES, "unique_ids": len(set(ids)), "ids_equal_bm25": ids == list(pipe.bm25.ids),
            "ids_equal_master_order": ids == list(pipe.services.index),
            "empty_documents": int(sum(not t.strip() for t in build_texts(pipe.services, profile))),
            "emb_rows": int(d.emb.shape[0]), "emb_dim": int(d.emb.shape[1]), "finite": bool(np.isfinite(d.emb).all()),
            "norm_min": float(norms.min()), "norm_max": float(norms.max()),
            "cache_path": str(d.cache_path), "cache_hit": d.cache_hit, "fingerprint": d.fingerprint,
            "stored_fingerprint_matches": stored == d.fingerprint, "cache_sha256": sha(d.cache_path) if d.cache_path else None,
            "q1_attached": q1_col is not None and bool(q1_col.notna().all()) if q1_col is not None else False,
            "pipeline_seconds": seconds}


# ── 질의 ───────────────────────────────────────────────────────
def handoff_queries():
    return [{"query_id": f"handoff:{r['id']}", "set": "handoff", "bm25": r.get("bm25") or r["dense"],
             "dense": r["dense"], "session": None, "needs": "", "access": "", "access_mode": ""} for r in read_jsonl(HANDOFF)]


def stage2_queries(pipe, extra=None):
    """개발 말뭉치 사람 label 세션 → 현재 build_queries (#75 접근 제약·부정 처리 반영).
    extra: 같은 형식의 별도 평가 질의 파일(few-shot 풀과 분리). id가 말뭉치와 겹치면 중단."""
    from eval.stage2_query_review import dev_cases
    from query.build import build_queries
    rows, skipped = [], []
    records = read_jsonl(CORPUS)
    if extra:
        added = read_jsonl(extra)
        clash = {r["id"] for r in added} & {r["id"] for r in records}
        if clash:
            raise ValueError(f"추가 질의 id가 개발 말뭉치와 겹침: {sorted(clash)}")
        records += added
    for case in dev_cases(records):
        try:
            q = build_queries(case["session"], pipe.terms, pipe.cfg)
        except ValueError as e:
            skipped.append({"query_id": case["case_id"], "reason": str(e)})
            continue
        needs = case["session"]["confirmed_profile"]["needs"]
        rows.append({"query_id": f"dev:{case['case_id']}", "set": "stage2_dev", "bm25": q["bm25"], "dense": q["dense"],
                     "session": case["session"], "text": case["text"],
                     "needs": "|".join(f"{n['domain']}:{n.get('level', 'main')}" for n in needs),
                     "n_needs": len(needs),
                     "access": json.dumps(q["access"], ensure_ascii=False), "n_access": len(q["access"]),
                     "access_mode": q["access_mode"]})
    return rows, skipped


def run_queries(profile, pipe, queries, top_n):
    qrows, cards, ranks = [], [], []
    for q in queries:
        t0 = perf_counter()
        if q["session"] is not None and q["session"].get("status") == "ready":
            out = pipe.search(copy.deepcopy(q["session"]))
            found, result_cards = out["ranks"], out["cards"]
            retrieval_status, n_excluded = out["retrieval_status"], len(out.get("excluded", []))
        else:   # 인계 질의: 세션 없음 → 검색만
            found, result_cards = pipe.search_queries(q["bm25"], q["dense"]), []
            retrieval_status, n_excluded = found["status"], None
        ms = (perf_counter() - t0) * 1000
        needs = (q["session"] or {}).get("confirmed_profile", {}).get("needs", [])
        qrows.append({"profile": profile, "query_id": q["query_id"], "set": q["set"], "needs": q["needs"],
                      "access": q["access"], "access_mode": q["access_mode"], "bm25_query": q["bm25"],
                      "dense_query": q["dense"], "status": retrieval_status,
                      "errors": json.dumps(found.get("errors") or [], ensure_ascii=False),
                      "n_dense": len(found["dense"]), "n_bm25": len(found["bm25"]), "n_rrf": len(found["fused"]),
                      "n_cards": len(result_cards), "n_excluded": n_excluded, "ms": ms})
        for method in ("dense", "bm25", "fused"):
            for rank, sid in enumerate(found[method][:top_n], 1):
                ranks.append({"profile": profile, "query_id": q["query_id"], "method": method, "rank": rank, "servId": sid})
        for rank, c in enumerate(result_cards, 1):
            svc = pipe.services.loc[c["servId"]].to_dict()
            cards.append({"profile": profile, "query_id": q["query_id"], "rank": rank, "servId": c["servId"],
                          "servNm": c["servNm"], "q1_main_preference": _domain_rank(svc, needs)[1],
                          "checks": json.dumps(c["checks"], ensure_ascii=False),
                          "has_q1_access_review": "q1_access_review" in c,
                          "q2_fields": sum(k.startswith("q2") for k in c), "fallback": c.get("fallback")})
    return qrows, cards, ranks


def compare_profiles(ranks, cards, a="baseline", b="core"):
    r, c = pd.DataFrame(ranks), pd.DataFrame(cards)
    rows = []
    for qid in sorted(set(r["query_id"])):
        row = {"query_id": qid}
        for method, ns in (("dense", (5, 10, 100)), ("fused", (5, 10))):
            for n in ns:
                sa = set(r[(r.query_id == qid) & (r.profile == a) & (r.method == method) & (r["rank"] <= n)].servId)
                sb = set(r[(r.query_id == qid) & (r.profile == b) & (r.method == method) & (r["rank"] <= n)].servId)
                row[f"{method}_common_top{n}"] = len(sa & sb)
        if len(c):
            ca = c[(c.query_id == qid) & (c.profile == a)].sort_values("rank").servId.tolist()
            cb = c[(c.query_id == qid) & (c.profile == b)].sort_values("rank").servId.tolist()
            row.update({"cards_common": len(set(ca) & set(cb)), f"cards_{a}": "|".join(ca), f"cards_{b}": "|".join(cb)})
        rows.append(row)
    return pd.DataFrame(rows)


# ── 예외 처리 ──────────────────────────────────────────────────
class Broken:
    def __init__(self, ids):
        self.ids = ids

    def search(self, query, k=100):
        raise RuntimeError("의도한 검색 실패(검증용)")


def exception_checks(pipe, sample):
    rows = []
    found = pipe.search_queries("", "")
    rows.append({"case": "빈 질의", "status": found["status"], "n_fused": len(found["fused"]), "expected": "empty"})
    real = {"bm25": pipe.bm25, "dense": pipe.dense}
    try:
        for name, broken in (("dense 실패", "dense"), ("bm25 실패", "bm25")):
            setattr(pipe, broken, Broken(real[broken].ids))
            found = pipe.search_queries(sample["bm25"], sample["dense"])
            rows.append({"case": name, "status": found["status"], "n_fused": len(found["fused"]), "expected": "partial"})
            setattr(pipe, broken, real[broken])
        pipe.bm25, pipe.dense = Broken(real["bm25"].ids), Broken(real["dense"].ids)
        found = pipe.search_queries(sample["bm25"], sample["dense"])
        rows.append({"case": "전체 실패", "status": found["status"], "n_fused": len(found["fused"]), "expected": "error"})
        if sample.get("session"):
            out = pipe.search(copy.deepcopy(sample["session"]))
            rows.append({"case": "전체 실패 카드", "status": out["retrieval_status"], "n_fused": len(out["cards"]),
                         "expected": "error"})
    finally:
        pipe.bm25, pipe.dense = real["bm25"], real["dense"]
    for r in rows:
        r["ok"] = r["status"] == r["expected"] and (r["expected"] != "error" or r["n_fused"] == 0)
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profiles", nargs="+", choices=list(PROFILES), default=["baseline", "core"])
    ap.add_argument("--device", choices=["cpu", "cuda", "mps"])
    ap.add_argument("--batch-size", type=int, default=1, help="PR #71 실행과 같은 1. 최대 길이 8192에서 크게 잡으면 T4 메모리 부족")
    ap.add_argument("--max-length", type=int, help="기본: 모델 값(BGE-M3 8192, PR #71과 같음)")
    ap.add_argument("--revision", help="재현용 Hugging Face 커밋 (기본: 최신. PR #71은 5617a9f)")
    ap.add_argument("--mock", action="store_true", help="MockEmbedder 연결 확인. 실제 모델 검증 아님")
    ap.add_argument("--real-llm", action="store_true", help="설명문에 config의 Kanana 사용 (기본 MockLLM)")
    ap.add_argument("--as-of", help="모집기간 기준일 YYYY-MM-DD (기본 오늘)")
    ap.add_argument("--top-n", type=int, default=100)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--extra-corpus", type=Path, help="별도 평가 질의 JSONL(dev_corpus 형식). 개발 질의에 추가")
    args = ap.parse_args(argv)
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    tag = datetime.now().strftime("%Y%m%d_%H%M") + ("_mock" if args.mock else "")
    run_dir = args.out or ROOT / "outputs" / "10_dense_ops" / tag
    (run_dir / "tables").mkdir(parents=True, exist_ok=True)
    (run_dir / "runs").mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "runs" / "progress.log"

    def log(message):
        line = f"{datetime.now().isoformat(timespec='seconds')} {message}"
        print(line, flush=True)
        with log_path.open("a", encoding="utf-8") as stream:
            print(line, file=stream)

    try:
        return _run(args, cfg, run_dir, log)
    except BaseException:
        import traceback
        (run_dir / "runs" / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        log("실패: runs/error.txt 참고")
        raise


def _gpu_memory():
    try:
        import torch
        if torch.cuda.is_available():
            return {"allocated_gb": round(torch.cuda.memory_allocated() / 2**30, 2),
                    "max_allocated_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2)}
    except ImportError:
        pass
    return None


def _run(args, cfg, run_dir, log):
    log(f"시작 profiles={args.profiles} device={args.device} batch_size={args.batch_size} mock={args.mock}")
    embedder, model_meta = load_embedder(args, cfg)
    log(f"모델 로드 {json.dumps(model_meta, ensure_ascii=False)} gpu={_gpu_memory()}")
    llm = load_llm(args, cfg)
    # mock은 실제 캐시를 건드리지 않도록 실행 폴더에 캐시
    mock_cache = (run_dir / "cache" / "service_emb_{profile}.npz").resolve() if args.mock else None
    checks, queries_all, cards_all, ranks_all, exc_all, skipped = [], [], [], [], [], []
    caches = {}
    for phase in ("build", "reuse"):
        for profile in args.profiles:
            log(f"색인 {phase} {profile} 시작")
            pipe, seconds = build_pipeline(profile, embedder, llm, run_dir, args.as_of, mock_cache)
            row = {"phase": phase, **index_checks(profile, pipe, seconds)}
            checks.append(row)
            pd.DataFrame(checks).to_csv(run_dir / "tables" / "index_checks.csv", **CSV)   # 중간 저장
            log("색인 " + json.dumps({k: row[k] for k in ("phase", "profile", "services", "emb_rows", "emb_dim",
                                                         "cache_hit", "stored_fingerprint_matches",
                                                         "pipeline_seconds")}, ensure_ascii=False)
                + f" gpu={_gpu_memory()}")
            caches.setdefault(profile, []).append(row["cache_sha256"])
            if phase == "reuse":
                continue
            queries = handoff_queries()
            dev, skipped = stage2_queries(pipe, args.extra_corpus)
            queries += dev
            log(f"검색 {profile} 질의 {len(queries)}개")
            qrows, cards, ranks = run_queries(profile, pipe, queries, args.top_n)
            queries_all += qrows
            cards_all += cards
            ranks_all += ranks
            sample = next((q for q in dev if q["session"] and q["session"].get("status") == "ready"), queries[0])
            exc_all += [{"profile": profile, **r} for r in exception_checks(pipe, sample)]
            del pipe
    t = run_dir / "tables"
    cdf, qdf = pd.DataFrame(checks), pd.DataFrame(queries_all)
    cdf.to_csv(t / "index_checks.csv", **CSV)
    qdf.to_csv(t / "queries.csv", **CSV)
    pd.DataFrame(cards_all).to_csv(t / "top5_cards.csv", **CSV)
    pd.DataFrame(ranks_all).to_csv(t / "rankings.csv", **CSV)
    pd.DataFrame(exc_all).to_csv(t / "exceptions.csv", **CSV)
    pd.DataFrame(skipped).to_csv(t / "skipped_queries.csv", **CSV)
    cmp = compare_profiles(ranks_all, cards_all) if {"baseline", "core"} <= set(args.profiles) else pd.DataFrame()
    cmp.to_csv(t / "profile_compare.csv", **CSV)

    build, reuse = cdf[cdf.phase == "build"], cdf[cdf.phase == "reuse"]
    master = (AI / cfg["paths"]["services"]).resolve()
    summary = {
        "real_model": model_meta["real_model"],
        "services_ok": bool((cdf.services == EXPECTED_SERVICES).all() and (cdf.unique_ids == EXPECTED_SERVICES).all()),
        "ids_ok": bool(cdf.ids_equal_bm25.all() and cdf.ids_equal_master_order.all()),
        "emb_ok": bool((cdf.emb_rows == cdf.services).all() and cdf.finite.all()
                       and (cdf.norm_min > 0.999).all() and (cdf.norm_max < 1.001).all()),
        "emb_dim": sorted(set(cdf.emb_dim.astype(int))),
        "fingerprint_ok": bool(cdf.stored_fingerprint_matches.all()),
        "reuse_cache_hit": bool(reuse.cache_hit.all()),
        "cache_files_distinct": len({r for r in build.cache_path}) == len(build),
        "cache_unchanged_on_reuse_and_switch": all(len(set(v)) == 1 for v in caches.values()),
        "q1_attached": bool(cdf.q1_attached.all()),
        "status_counts": qdf.groupby("profile").status.value_counts().unstack(fill_value=0).to_dict("index"),
        "dense_top100_full": bool((qdf.n_dense == args.top_n).all()),
        "stage2_cards_5": bool((qdf[qdf.set == "stage2_dev"].n_cards == 5).all()),
        "stage2_cards_dist": qdf[qdf.set == "stage2_dev"].groupby("profile").n_cards.value_counts().unstack(fill_value=0).to_dict("index"),
        "exceptions_ok": bool(pd.DataFrame(exc_all).ok.all()),
        "skipped_queries": len(skipped),
    }
    if len(cmp):
        summary["baseline_core_mean_common"] = {c: float(cmp[c].mean()) for c in cmp if c.endswith(tuple("0123456789")) or c == "cards_common"}
    meta = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "model": model_meta, "llm_for_explain":
            "kanana" if args.real_llm and not args.mock else "MockLLM", "profiles": args.profiles,
            "fields": {p: list(PROFILES[p]) for p in args.profiles},
            "master_sha256": sha(master), "master_matches_pr71": sha(master) == EVAL_MASTER_SHA,
            "input_sha256": {str(p.relative_to(ROOT)): sha(p) for p in
                             [CONFIG, HANDOFF, CORPUS, ROOT / "outputs/03_q1_recommendation/q1_review.json",
                              ROOT / "outputs/04_q2_recommendation/q2_info.json"]},
            "retrieval": cfg["retrieval"], "as_of": args.as_of, "summary": summary,
            "python": sys.version, "platform": platform.platform(),
            "note": "baseline·core 순위 차이와 연동 확인. 관련성 정확도·최종 추천 품질 평가 아님. Q3 카드 미사용"}
    (run_dir / "runs" / "metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2, default=str),
                                                    encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    log(f"완료 출력: {run_dir}")
    ok = all(summary[k] for k in ("services_ok", "ids_ok", "emb_ok", "fingerprint_ok", "reuse_cache_hit",
                                   "cache_files_distinct", "cache_unchanged_on_reuse_and_switch", "exceptions_ok"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
