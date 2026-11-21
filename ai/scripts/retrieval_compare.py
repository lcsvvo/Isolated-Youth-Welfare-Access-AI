"""BM25 단독 · Dense 단독 · BM25+Dense RRF를 같은 질의로 실행해 순위를 저장한다 (LLM 불필요).

저장소 루트에서:
  # 실제 2단계 쿼리(개발 말뭉치 사람 label → build_queries) + BGE-M3 (config 기본 모델)
  python ai/scripts/retrieval_compare.py --query-set stage2_dev --dense-model bge-m3
  # 인계 질의(dense만 있음 → BM25에도 같은 문장 사용, query_source 열에 기록)
  python ai/scripts/retrieval_compare.py --query-set handoff --dense-model kure-v1
  # 직접 만든 JSONL (id, dense 필수, bm25 선택)
  python ai/scripts/retrieval_compare.py --queries my.jsonl --dense-model bge-m3
  # 모델 없이 연결만 확인 (MockEmbedder, 순위 품질 의미 없음)
  python ai/scripts/retrieval_compare.py --query-set stage2_dev --dense-model mock

출력: outputs/09_retrieval_compare/  (run_tag = <query_set>__<dense_model>[__<profile>])
  tables/<run_tag>__rankings.csv   질의·방법(bm25|dense|rrf)별 Top-N 순위, 서비스명, BM25·Dense 순위, RRF 점수
  tables/<run_tag>__queries.csv    질의별 쿼리 문자열, 상태(ok|empty|partial|error), 오류, 검색 시간(ms)
  tables/<run_tag>__overlap.csv    질의별 Top-10·Top-100 방법 간 공통 서비스 수, RRF Top-10 출처
  runs/<run_tag>__metadata.json    모델·리비전·문서 구성·파라미터·입력 해시·실행 시간

해석 범위
- 관련성 정답셋 전. 공통 서비스 수·순위는 방법 간 차이 기술이며 정확도·최적 파라미터 근거가 아님
- 결합 로직은 pipeline.search_queries와 같은 retrieve.fusion.search_and_fuse 사용
- Dense 모델을 바꿔 다시 실행해도 이 스크립트·RRF 코드 수정 불필요 (--dense-model만 변경)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import pandas as pd
import yaml

AI = Path(__file__).resolve().parents[1]
ROOT = AI.parent
sys.path.insert(0, str(AI))

from retrieve.bm25 import BM25Index, build_texts as build_bm25_texts  # noqa: E402
from retrieve.dense import DenseIndex  # noqa: E402
from retrieve.dense_texts import PROFILES, build_texts as build_dense_texts  # noqa: E402
from retrieve.fusion import check_params, search_and_fuse  # noqa: E402

MODELS = {"bge-m3": "BAAI/bge-m3", "kure-v1": "nlpai-lab/KURE-v1", "mock": None}
METHODS = ("bm25", "dense", "rrf")
OUT = ROOT / "outputs" / "09_retrieval_compare"
CONFIG = AI / "configs" / "config.yaml"
HANDOFF = AI / "data" / "dense_handoff_queries.jsonl"
CORPUS = AI / "data" / "dev_corpus.jsonl"
MASTER = ROOT / "data" / "processed" / "services_master.csv"
NOTE = "관련성 정답셋 전. 방법 간 순위 차이 기술용이며 검색 정확도·최적 파라미터 근거 아님"


def sha(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def environment() -> dict:
    """실행 환경 기록 (패키지 버전·GPU). 없는 패키지는 None."""
    from importlib.metadata import PackageNotFoundError, version
    pkgs = {}
    for name in ["torch", "sentence-transformers", "transformers", "kiwipiepy", "kiwipiepy_model", "rank-bm25",
                 "numpy", "pandas"]:
        try:
            pkgs[name] = version(name)
        except PackageNotFoundError:
            pkgs[name] = None
    gpu = None
    try:
        import torch
        if torch.cuda.is_available():
            gpu = {"name": torch.cuda.get_device_name(0), "cuda": torch.version.cuda}
    except ImportError:
        pass
    return {"packages": pkgs, "gpu": gpu}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8-sig").splitlines() if l.strip()]


# ── 질의 ───────────────────────────────────────────────────────
def queries_from_jsonl(path: Path) -> list[dict]:
    rows = []
    for r in read_jsonl(path):
        if not isinstance(r.get("id"), str) or not isinstance(r.get("dense"), str):
            raise ValueError(f"{path}: 각 행에 문자열 id, dense 필요: {r}")
        has_bm25 = isinstance(r.get("bm25"), str)
        rows.append({"query_id": r["id"], "bm25": r["bm25"] if has_bm25 else r["dense"], "dense": r["dense"],
                     "query_source": "jsonl" if has_bm25 else "jsonl_dense_text_for_bm25"})
    return rows


def queries_stage2_dev(cfg: dict) -> tuple[list[dict], list[dict]]:
    """개발 말뭉치 사람 label을 확정값으로 둔 세션 → 실제 2단계 build_queries 결과 (Q3 평가 자료 미사용)."""
    from eval.stage2_query_review import dev_cases
    from query.build import build_queries, load_terms
    terms = load_terms(AI / cfg["paths"]["query_terms"])
    rows, skipped = [], []
    for case in dev_cases(read_jsonl(CORPUS)):
        try:
            q = build_queries(case["session"], terms, cfg)
        except ValueError as e:          # explore_only 등 검색하지 않는 세션
            skipped.append({"query_id": case["case_id"], "reason": str(e)})
            continue
        rows.append({"query_id": case["case_id"], "bm25": q["bm25"], "dense": q["dense"],
                     "query_source": f"stage2_build_queries:{q['access_mode']}"})
    return rows, skipped


# ── 색인 ───────────────────────────────────────────────────────
def build_dense(args, ids, texts):
    if args.dense_model == "mock":
        from core.mock import MockEmbedder
        return DenseIndex(MockEmbedder(), ids, texts), {"model": "mock", "model_identity": None}
    from core.models import Embedder
    embedder = Embedder(args.model_path or MODELS[args.dense_model], cache_dir=str(AI / "data/models"),
                        device=args.device, batch_size=args.batch_size, max_length=args.max_length,
                        revision=args.revision, local_files_only=bool(args.model_path))
    # dense_demo.py와 같은 캐시 이름 → 같은 모델·문서면 임베딩 재사용(지문 검증 후)
    name = f"dense_{args.dense_model}.npz" if args.profile == "baseline" else f"dense_{args.dense_model}_{args.profile}.npz"
    index = DenseIndex(embedder, ids, texts, AI / "data" / name)
    return index, {"model": MODELS[args.dense_model], "model_path": args.model_path, "revision": args.revision,
                   "model_identity": embedder.cache_key, "max_length": args.max_length, "batch_size": args.batch_size}


# ── 표 ─────────────────────────────────────────────────────────
def ranking_rows(tag, qid, found, names, top):
    rrf_by = {r["servId"]: r for r in found["rrf"]}
    pos = {m: {sid: i for i, sid in enumerate(found[m], 1)} for m in ("bm25", "dense")}
    out = []
    for method in METHODS:
        ids = found["fused"] if method == "rrf" else found[method]
        for rank, sid in enumerate(ids[:top], 1):
            r = rrf_by.get(sid)
            out.append({"run_tag": tag, "query_id": qid, "method": method, "rank": rank, "servId": sid,
                        "servNm": names.get(sid), "bm25_rank": pos["bm25"].get(sid), "dense_rank": pos["dense"].get(sid),
                        "rrf_rank": r["rrf_rank"] if r else None, "rrf_score": r["rrf_score"] if r else None})
    return out


def overlap_row(tag, qid, found):
    row = {"run_tag": tag, "query_id": qid}
    for n in (10, 100):
        b, d, f = (set(found[m][:n]) for m in ("bm25", "dense", "fused"))
        row.update({f"n_bm25_top{n}": len(b), f"n_dense_top{n}": len(d), f"n_rrf_top{n}": len(f),
                    f"bm25_dense_common_top{n}": len(b & d), f"rrf_bm25_common_top{n}": len(f & b),
                    f"rrf_dense_common_top{n}": len(f & d)})
    src = Counter("both" if r["ranks"]["bm25"] and r["ranks"]["dense"] else "bm25_only" if r["ranks"]["bm25"]
                  else "dense_only" for r in found["rrf"][:10])
    row.update({f"rrf_top10_from_{k}": src.get(k, 0) for k in ("both", "bm25_only", "dense_only")})
    return row


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--query-set", choices=["stage2_dev", "handoff"])
    g.add_argument("--queries", type=Path, help="id, dense(필수), bm25(선택) JSONL")
    ap.add_argument("--dense-model", choices=MODELS, default="bge-m3")
    ap.add_argument("--profile", choices=PROFILES, default=None, help="Dense 문서 구성 (기본: config dense_profile)")
    ap.add_argument("--model-path", help="다운로드한 모델 폴더 (네트워크 없이 사용)")
    ap.add_argument("--revision", help="재현용 Hugging Face 커밋 ID")
    ap.add_argument("--device", choices=["cpu", "cuda", "mps"])
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-length", type=int, default=8192)
    ap.add_argument("--top-k", type=int, default=None, help="검색별 후보 수 (기본: config top_k_each)")
    ap.add_argument("--rrf-k", type=float, default=None, help="기본: config rrf_k")
    ap.add_argument("--rrf-top-n", type=int, default=None, help="기본: config rrf_top_n")
    ap.add_argument("--save-top", type=int, default=100, help="rankings.csv에 저장할 방법별 순위 수")
    ap.add_argument("--master", type=Path, default=MASTER)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--bm25-cache", type=Path, default=None, help="BM25 토큰 캐시 (기본: config paths.bm25_cache)")
    ap.add_argument("--tag", help="run_tag 직접 지정")
    args = ap.parse_args(argv)

    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    r = cfg["retrieval"]
    args.profile = args.profile or r.get("dense_profile", "baseline")
    rrf_k = args.rrf_k if args.rrf_k is not None else r["rrf_k"]
    rrf_k = int(rrf_k) if float(rrf_k).is_integer() else rrf_k
    params = check_params(args.top_k or r["top_k_each"], rrf_k,
                          args.rrf_top_n or r.get("rrf_top_n", 100))
    if args.save_top < 1:
        ap.error("--save-top은 1 이상")

    if args.queries:
        query_set, queries, skipped = args.queries.stem, queries_from_jsonl(args.queries), []
    elif args.query_set == "handoff":
        query_set, queries, skipped = "handoff", queries_from_jsonl(HANDOFF), []
    else:
        query_set, (queries, skipped) = "stage2_dev", queries_stage2_dev(cfg)
    if not queries:
        ap.error("실행할 질의가 없습니다")
    if len({q["query_id"] for q in queries}) != len(queries):
        ap.error("질의 id 중복")
    tag = args.tag or "__".join([query_set, args.dense_model] + ([args.profile] if args.profile != "baseline" else []))

    master = pd.read_csv(args.master, dtype=str, encoding="utf-8-sig", keep_default_na=False)
    if master["servId"].duplicated().any() or master["servId"].str.strip().eq("").any():
        raise ValueError("서비스 ID 중복 또는 빈 값")
    services = master.set_index("servId", drop=False)
    ids = list(services.index)
    names = services["servNm"].to_dict()

    t0 = perf_counter()
    bm25 = BM25Index(ids, build_bm25_texts(services), cache_path=args.bm25_cache or AI / cfg["paths"]["bm25_cache"])
    t1 = perf_counter()
    dense, dense_meta = build_dense(args, ids, build_dense_texts(services, args.profile))
    t2 = perf_counter()
    if list(bm25.ids) != list(dense.ids):
        raise ValueError("BM25와 Dense의 서비스 ID 순서가 다릅니다")

    retrievers = {"bm25": bm25, "dense": dense}
    rankings, qrows, overlaps = [], [], []
    for q in queries:
        found = search_and_fuse(retrievers, {"bm25": q["bm25"], "dense": q["dense"]}, params,
                                known_ids=services.index)
        m = found["modules"]
        qrows.append({"run_tag": tag, "query_id": q["query_id"], "query_source": q["query_source"],
                      "bm25_query": q["bm25"], "dense_query": q["dense"], "status": found["status"],
                      "errors": json.dumps(found["errors"], ensure_ascii=False) if found["errors"] else "",
                      "n_bm25": m["bm25"]["count"], "n_dense": m["dense"]["count"], "n_rrf": m["rrf"]["count"],
                      "bm25_ms": m["bm25"]["ms"], "dense_ms": m["dense"]["ms"], "rrf_ms": m["rrf"]["ms"]})
        rankings += ranking_rows(tag, q["query_id"], found, names, args.save_top)
        overlaps.append(overlap_row(tag, q["query_id"], found))
    t3 = perf_counter()

    tables, runs = args.out / "tables", args.out / "runs"
    tables.mkdir(parents=True, exist_ok=True)
    runs.mkdir(parents=True, exist_ok=True)
    csv = {"index": False, "encoding": "utf-8-sig"}
    pd.DataFrame(rankings).to_csv(tables / f"{tag}__rankings.csv", **csv)
    qdf = pd.DataFrame(qrows)
    qdf.to_csv(tables / f"{tag}__queries.csv", **csv)
    pd.DataFrame(overlaps).to_csv(tables / f"{tag}__overlap.csv", **csv)
    status_counts = qdf["status"].value_counts().to_dict()
    meta = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "run_tag": tag, "query_set": query_set,
        "n_queries": len(queries), "skipped_queries": skipped, "status_counts": status_counts,
        "documents": len(services), "dense": {**dense_meta, "profile": args.profile,
                                               "fields": list(PROFILES[args.profile]),
                                               "fingerprint": dense.fingerprint, "cache_hit": dense.cache_hit},
        "params": params, "save_top": args.save_top,
        "seconds": {"bm25_index": t1 - t0, "dense_index": t2 - t1, "queries": t3 - t2},
        "input_sha256": {str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p): sha(p)
                         for p in [args.master, CONFIG, AI / cfg["paths"]["query_terms"], CORPUS, HANDOFF,
                                   *([args.queries] if args.queries else [])]},
        "code_sha256": {str(p.relative_to(ROOT)): sha(p) for p in
                        [AI / "retrieve/rrf.py", AI / "retrieve/fusion.py", AI / "retrieve/bm25.py",
                         AI / "retrieve/dense.py", AI / "retrieve/dense_texts.py", AI / "query/build.py",
                         Path(__file__).resolve()]},
        "python": sys.version, "platform": platform.platform(), "device": args.device, **environment(),
        "note": NOTE,
    }
    (runs / f"{tag}__metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    ov = pd.DataFrame(overlaps)
    print(json.dumps({"run_tag": tag, "n_queries": len(queries), "skipped": len(skipped),
                      "status_counts": status_counts, "dense_cache_hit": dense.cache_hit,
                      "median_ms": {k: float(qdf[f"{k}_ms"].median()) for k in METHODS},
                      "mean_bm25_dense_common_top10": float(ov["bm25_dense_common_top10"].mean()),
                      "mean_bm25_dense_common_top100": float(ov["bm25_dense_common_top100"].mean()),
                      "out": str(args.out), "note": NOTE}, ensure_ascii=False, indent=2))
    return 0 if not status_counts.get("error") else 1


if __name__ == "__main__":
    sys.exit(main())
