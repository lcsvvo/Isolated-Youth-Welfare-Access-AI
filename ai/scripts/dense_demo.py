"""LLM 없이 BGE-M3 / KURE-v1 의미 검색만 실행한다.

저장소 루트에서:
python ai/scripts/dense_demo.py "월세 지원을 받고 싶어요" --model bge-m3
python ai/scripts/dense_demo.py "월세 지원을 받고 싶어요" --model kure-v1
"""
import argparse
import json
from pathlib import Path
import sys
from time import perf_counter

import pandas as pd

AI = Path(__file__).resolve().parents[1]
ROOT = AI.parent
sys.path.insert(0, str(AI))

from core.models import Embedder
from retrieve.dense import DenseIndex
from retrieve.dense_texts import PROFILES, build_texts

MODELS = {"bge-m3": "BAAI/bge-m3", "kure-v1": "nlpai-lab/KURE-v1"}
# 팀 파이프라인의 기존 Dense 검색 필드와 동일. BM25 필드와 독립.
FIELDS = ["servNm", "target_text", "benefit_text", "apply_text"]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("query", nargs="?", help="Step 2 queries.dense 문자열을 그대로 입력")
    ap.add_argument("--queries", type=Path, help="id, dense 키를 가진 JSONL로 여러 질의를 한 번에 검색")
    ap.add_argument("--model", choices=MODELS, default="bge-m3")
    ap.add_argument("--profile", choices=PROFILES, default="baseline")
    ap.add_argument("--model-path", help="다운로드한 모델 폴더 (네트워크 없이 사용)")
    ap.add_argument("--revision", help="재현용 Hugging Face 커밋 ID")
    ap.add_argument("--device", choices=["cpu", "cuda", "mps"])
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-length", type=int, default=8192)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--master", type=Path, default=ROOT / "data/processed/services_master.csv")
    ap.add_argument("--output", type=Path, help="선택: 실행 조건·검색 결과를 JSON으로 저장")
    args = ap.parse_args()
    if bool(args.query) == bool(args.queries):
        ap.error("query 또는 --queries 중 하나만 지정하세요")
    queries = ([json.loads(line) for line in args.queries.read_text(encoding="utf-8-sig").splitlines()
                if line.strip()] if args.queries else [{"id": "manual", "dense": args.query}])
    if not queries or any(not isinstance(q.get("dense"), str) or not q["dense"].strip() or
                          not isinstance(q.get("id"), str) or not q["id"].strip() for q in queries):
        ap.error("각 질의에는 빈 값이 아닌 문자열 id, dense가 필요합니다")
    if len({q["id"] for q in queries}) != len(queries):
        ap.error("질의 id가 중복되었습니다")
    if args.k < 1 or args.batch_size < 1 or not 1 <= args.max_length <= 8192:
        ap.error("빈 질의 불가. k·batch-size는 양수, max-length는 1~8192")
    master = pd.read_csv(args.master, dtype=str, encoding="utf-8-sig", keep_default_na=False)
    fields = list(PROFILES[args.profile])
    missing = [c for c in ["servId", *fields] if c not in master]
    if missing:
        ap.error(f"필수 검색 칼럼 없음: {missing}")
    if not len(master) or master["servId"].str.strip().eq("").any():
        ap.error("서비스 데이터 또는 서비스 ID가 비었습니다")
    texts = build_texts(master, args.profile)
    started = perf_counter()
    model = Embedder(args.model_path or MODELS[args.model],
                     cache_dir=str(AI / "data/models"), device=args.device,
                     batch_size=args.batch_size, max_length=args.max_length,
                     revision=args.revision, local_files_only=bool(args.model_path))
    loaded = perf_counter()
    index = DenseIndex(model, master["servId"], texts,
                       AI / "data" / (f"dense_{args.model}.npz" if args.profile == "baseline"
                                      else f"dense_{args.model}_{args.profile}.npz"))
    indexed = perf_counter()
    table = master.set_index("servId")
    runs = []
    for query in queries:
        before = perf_counter()
        results = index.search_with_scores(query["dense"], args.k)
        runs.append({"id": query["id"], "query": query["dense"],
                     "seconds": perf_counter()-before,
                     "results": [{"rank": rank, "servId": sid, "score": score,
                                  "servNm": table.at[sid, "servNm"]}
                                 for rank, (sid, score) in enumerate(results, 1)]})
    finished = perf_counter()
    report = {"model": args.model, "model_identity": model.cache_key,
              "documents": len(master), "profile": args.profile, "fields": fields,
              "batch_size": args.batch_size, "max_length": args.max_length,
              "fingerprint": index.fingerprint, "cache_hit": index.cache_hit,
              "seconds": {"model_load": loaded-started, "index": indexed-loaded,
                          "query": finished-indexed}, "queries": runs,
              "note": "유사도는 자격 충족 확률이 아님. 관련성 정답표 기반 정확도 평가 전."}
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
