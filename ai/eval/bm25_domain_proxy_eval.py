"""BM25 대체 평가: 개발 세트 글로 검색해 상위 10개 서비스의 영역(app_domain)이 글의 정답 영역과 맞는지 센다.

Q3 관련성 정답표가 없을 때 쓰는 대체 지표다. 정식 정확도가 아니다.
- 영역 일치율@10: 상위 10개 중 서비스 영역이 정답 영역과 하나라도 겹치는 비율 (결과가 10개보다 적으면 빈 자리는 불일치)
- 적중률@10: 그런 서비스가 상위 10개 안에 하나라도 있는 글의 비율
- 대상: dev_corpus.jsonl 중 사람 정답 위기 N이고 영역이 있는 글. 서비스에 없는 영역(미래설계·노동)뿐인 글을 뺀 결과도 함께 낸다.
- 질의: (가) 글만, (나) 글 + 정답 영역 이름 (pipeline.py처럼 띄어 붙임)
- 조건: 개선 전(--before 커밋의 bm25.py와 기존 검색 칸), 지금(ai/retrieve), 지금 + 실험 불용어(--extra-stopwords, 실제 파일은 그대로)

실행 (ai/ 폴더에서, 처음은 조건마다 토큰화에 약 1분)
  python -m eval.bm25_domain_proxy_eval
  python -m eval.bm25_domain_proxy_eval --extra-stopwords 힘들 --out ../logs/validation/bm25_inspect/bm25_domain_proxy.txt
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path

import pandas as pd

AI = Path(__file__).resolve().parents[1]
ROOT = AI.parent
RETRIEVE = AI / "retrieve"
RUNS = AI / "eval" / "runs" / "bm25_domain_proxy"
BEFORE_COLS = ["servNm", "target_text", "benefit_text", "apply_text"]  # 개선 전 pipeline.py SEARCH_COLS
UNREACHABLE = {"미래설계", "노동(일하는 중)"}                       # app_domain에 붙은 서비스가 없는 영역
MASTER_READ = dict(dtype=str, encoding="utf-8-sig", keep_default_na=False, na_values=[""])  # q3_search.py와 같음


def load_bm25(folder: Path, name: str):
    spec = importlib.util.spec_from_file_location(f"bm25_{name}", folder / "bm25.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def prepare_conditions(tmp: Path, before: str, extra_stopwords: list[str]) -> list[tuple[str, Path]]:
    """(이름, bm25.py가 있는 폴더) 목록. 실험 조건은 임시 폴더에 복사해서 바꾼다."""
    conds = []
    if before:
        d = tmp / "before"
        d.mkdir()
        src = subprocess.run(["git", "show", f"{before}:ai/retrieve/bm25.py"], cwd=ROOT,
                             capture_output=True, check=True).stdout
        (d / "bm25.py").write_bytes(src)
        conds.append((f"개선 전({before})", d))
    conds.append(("지금", RETRIEVE))
    for i, words in enumerate(extra_stopwords):
        d = tmp / f"stop{i}"
        shutil.copytree(RETRIEVE, d, ignore=shutil.ignore_patterns("__pycache__"))
        with open(d / "stopwords.txt", "a", encoding="utf-8") as f:
            f.write("".join(f"{w.strip()}\n" for w in words.split(",") if w.strip()))
        conds.append((f"지금+불용어 {words}", d))
    return conds


def evaluate(mod, m: pd.DataFrame, rows: list[dict], cache: Path | None) -> list[dict]:
    if hasattr(mod, "build_texts"):
        index = mod.BM25Index(m["servId"], mod.build_texts(m), cache_path=str(cache) if cache else None)
    else:  # 개선 전 코드: 검색 칸 설정이 없어 기존 열을 그대로 이어 붙인다
        index = mod.BM25Index(m["servId"], m[BEFORE_COLS].agg(" ".join, axis=1))
    dom = dict(zip(m["servId"], m["app_domain"].map(lambda v: set(v.split("|")) - {"N", ""})))
    names = dict(zip(m["servId"], m["servNm"]))
    out = []
    for r in rows:
        gold = set(r["gold"]["영역"])
        rec = {"id": r["id"], "text": r["text"], "gold": sorted(gold)}
        for kind, q in (("가", r["text"]), ("나", " ".join([r["text"], *r["gold"]["영역"]]))):
            top = index.search(q, k=10)
            rec[kind] = {"tokens": mod.tokenize(q), "top": [names[s] for s in top[:3]],
                         "match": sum(bool(dom[s] & gold) for s in top)}
        out.append(rec)
    return out


def rates(res: list[dict], idx: list[int], kind: str) -> tuple[float, float]:
    rs = [res[i][kind] for i in idx]
    return sum(r["match"] for r in rs) / (10 * len(rs)), sum(r["match"] > 0 for r in rs) / len(rs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--before", default="60c1cc4", help="개선 전 bm25.py를 가져올 커밋 (빈 값이면 생략)")
    ap.add_argument("--extra-stopwords", action="append", default=[],
                    help="실험 불용어(쉼표로 여러 개). 줄 때마다 조건이 하나씩 늘어남")
    ap.add_argument("--master", default=str(ROOT / "data" / "processed" / "services_master.csv"))
    ap.add_argument("--corpus", default=str(AI / "data" / "dev_corpus.jsonl"))
    ap.add_argument("--out", help="표를 함께 저장할 파일 (UTF-8)")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    warnings.filterwarnings("ignore")

    m = pd.read_csv(args.master, **MASTER_READ).fillna("")
    rows = [json.loads(line) for line in open(args.corpus, encoding="utf-8") if line.strip()]
    rows = [r for r in rows if r["gold"].get("위기") == "N" and r["gold"].get("영역")]
    reach = [i for i, r in enumerate(rows) if set(r["gold"]["영역"]) - UNREACHABLE]
    RUNS.mkdir(parents=True, exist_ok=True)

    results = {}
    with tempfile.TemporaryDirectory() as tmp:
        for k, (name, folder) in enumerate(prepare_conditions(Path(tmp), args.before, args.extra_stopwords)):
            mod = load_bm25(folder, f"c{k}")
            results[name] = evaluate(mod, m, rows, RUNS / f"tokens_c{k}.pkl")
            (RUNS / f"c{k}.json").write_text(json.dumps({"condition": name, "results": results[name]},
                                                         ensure_ascii=False), encoding="utf-8")

    lines = [f"BM25 대체 평가 (영역 일치, 정식 정확도 아님) | 평가 글 {len(rows)}건, "
             f"서비스로 맞힐 수 있는 글 {len(reach)}건 (정답이 미래설계·노동뿐인 {len(rows) - len(reach)}건 제외)",
             "조건 | (가) 일치율@10 적중률@10 | (나) 일치율@10 적중률@10 || 맞힐 수 있는 글 (가) 일치/적중 (나) 일치/적중"]
    every = list(range(len(rows)))
    for name, res in results.items():
        a, b, c, d = rates(res, every, "가"), rates(res, every, "나"), rates(res, reach, "가"), rates(res, reach, "나")
        lines.append(f"{name} | {a[0]:.1%} {a[1]:.1%} | {b[0]:.1%} {b[1]:.1%} || "
                     f"{c[0]:.1%}/{c[1]:.1%} {d[0]:.1%}/{d[1]:.1%}")
    print("\n".join(lines))
    print(f"질의별 결과: {RUNS}")
    if args.out:
        Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
