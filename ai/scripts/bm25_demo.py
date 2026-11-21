"""BM25 검색 데모. 질문 하나를 넣으면 '점수, 서비스명(지역), 겹친 단어'를 보여 준다.

--domain을 주면 pipeline.py처럼 질의 뒤에 영역 이름을 붙인다(여러 번 줄 수 있음).
마스터는 src/recommend/q3_search.py의 MASTER_READ와 같은 옵션으로 읽고, 토큰 캐시는 ai/data/에 둔다(커밋 안 됨).

실행 (ai/ 폴더에서):
    python scripts/bm25_demo.py "밖에 나가기 힘든데 일자리 찾고 싶어요"
    python scripts/bm25_demo.py "밖에 나가기 힘든데 일자리 찾고 싶어요" --domain "미취업(구직)"
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

AI = Path(__file__).resolve().parents[1]
ROOT = AI.parent
sys.path.insert(0, str(AI))

from retrieve.bm25 import BM25Index, build_texts, tokenize  # noqa: E402

MASTER_READ = dict(dtype=str, encoding="utf-8-sig", keep_default_na=False, na_values=[""])  # q3_search.py와 같음


def build_query(text, domains=()):
    """pipeline.py와 같은 방식: 질문 뒤에 영역 이름을 띄어 붙인다."""
    return " ".join([text, *domains])


def search_lines(index, names, regions, query, k=10):
    """결과를 사람이 읽을 줄 목록으로 만든다. 점수가 0인 문서는 뺀다."""
    q = tokenize(query)
    scores = index.bm25.get_scores(q)
    lines = [f"질의 토큰: {', '.join(q)}"]
    for rank, i in enumerate(scores.argsort()[::-1][:k], 1):
        if scores[i] <= 0:
            break
        doc = set(index.doc_tokens[i])
        hit = [w for w in dict.fromkeys(q) if w in doc]
        lines.append(f"{rank:>2}. [{scores[i]:.1f}] {names[i]} ({regions[i] or '지역 빈칸'})   겹친 단어: {', '.join(hit)}")
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("text", help="사용자 질문")
    ap.add_argument("--domain", action="append", default=[], help="질의 뒤에 붙일 영역 이름 (여러 번 가능)")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--master", default=str(ROOT / "data" / "processed" / "services_master.csv"))
    ap.add_argument("--cache", default=str(AI / "data" / "bm25_tokens_master.pkl"),
                    help="문서 토큰 캐시 파일 (처음 실행은 약 1분, 이후 1초 안팎)")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    m = pd.read_csv(args.master, **MASTER_READ).fillna("")
    index = BM25Index(m["servId"], build_texts(m), cache_path=args.cache or None)
    regions = m["region_sido"].tolist() if "region_sido" in m else [""] * len(m)
    query = build_query(args.text, args.domain)
    print(f"질문> {query}")
    print("\n".join(search_lines(index, m["servNm"].tolist(), regions, query, args.k)))


if __name__ == "__main__":
    main()
