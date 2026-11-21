"""현재 BM25(retrieve/bm25.py)를 실제 SSIS 마스터에 돌려 보는 진단 스크립트. --out을 주면 화면 출력을 파일로도 저장한다.

색인·검색은 retrieve/bm25.py의 tokenize와 BM25Index를 그대로 쓴다.
검색 대상 열은 --cols로 바꿀 수 있고(열:반복횟수), 주지 않으면 retrieve/bm25.py의 BM25_FIELDS와 같다.
마스터는 src/recommend/q3_search.py의 MASTER_READ와 같은 옵션으로 읽는다.
개선 전후 비교용 기록은 logs/validation/bm25_inspect/에 둔다.

실행 (ai/ 폴더에서):
    python scripts/bm25_inspect.py
    python scripts/bm25_inspect.py --master ../data/processed/services_master.csv \
        --queries ../data/raw/q3_query_pairs/q3_query_pairs_v2.csv --seed 20261111 \
        --out ../logs/validation/bm25_inspect/bm25_inspect_baseline.txt
    python scripts/bm25_inspect.py --cols servNm,target_text,benefit_text,apply_text   # 4단계 이전 열
"""
import argparse
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

AI = Path(__file__).resolve().parents[1]
ROOT = AI.parent
sys.path.insert(0, str(AI))

from retrieve import bm25  # noqa: E402
from retrieve.bm25 import BM25_FIELDS, BM25Index, _kiwi, build_texts  # noqa: E402

DEFAULT_COLS = ",".join(f"{c}:{n}" if n > 1 else c for c, n in BM25_FIELDS)
MASTER_READ = dict(dtype=str, encoding="utf-8-sig", keep_default_na=False, na_values=[""])  # q3_search.py와 같음
TOP_K = 100            # config.yaml retrieval.top_k_each
ADMIN_WORDS = ["고립은둔", "고립·은둔", "은둔청년", "일경험", "월세지원", "마음건강", "재가", "비대면", "심리상담",
               "취업지원", "구직활동", "자립준비청년", "주거급여", "바우처", "멘토링", "직업훈련", "생활비", "긴급복지",
               "미취업", "저소득층", "1인가구", "20대", "무주택"]


def tokenize(text):
    """--verb-mode를 반영한 retrieve.bm25.tokenize."""
    return bm25.tokenize(text, bm25.VERB_MODE)


def parse_cols(spec):
    """"servNm:2,target_text" → [("servNm", 2), ("target_text", 1)]."""
    out = []
    for item in spec.split(","):
        col, _, n = item.strip().partition(":")
        out.append((col, int(n or 1)))
    return out


def search_scored(index, query, k=TOP_K):
    """BM25Index.search와 같은 결과에 점수를 붙여 돌려준다 (진단용)."""
    scores = index.bm25.get_scores(tokenize(query))
    order = scores.argsort()[::-1][:k]
    return [(index.ids[i], float(scores[i])) for i in order if scores[i] > 0]


def verb_ratio(query):
    """tokenize 결과 중 Kiwi가 동사·형용사(VV·VA)로 분석한 형태와 같은 토큰의 비율. 토큰이 없으면 0."""
    toks = tokenize(query)
    verbs = {t.form for t in _kiwi.tokenize(query) if t.tag.startswith(("VV", "VA"))}
    return sum(t in verbs for t in toks) / len(toks) if toks else 0.0


class Tee:
    """화면과 파일에 함께 쓴다."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s)

    def flush(self):
        for st in self.streams:
            st.flush()


def section(title):
    print(f"\n{'=' * 8} {title} {'=' * 8}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--master", default=str(ROOT / "data" / "processed" / "services_master.csv"))
    ap.add_argument("--queries", default=str(ROOT / "data" / "raw" / "q3_query_pairs" / "q3_query_pairs_v2.csv"))
    ap.add_argument("--seed", type=int, default=20261111)
    ap.add_argument("--out", help="화면 출력을 함께 저장할 파일 경로 (UTF-8)")
    ap.add_argument("--verb-mode", choices=bm25.VERB_MODES, default=bm25.VERB_MODE,
                    help="동사·형용사 처리: drop(A, 모두 제외) / stopwords(B, stopwords.txt 조각만 제외)")
    ap.add_argument("--cols", default=DEFAULT_COLS,
                    help="검색 대상 열과 반복 횟수, 예: servNm:2,summary_text,target_text (기본: bm25.py BM25_FIELDS)")
    ap.add_argument("--cache", help="문서 토큰 캐시 파일 경로, 예: data/bm25_tokens_inspect.pkl (기본: 사용 안 함)")
    args = ap.parse_args()
    bm25.VERB_MODE = args.verb_mode
    cols = parse_cols(args.cols)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        out = open(args.out, "w", encoding="utf-8", newline="\n")
        sys.stdout = Tee(sys.__stdout__, out)

    m = pd.read_csv(args.master, **MASTER_READ)
    m = m.fillna("")
    texts = build_texts(m, cols)
    names = dict(zip(m["servId"], m["servNm"]))

    section("색인")
    t0 = time.perf_counter()
    index = BM25Index(m["servId"], texts, cache_path=args.cache)
    t1 = time.perf_counter()
    toks = index.doc_tokens
    lens = np.array([len(t) for t in toks])
    print(f"문서 {len(m):,}건, 검색 열 {', '.join(f'{c}×{n}' for c, n in cols)}, verb_mode {bm25.VERB_MODE}")
    print(f"BM25Index 생성 {t1 - t0:.1f}초 (토큰 캐시 {args.cache or '사용 안 함'})")
    print(f"문서별 토큰 수 평균 {lens.mean():.1f}, 중앙값 {np.median(lens):.0f}, 최소 {lens.min()}, 최대 {lens.max()}")
    print(f"토큰 0개 문서 {int((lens == 0).sum())}건")

    section("무작위 서비스 5건: 원문 → 토큰")
    rng = np.random.default_rng(args.seed)
    for i in rng.choice(len(m), 5, replace=False):
        print(f"\n[{m.servId[i]}] {m.servNm[i]} | {m.target_text[i][:80]}")
        print("  →", " ".join(tokenize(m.servNm[i] + " " + m.target_text[i][:80])))

    section("문서 빈도 상위 40 토큰 (불용어 후보)")
    df = Counter(t for doc in toks for t in set(doc))
    for rank, (tok, n) in enumerate(df.most_common(40), 1):
        print(f"{rank:>2}. {tok}\t{n:,} ({n / len(m):.0%})", end="\n" if rank % 4 == 0 else "    ")

    section("Q3 질의쌍 30개")
    q = pd.read_csv(args.queries, dtype=str, encoding="utf-8-sig", keep_default_na=False)
    rows = []
    for r in q.itertuples(index=False):
        res = {}
        print(f"\n[{r.pair_id}] {r.query_group} | {r.intent}")
        vr = {}
        for kind in ("daily", "admin"):
            query = getattr(r, f"{kind}_query")
            hits = search_scored(index, query)
            res[kind] = [sid for sid, _ in hits]
            vr[kind] = verb_ratio(query)
            print(f"  {kind}: {query}")
            print(f"    토큰: {tokenize(query)} (동사·형용사 {vr[kind]:.0%})")
            print(f"    결과 {len(hits)}건, 상위 5: "
                  + " / ".join(f"{names[sid]}({s:.1f})" for sid, s in hits[:5]))
        overlap = len(set(res["daily"][:10]) & set(res["admin"][:10]))
        print(f"  상위 10 겹침: {overlap}")
        rows.append({"pair": r.pair_id, "daily_n": len(res["daily"]), "admin_n": len(res["admin"]),
                     "daily_vr": vr["daily"], "admin_vr": vr["admin"], "overlap10": overlap,
                     "admin_top3": " / ".join(names[sid] for sid in res["admin"][:3])})

    s = pd.DataFrame(rows)
    section("질의별 요약 (동사·형용사 비율 daily/admin, 상위 10 겹침, admin 상위 3)")
    for x in s.itertuples(index=False):
        print(f"{x.pair}  {x.daily_vr:>4.0%} / {x.admin_vr:>4.0%}  겹침 {x.overlap10:>2}  | {x.admin_top3}")

    section("Q3 요약")
    print(f"쌍 {len(s)}개")
    print(f"결과 수 중앙값 daily {s.daily_n.median():.0f}, admin {s.admin_n.median():.0f} (상한 {TOP_K})")
    print(f"질의 토큰 중 동사·형용사 비율 평균 daily {s.daily_vr.mean():.0%}, admin {s.admin_vr.mean():.0%}")
    print(f"같은 쌍 상위 10 교집합 평균 {s.overlap10.mean():.2f}개, 0개인 쌍 {int(s.overlap10.eq(0).sum())}개")

    section("행정어 토큰화")
    vocab_df = df
    for w in ADMIN_WORDS:
        t = tokenize(w)
        print(f"{w:<10} → {t}  (문서 빈도: " + ", ".join(f"{x} {vocab_df.get(x, 0):,}" for x in t) + ")")

    section("사용자 사전 단어: 붙여 쓴 질의와 띄어 쓴 질의의 상위 3 (점수 부풀림 확인)")
    print(f"합성어 부분: {bm25.COMPOUND_PARTS}")
    for w in bm25._read_lines(bm25.USER_DICT_PATH):
        word = w.split()[0]
        for query in dict.fromkeys([word, " ".join(bm25.COMPOUND_PARTS.get(word, [word]))]):
            hits = search_scored(index, query)
            print(f"{query:<10} → {tokenize(query)} 결과 {len(hits)}건 | "
                  + " / ".join(f"{names[sid]}({sc:.1f})" for sid, sc in hits[:3]))


if __name__ == "__main__":
    main()
