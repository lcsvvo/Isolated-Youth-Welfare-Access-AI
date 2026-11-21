"""질의 쪽 토큰화 진단. 영역 이름과 사용자 글(dev_corpus.jsonl)을 retrieve/bm25.py의 tokenize로 나눠 본다.
--out을 주면 화면 출력을 파일로도 저장한다. 기록은 logs/validation/bm25_inspect/에 둔다.

실행 (ai/ 폴더에서):
    python scripts/bm25_query_diag.py --out ../logs/validation/bm25_inspect/bm25_query_diag.txt
"""
import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

import yaml

AI = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AI))

from retrieve.bm25 import _kiwi, tokenize  # noqa: E402

PIPELINE_QUERY = "취업은 하고 싶은데 밖에 나가는 게 부담스러워요."
JAMO = re.compile(r"[ㄱ-ㅎㅏ-ㅣ]|[;~^]{2,}|[!?.]{2,}|[\U0001F300-\U0001FAFF]")
JAMO_RUN = re.compile(r"[ㄱ-ㅎㅏ-ㅣ]+|[;~^]{2,}|[!?.]{2,}|[\U0001F300-\U0001FAFF]+")
# 한 글자 토큰이 생긴 어절을 사람이 보고 고른 사례 (어절, 유형, 설명)
CASES = [("원데이클래스", "붙여 쓴 외래어", "원데이 클래스 → '원' 조각"),
         ("같은걸로", "띄어쓰기", "같은 걸로 → '걸' 명사 조각"),
         ("다에요", "띄어쓰기·맞춤법", "다예요 → '다' 명사 조각"),
         ("4년전이에요", "띄어쓰기", "4년 전 → 숫자 규칙이 '4년전'으로 합침"),
         ("취준", "은어(문맥)", "단독은 취준, 문장 속에서는 취/준으로 쪼개짐"),
         ("광탈이라", "은어", "광속 탈락 → 광/탈"),
         ("은따", "은어", "은근히 따돌림 → 은/따"),
         ("몰겠어여", "오타·구어", "모르겠어요 → '몰'(사전형 '모르'와 다름)"),
         ("고3때도", "붙여 쓴 줄임말", "고3 때도 → '고' + '3때' (고3이 사라짐)"),
         ("아는 사람", "동음이의", "아는 사람 → '아'(명사)로 잘림, '알다'가 사라짐")]


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
    ap.add_argument("--corpus", default=str(AI / "data" / "dev_corpus.jsonl"))
    ap.add_argument("--domains", default=str(AI / "configs" / "domains.yaml"))
    ap.add_argument("--out", help="화면 출력을 함께 저장할 파일 경로 (UTF-8)")
    args = ap.parse_args()
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        out = open(args.out, "w", encoding="utf-8", newline="\n")
        sys.stdout = Tee(sys.__stdout__, out)

    print("BM25 질의 쪽 토큰화 진단 (retrieve/bm25.py tokenize, verb_mode 기본값)")
    section("1단계 영역 이름 9개 (configs/domains.yaml)")
    doms = yaml.safe_load(open(args.domains, encoding="utf-8"))["domains"]
    for n in (d["name"] if isinstance(d, dict) else d for d in doms):
        print(f"{n:<10} → {tokenize(n)}   (Kiwi: {[(t.form, t.tag) for t in _kiwi.tokenize(n)]})")
    print(f"\npipeline 질의 예: '{PIPELINE_QUERY}' + 영역 '미취업(구직)'")
    print(f"  글만      → {tokenize(PIPELINE_QUERY)}")
    print(f"  영역 붙임 → {tokenize(PIPELINE_QUERY + ' 미취업(구직)')}")

    texts = [json.loads(line)["text"] for line in open(args.corpus, encoding="utf-8") if line.strip()]
    toks = [tokenize(t) for t in texts]
    section(f"dev_corpus.jsonl {len(texts)}건 개요")
    lens = [len(t) for t in toks]
    print(f"글당 토큰 평균 {sum(lens)/len(lens):.1f}, 최소 {min(lens)}, 최대 {max(lens)}, 토큰 0개 글 {lens.count(0)}건")
    df = Counter(w for d in toks for w in dict.fromkeys(d))  # set 대신 순서 고정 (빈도가 같을 때 실행마다 순서가 바뀌지 않게)

    section("자주 나오는 토큰 상위 50 (글 빈도)")
    top = [f"{i:>2}. {w} {n}" for i, (w, n) in enumerate(df.most_common(50), 1)]
    for i in range(0, len(top), 5):
        print("    ".join(top[i:i + 5]))

    section("한 글자 토큰 (글 빈도순)")
    one = [(w, n) for w, n in df.most_common() if len(w) == 1]
    print(f"{len(one)}종: " + ", ".join(f"{w} {n}" for w, n in one))
    ex = {}
    for t, d in zip(texts, toks):
        for w in d:
            if len(w) == 1 and w not in ex:
                ex[w] = t
    for w, _ in one[:15]:
        print(f"  {w}: {ex[w][:60]}")

    section("기호·자모·이모티콘 (ㅠㅠ, ㅋㅋ, ;; 등)")
    raw, kept, tags = Counter(), Counter(), Counter()
    for t, d in zip(texts, toks):
        raw.update(JAMO_RUN.findall(t))
        tags.update((k.form, k.tag) for k in _kiwi.tokenize(t) if JAMO.search(k.form))
        kept.update(w for w in d if JAMO.search(w))
    print(f"원문에 나온 것 {sum(raw.values())}회, 포함 글 {sum(1 for t in texts if JAMO.search(t))}건: "
          + ", ".join(f"{w!r} {n}" for w, n in raw.most_common(20)))
    print("Kiwi 품사: " + ", ".join(f"{f}/{g} {n}" for (f, g), n in tags.most_common(15)))
    print(f"tokenize 결과에 남은 것: {dict(kept) or '없음'}")

    section("띄어쓰기 오류·은어로 잘못 잘린 것으로 보이는 사례 10개")
    print("찾은 방법: 한 글자 토큰이 생긴 어절 263개를 사람이 보고 고름. 대부분(먹어요→먹, 살아→살 등)은 정상 동사 어간이라 제외.")
    print("참고: 흔한 붙여 쓰기(안나가, 못갔어요, 집앞, 가는게 등)는 띄어 써도 토큰이 같았음 → Kiwi가 대부분 잘 처리.")
    print("※ 처음 후보였던 여는게·버는데·치자고는 어절만 보면 틀리지만 문장 속에서는 열·벌·치로 바르게 나뉘어 제외")
    for word, kind, note in CASES:
        sent = next((t for t in texts if word in t), "")
        i = sent.find(word)
        print()
        print(f"- [{kind}] {word}: {note}")
        print(f"  원문 일부: …{sent[max(0, i - 15): i + len(word) + 10]}…")
        print(f"  어절만 tokenize: {tokenize(word)}")
        print(f"  문장 전체 tokenize: {tokenize(sent)}")


if __name__ == "__main__":
    main()
