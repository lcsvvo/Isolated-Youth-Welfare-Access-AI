"""2단계 쿼리 사전(configs/query_terms.yaml) 후보를 뽑아 사람 검수용 CSV로 낸다.

자료는 두 가지만 쓴다.
- dev_corpus.jsonl의 사람 label evidence (일상어 쪽)
- services_master.csv (행정어 쪽: 서비스명 + 요약 + 지원내용, 신청방법은 따로 셈)

평가 자료(Q3 질의 쌍, 평가 페르소나 카드)는 읽지 않는다.

출력 (ai/eval/runs/query_terms_candidates/, git 제외)
- domain_core_candidates.csv : 영역별 마스터 토큰 후보. 영역 안 문서 수, 전체 문서 수, 영역 집중도(lift), 예시 servId
- evidence_term_candidates.csv: 영역별 label evidence 일상어 토큰. evidence 건수, 마스터 출현 문서 수
- access_term_candidates.csv : 접근 표현 후보의 부분문자열 출현 수(본문·신청방법), 노인·장애 동시 출현,
                               Kiwi 토큰, BM25 사용 가능 여부(방문·전화·대면 토큰 생성 여부)
- phrase_counts.csv          : 사전에 넣은 add_terms·positive_terms 문구의 마스터 출현 수 (사전 evidence 기록용)

실행 (ai/ 폴더에서, kiwipiepy 필요)
  python -m eval.build_query_terms_candidates
  python -m eval.build_query_terms_candidates --check-yaml configs/query_terms.yaml
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
import yaml

AI = Path(__file__).resolve().parents[1]
ROOT = AI.parent
DEFAULTS = {
    "corpus": AI / "data" / "dev_corpus.jsonl",
    "services": ROOT / "data" / "processed" / "services_master.csv",
    "domains": AI / "configs" / "domains.yaml",
    "out": AI / "eval" / "runs" / "query_terms_candidates",
}
BODY_COLS = ["servNm", "summary_text", "benefit_text"]
APPLY_COLS = ["apply_text"]
ELDER_DISABLED = re.compile(r"(?:노인|어르신|장애)")
CHANNEL_TOKENS = {"방문", "전화", "대면", "통화"}
# 접근 표현 후보 (사람이 검수할 목록의 출발점). 후보일 뿐 채택이 아님
ACCESS_CANDIDATES = ["화상", "영상통화", "찾아가는", "가정방문", "방문형", "비대면", "재택", "온라인",
                     "온라인 상담", "온라인 신청", "인터넷", "복지로", "정부24", "문자", "채팅", "카카오톡",
                     "모바일", "앱", "우편", "대리 신청", "재가"]


def read_services(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, encoding="utf-8-sig", keep_default_na=False, na_values=[""])


def text_of(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    return df[[c for c in cols if c in df]].fillna("").agg(" ".join, axis=1)


def domain_sets(df: pd.DataFrame) -> pd.Series:
    return df["app_domain"].fillna("N").map(lambda v: set(str(v).split("|")) - {"N", ""})


def core_candidates(df, body_tokens, domains, top=40, min_df=5):
    total = Counter()
    for toks in body_tokens:
        total.update(set(toks))
    n_all = len(body_tokens)
    dsets = domain_sets(df)
    rows = []
    for d in domains:
        idx = [i for i, s in enumerate(dsets) if d in s]
        if not idx:
            continue
        inside = Counter()
        for i in idx:
            inside.update(set(body_tokens[i]))
        for tok, c in inside.items():
            if c < min_df or len(tok) < 2:
                continue
            lift = (c / len(idx)) / (total[tok] / n_all)
            rows.append({"domain": d, "token": tok, "domain_docs": c, "domain_n": len(idx), "all_docs": total[tok],
                         "lift": round(lift, 2),
                         "example_servIds": "|".join([df.iloc[i]["servId"] for i in idx if tok in body_tokens[i]][:3])})
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = out.sort_values(["domain", "lift", "domain_docs"], ascending=[True, False, False])
    return out.groupby("domain").head(top)


def evidence_candidates(corpus, tokenize, body_tokens, domains):
    ev_count = defaultdict(Counter)
    for r in corpus:
        for n in (r.get("label") or {}).get("needs", []) or []:
            if n.get("domain") in domains and n.get("evidence"):
                ev_count[n["domain"]].update(set(tokenize(n["evidence"])))
    df_master = Counter()
    for toks in body_tokens:
        df_master.update(set(toks))
    rows = [{"domain": d, "token": t, "evidence_n": c, "master_docs": df_master.get(t, 0)}
            for d, cnt in ev_count.items() for t, c in cnt.items() if c >= 2 and len(t) >= 1]
    return pd.DataFrame(rows).sort_values(["domain", "evidence_n"], ascending=[True, False])


def phrase_count(texts: pd.Series, phrase: str) -> int:
    return int(texts.str.contains(re.escape(phrase), regex=True).sum())


def phrase_examples(df, texts, phrase, k=3):
    hit = df.loc[texts.str.contains(re.escape(phrase), regex=True), "servId"]
    return "|".join(hit.head(k))


def access_candidates(df, tokenize):
    body, apply_ = text_of(df, BODY_COLS), text_of(df, APPLY_COLS)
    rows = []
    for p in ACCESS_CANDIDATES:
        hit = body.str.contains(re.escape(p), regex=True)
        toks = tokenize(p)
        rows.append({"phrase": p, "body_docs": int(hit.sum()), "apply_docs": phrase_count(apply_, p),
                     "body_with_elder_disabled": int((hit & body.str.contains(ELDER_DISABLED)).sum()),
                     "kiwi_tokens": " ".join(toks),
                     "bm25_safe": not (set(toks) & CHANNEL_TOKENS) and bool(toks),
                     "example_servIds": phrase_examples(df, body, p)})
    return pd.DataFrame(rows)


def yaml_phrases(terms: dict) -> list[tuple[str, str, str]]:
    out = []
    for d, v in (terms.get("domains") or {}).items():
        out += [("core", d, t) for t in v.get("core", [])]
    for e in terms.get("expansions") or []:
        out += [("expansion:" + e["id"], e["domain"], t) for t in e.get("add_terms", [])]
    for a in terms.get("access_terms") or []:
        out += [("access:" + a["id"], a.get("scope", ""), t) for t in a.get("positive_terms", [])]
    return out


def check_yaml(df, terms) -> pd.DataFrame:
    """사전 문구의 마스터 출현 수. 0이면 채택하면 안 됨."""
    body, apply_ = text_of(df, BODY_COLS), text_of(df, APPLY_COLS)
    rows = []
    for kind, owner, t in yaml_phrases(terms):
        rows.append({"kind": kind, "owner": owner, "phrase": t, "body_docs": phrase_count(body, t),
                     "apply_docs": phrase_count(apply_, t), "example_servIds": phrase_examples(df, body, t)})
    return pd.DataFrame(rows)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=str(DEFAULTS["corpus"]))
    ap.add_argument("--services", default=str(DEFAULTS["services"]))
    ap.add_argument("--out", default=str(DEFAULTS["out"]))
    ap.add_argument("--check-yaml", default=None, help="사전 파일의 문구별 마스터 출현 수만 계산")
    args = ap.parse_args(argv)

    from retrieve.bm25 import tokenize  # kiwipiepy 필요. 함수 안에서 지연 import

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df = read_services(Path(args.services))
    if args.check_yaml:
        terms = yaml.safe_load(open(args.check_yaml, encoding="utf-8"))
        res = check_yaml(df, terms)
        res.to_csv(out / "phrase_counts.csv", index=False, encoding="utf-8-sig")
        print(res.to_string())
        return res

    domains = yaml.safe_load(open(DEFAULTS["domains"], encoding="utf-8"))["domains"]
    corpus = [json.loads(l) for l in open(args.corpus, encoding="utf-8") if l.strip()]
    body_tokens = [tokenize(t) for t in text_of(df, BODY_COLS)]
    core_candidates(df, body_tokens, domains).to_csv(out / "domain_core_candidates.csv", index=False,
                                                     encoding="utf-8-sig", quoting=csv.QUOTE_MINIMAL)
    evidence_candidates(corpus, tokenize, body_tokens, domains).to_csv(
        out / "evidence_term_candidates.csv", index=False, encoding="utf-8-sig")
    access_candidates(df, tokenize).to_csv(out / "access_term_candidates.csv", index=False, encoding="utf-8-sig")
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
