"""Q3 질의셋 점검: 개발용 말뭉치와의 중복, 일상어·행정어 어휘 겹침.

- 말뭉치 중복: 사용자 입력 말뭉치 v2(개발용)와 Q3 일상어 질의가 겹치면, Q3 질의셋을 쓰는 별도 서비스 검증에서
  개발용 문장으로 평가하게 된다. 말뭉치 README의 기준(공백·문장부호·ㅠㅜㅋㅎ 제거 후
  max(difflib 비율, 문자 bigram Jaccard) < 0.70)을 그대로 쓴다.
- 어휘 겹침: 일상어 질의가 행정어 질의의 검색 토큰을 그대로 포함하는지. 값은 검수자가 '일상어로 자연스러운
  공통어'인지 판단하는 참고 정보이며 자동 탈락 기준이 아니다.
"""
from __future__ import annotations

import re
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd

from src.preprocess.xlsx_reader import read_sheet
from src.recommend import q3_search as qs

ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "data" / "raw" / "user_input_corpus" / "user_input_corpus_300_v2.xlsx"
CORPUS_SHEET = "입력말뭉치_300"
SIM_MAX = 0.70
_STRIP = re.compile(r"[\s\W_ㅠㅜㅋㅎ]+")


def _clean(s: str) -> str:
    return _STRIP.sub("", str(s))


def similarity(a: str, b: str) -> float:
    a, b = _clean(a), _clean(b)
    if not a or not b:
        return 0.0
    ratio = SequenceMatcher(None, a, b).ratio()
    ga = {a[i : i + 2] for i in range(len(a) - 1)} or {a}
    gb = {b[i : i + 2] for i in range(len(b) - 1)} or {b}
    return max(ratio, len(ga & gb) / len(ga | gb))


def load_corpus(path: Path = CORPUS) -> pd.DataFrame:
    df = read_sheet(path, CORPUS_SHEET)
    return df[["말뭉치ID", "사용자 입력"]]


def query_checks(queries_wide: pd.DataFrame, corpus: pd.DataFrame | None) -> pd.DataFrame:
    rows = []
    for _, r in queries_wide.iterrows():
        d_terms = set(qs.tokenize_t0(r["daily_query"]))
        shared = [t for t in qs.tokenize_t0(r["admin_query"]) if any(t in d or d in t for d in d_terms)]
        row = {"pair_id": r["pair_id"], "shared_admin_terms": "|".join(shared)}
        if corpus is not None and len(corpus):
            sims = [similarity(r["daily_query"], s) for s in corpus["사용자 입력"]]
            i = max(range(len(sims)), key=sims.__getitem__)
            row.update(corpus_max_sim=round(sims[i], 4), corpus_nearest_id=corpus["말뭉치ID"].iloc[i],
                       corpus_overlap=sims[i] >= SIM_MAX)
        rows.append(row)
    return pd.DataFrame(rows)
