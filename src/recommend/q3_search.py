"""Q3 기존 검색 기준선: SSIS 마스터에 적용하는 단순 키워드 검색.

이 모듈은 복지로·정부24의 실제 검색 엔진을 재현하지 않는다. 같은 SSIS DB에 고정 규칙의
키워드 검색을 적용해 질의 표현(일상어·행정어)과 검색 대상 필드(서비스명·요약 / 상세 본문 포함)의
효과를 분리하기 위한 기준선이다. BM25·임베딩 등 가중치·의미 검색은 포함하지 않는다.

규칙 (사전 고정, 결과를 본 뒤 바꾸면 버전을 올리고 기록한다)
- 정규화: NFKC → 소문자 → 한글·영문·숫자 외 문자는 공백 → 연속 공백 1개.
- 질의 토큰(T0): 공백 분리 → 가장 긴 끝 조사 1개 제거(남은 길이 2자 미만이면 제거하지 않음) → 불용어·1자 토큰 제거 → 중복 제거.
- 문서 매칭: 토큰이 정규화한 문서 문자열에 부분 문자열로 있으면 일치(형태소 분석 없음).
- 문서빈도 불용어: 상세 포함(full) 범위 문서의 DF_MAX(20%) 초과 문서에 나오는 질의 토큰은 버린다.
  같은 목록을 두 범위에 적용한다(범위별로 다르게 버리면 필드 비교가 섞인다). v1 첫 실행에서 '지원'이
  약 90% 문서와 일치해 상위 10이 동점 servId 순서로 정해지는 문제를 확인하고, 관련성 판정 전에 추가했다.
  끄는 조건(df_max=None)은 민감도로 함께 실행한다.
- 점수: 일치한 서로 다른 질의 토큰 수(coordination match). 단어별 가중치 없음.
- 정렬: 점수 내림차순 → 일치 토큰의 문서 내 출현 횟수 합 내림차순 → servId 오름차순.
- 미검색: 점수 0인 서비스는 결과에 넣지 않는다. 결과가 10건 미만일 수 있다.
- 일치 방식: 'or'(주 분석, 1개 이상 일치) / 'and'(민감도, 모든 토큰 일치).
- 민감도 토크나이저(T1): T0 토큰을 문자 bigram으로 쪼개 같은 규칙으로 매칭. 조사·어미 때문에
  생기는 누락이 T0의 토큰화 결과인지 확인하는 용도이며 주 분석이 아니다.
"""
from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

BASELINE_VERSION = "q3_kw_v1.1"
DF_MAX = 0.20

MASTER_READ = dict(dtype=str, encoding="utf-8-sig", keep_default_na=False, na_values=[""])

# 검색 대상 필드. apply_text(신청방법·연락처)는 욕구 내용이 아니라 절차·연락처라 두 범위 모두에서 제외한다.
FIELD_SCOPES: dict[str, tuple[str, ...]] = {
    "short": ("servNm", "summary_text"),
    "full": ("servNm", "summary_text", "target_text", "criteria_text", "benefit_text", "overview_text"),
}

# 끝 조사·어미 일부. 긴 것부터 1회만 제거한다.
JOSA_SUFFIXES: tuple[str, ...] = tuple(sorted({
    "에서는", "으로는", "에게서", "이에요", "인데요", "이라도", "이랑", "에서", "으로", "에게", "한테",
    "까지", "부터", "처럼", "인데", "예요", "라도", "랑", "하고", "은", "는", "이", "가", "을", "를",
    "에", "의", "도", "만", "로", "와", "과",
}, key=len, reverse=True))

# 검색 의도와 무관한 기능어·구어 표현. 질의셋을 작성하기 전에 정한 목록이다.
STOPWORDS: frozenset[str] = frozenset({
    "있을까요", "있을까", "있나요", "있는", "있어요", "있으면", "없어요", "없어서", "없는", "없고", "있었으면",
    "싶어요", "싶은데", "싶고", "싶어도", "좋겠어요", "할까요", "할지", "해야", "하는", "해서", "해요", "했어요",
    "너무", "요즘", "그냥", "어떻게", "어디", "제가", "저는", "저도", "뭐부터", "뭘", "무엇", "같은", "같이",
    "수", "것", "거", "게", "데", "곳", "좀", "더", "또", "다시", "아무", "하나도", "한번도", "번도",
    "있을", "있기", "있다", "되나요", "되는", "될까요", "받을", "받고", "받는", "도움받을", "방법", "때", "처음엔", "인데",
    "저", "나", "내", "제", "이번", "달", "몇", "째", "계속", "잠깐", "잠깐이라도", "그런", "이런",
})

_NON_WORD = re.compile(r"[^0-9a-z가-힣]+")


def normalize(text: object) -> str:
    """검색용 정규화. 결측은 빈 문자열."""
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""
    s = unicodedata.normalize("NFKC", str(text)).lower()
    s = _NON_WORD.sub(" ", s)
    return re.sub(r"\s+", " ", s).strip()


def strip_josa(token: str) -> str:
    """처음 일치한(가장 긴) 끝 조사만 본다. 떼고 남는 길이가 2자 미만이면 토큰을 그대로 둔다
    (예: '집으로'를 '집으'로 자르지 않는다)."""
    for suf in JOSA_SUFFIXES:
        if token.endswith(suf):
            return token[: -len(suf)] if len(token) - len(suf) >= 2 else token
    return token


def tokenize_t0(query: str) -> list[str]:
    out: list[str] = []
    for raw in normalize(query).split():
        if raw in STOPWORDS:
            continue
        tok = strip_josa(raw)
        if len(tok) < 2 or tok in STOPWORDS or tok in out:
            continue
        out.append(tok)
    return out


def tokenize_t1(query: str) -> list[str]:
    out: list[str] = []
    for tok in tokenize_t0(query):
        grams = [tok] if len(tok) <= 2 else [tok[i : i + 2] for i in range(len(tok) - 1)]
        for g in grams:
            if g not in out:
                out.append(g)
    return out


TOKENIZERS = {"t0": tokenize_t0, "t1": tokenize_t1}


@dataclass(frozen=True)
class Condition:
    """검색 조건. cond_id는 결과 파일의 열 값으로 쓴다."""

    scope: str  # short | full
    tokenizer: str = "t0"  # t0 | t1
    mode: str = "or"  # or | and
    df_max: float | None = DF_MAX

    @property
    def cond_id(self) -> str:
        base = f"{self.scope}_{self.tokenizer}_{self.mode}"
        if self.df_max is None:
            return base + "_nodf"
        return base if self.df_max == DF_MAX else f"{base}_df{int(self.df_max * 100)}"


PRIMARY_CONDITIONS = (Condition("short"), Condition("full"))
SENSITIVITY_CONDITIONS = (
    Condition("short", "t1"), Condition("full", "t1"),
    Condition("short", "t0", "and"), Condition("full", "t0", "and"),
    Condition("short", df_max=None), Condition("full", df_max=None),
)


def load_master(path: Path) -> pd.DataFrame:
    m = pd.read_csv(path, **MASTER_READ)
    needed = {"servId", *(f for fs in FIELD_SCOPES.values() for f in fs)}
    missing = needed - set(m.columns)
    if missing:
        raise ValueError(f"마스터에 검색 필드가 없음: {sorted(missing)}")
    if m["servId"].duplicated().any():
        raise ValueError("servId 중복")
    return m


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class KeywordIndex:
    """필드 범위별 정규화 문서 문자열. servId 오름차순으로 고정한다."""

    def __init__(self, master: pd.DataFrame):
        m = master.sort_values("servId", kind="mergesort").reset_index(drop=True)
        self.serv_ids: list[str] = m["servId"].tolist()
        self.docs: dict[str, list[str]] = {}
        for scope, fields in FIELD_SCOPES.items():
            joined = m[list(fields)].fillna("").astype(str).agg(" ".join, axis=1)
            self.docs[scope] = [normalize(t) for t in joined]
        self._df: dict[str, float] = {}

    def doc_freq(self, term: str) -> float:
        """full 범위 기준 문서빈도(비율)."""
        if term not in self._df:
            docs = self.docs["full"]
            self._df[term] = sum(term in d for d in docs) / len(docs) if docs else 0.0
        return self._df[term]

    def query_terms(self, query: str, cond: Condition) -> tuple[list[str], list[str]]:
        """(사용 토큰, 문서빈도로 버린 토큰)."""
        terms = TOKENIZERS[cond.tokenizer](query)
        if cond.df_max is None:
            return terms, []
        kept = [t for t in terms if self.doc_freq(t) <= cond.df_max]
        return kept, [t for t in terms if t not in kept]

    def search(self, query: str, cond: Condition) -> pd.DataFrame:
        """점수 > 0인 전체 순위. 열: rank, servId, score, tf, n_terms, matched_terms."""
        terms, _ = self.query_terms(query, cond)
        rows = []
        if terms:
            for sid, doc in zip(self.serv_ids, self.docs[cond.scope]):
                matched = [t for t in terms if t in doc]
                if not matched:
                    continue
                if cond.mode == "and" and len(matched) < len(terms):
                    continue
                rows.append((sid, len(matched), sum(doc.count(t) for t in matched), "|".join(matched)))
        res = pd.DataFrame(rows, columns=["servId", "score", "tf", "matched_terms"])
        res = res.sort_values(["score", "tf", "servId"], ascending=[False, False, True], kind="mergesort").reset_index(drop=True)
        res.insert(0, "rank", range(1, len(res) + 1))
        res["n_terms"] = len(terms)
        return res[["rank", "servId", "score", "tf", "n_terms", "matched_terms"]]


def cutoff_ties(ranked: pd.DataFrame, k: int = 10) -> dict:
    """k번째 결과의 동점 규모. score 동점(점수만)과 score·tf 동점(servId 순서로만 갈리는 경우)을 따로 센다."""
    n = len(ranked)
    if n <= k:
        return {"n_hits": n, "kth_score": (int(ranked["score"].iloc[-1]) if n else 0),
                "n_tied_kth_score": 0, "score_tie_crosses_cutoff": False,
                "n_tied_kth_score_tf": 0, "servid_order_decides": False}
    s_k, tf_k = ranked["score"].iloc[k - 1], ranked["tf"].iloc[k - 1]
    above = int((ranked["score"] > s_k).sum())
    tied = int((ranked["score"] == s_k).sum())
    same = (ranked["score"] == s_k) & (ranked["tf"] == tf_k)
    above_tf = int(((ranked["score"] > s_k) | ((ranked["score"] == s_k) & (ranked["tf"] > tf_k))).sum())
    tied_tf = int(same.sum())
    return {"n_hits": n, "kth_score": int(s_k), "n_tied_kth_score": tied,
            "score_tie_crosses_cutoff": above + tied > k,
            "n_tied_kth_score_tf": tied_tf, "servid_order_decides": above_tf + tied_tf > k}
