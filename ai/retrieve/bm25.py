"""2-a 단어 검색. Kiwi로 형태소를 나눈 뒤 BM25로 점수 계산.

문서와 질의는 반드시 같은 tokenize()를 거친다 (다른 단계도 이 함수를 쓴다).
- Kiwi에 넣기 전에 URL·이메일을 지우고, 가운뎃점 변형(ㆍ ‧ ･ •)을 ·(U+00B7)로 통일한다.
- 접두사(XPN: 비·미·저·무·재 등)가 바로 뒤 명사와 붙어 있으면 한 토큰으로 합친다: 비대면, 미취업, 저소득.
  합친 뒤에는 뒤 명사만 따로 남기지 않는다(비대면 ≠ 대면).
- 명사 바로 뒤에 붙은 접미사 생·권·인·형은 붙인 토큰과 원래 명사를 함께 남긴다: 대학생 → 대학생, 대학.
  그 밖의 접미사(적·들·별·층 등)는 지금처럼 버린다.
- 숫자(SN)가 바로 뒤 명사·단위와 붙어 있으면 이어진 명사까지 한 토큰으로 합친다: 20대, 1인가구.
  첫 단위(대·인 등) 뒤에 이어진 명사(가구)는 따로도 남긴다. 이상·이하·미만·초과·이내는 합치지 않는다: 65세이상 → 65세, 이상.
- "붙어 있다"는 원문 위치(start + len)가 다음 토큰의 start와 같다는 뜻이다. 띄어 쓴 경우는 합치지 않는다.
- 동사·형용사(VV·VA) 처리는 verb_mode로 고른다.
  "drop"(A): 모두 뺀다. "stopwords"(B): 남기되 stopwords.txt에 있는 흔한 조각(하·있·없 등)은 뺀다.
  stopwords.txt의 형태는 두 모드 모두에서 품사와 상관없이 뺀다.
- user_dict.txt의 단어를 Kiwi에 등록한다(고립은둔, 재가 등). 등록한 단어는 주변 단어 분석도 바꿀 수 있어 꼭 필요한 것만 넣는다.
  사전 단어가 여러 명사로 나뉘는 합성어면 통째 토큰 뒤에 부분 토큰도 남긴다: 고립은둔 → 고립은둔, 고립, 은둔.
  그래서 띄어 쓴 질의("고립 은둔")도 붙여 쓴 문서("고립은둔")와 맞는다.
  부분은 그 단어만 빼고(blocklist) 다시 분석해 명사들로 정확히 나뉠 때만 쓴다(재가처럼 안 나뉘면 부분 없음).
  셋째 칸에 부분을 직접 쓰면 그것을 쓴다(부분1+부분2, 없음은 -).
- abbrev.txt의 가운뎃점 줄임말은 미리 분석한 단어로 등록한다: 취·창업 → 취업, 창업.

BM25 색인 텍스트는 build_texts()가 BM25_FIELDS(열, 반복 횟수)로 만든다. 의미 검색(dense) 텍스트와는 따로다.
BM25Index(cache_path=...)를 주면 문서 토큰화 결과를 파일로 저장해 다시 쓴다.
캐시 키(cache_key)는 문서 ID·텍스트(데이터 내용과 BM25_FIELDS가 반영됨), 사전·불용어·줄임말 파일,
이 파일(bm25.py, 규칙 코드와 옵션 상수), verb_mode, BM25_FIELDS, kiwipiepy·kiwipiepy_model 버전으로 만든다. 하나라도 바뀌면 다시 토큰화한다.

Kiwi 버전: 분절 결과는 kiwipiepy_model(사전·언어 모델)과 kiwipiepy 버전에 따라 달라진다.
사전·불용어·분절 정답 목록은 TESTED_KIWI 버전에서 맞춘 것이고 requirements.txt도 그 버전으로 고정한다.
다른 버전이면 경고를 남긴다(모델 0.24.0은 서비스 5,288건 중 4,230건의 토큰이 달라짐).

검색 결과 순서: 점수가 같으면 넣은 문서 순서(ids 순서)를 따른다. DenseIndex와 같은 규칙이다.
"""
import hashlib
import logging
import os
import pickle
import re
from pathlib import Path

import kiwipiepy
import kiwipiepy_model
import numpy as np
from kiwipiepy import Kiwi, MorphemeSet
from rank_bm25 import BM25Okapi

log = logging.getLogger(__name__)

KEEP_TAGS = ("NNG", "NNP", "VV", "VA", "SL", "XR")  # 명사·동사·형용사 어간·외국어·어근
PREFIX_NOUN_TAGS = ("NNG", "NNP")                   # 접두사와 합칠 명사
NUMBER_UNIT_TAGS = ("NNG", "NNP", "NNB")            # 숫자와 합칠 명사·단위
NUMBER_STOP_WORDS = ("이상", "이하", "미만", "초과", "이내")  # 숫자 묶음에 합치지 않는 비교어
NOUN_SUFFIXES = ("생", "권", "인", "형")             # 앞 명사에 붙여 남길 접미사(XSN)
VERB_TAGS = ("VV", "VA")                            # VV-R, VA-I 같은 불규칙 활용 태그도 포함
VERB_MODES = ("drop", "stopwords")
VERB_MODE = "stopwords"
STOPWORDS_PATH = Path(__file__).with_name("stopwords.txt")
USER_DICT_PATH = Path(__file__).with_name("user_dict.txt")
ABBREV_PATH = Path(__file__).with_name("abbrev.txt")
BM25_FIELDS = (("servNm", 2), ("summary_text", 1), ("target_text", 1),   # 4단계 비교 ③
               ("criteria_text", 1), ("benefit_text", 1))
TESTED_KIWI = ("0.23.2", "0.23.0")  # (kiwipiepy, kiwipiepy_model). requirements.txt 고정 버전과 같게 유지
KIWI_VERSIONS = (kiwipiepy.__version__, getattr(kiwipiepy_model, "__version__", "?"))
if KIWI_VERSIONS != TESTED_KIWI:
    log.warning("kiwipiepy %s / kiwipiepy_model %s: 검증한 버전(%s / %s)과 달라 분절·검색 결과가 달라질 수 있음",
                *KIWI_VERSIONS, *TESTED_KIWI)
USER_WORD_SCORE = 10  # 0이면 "저소득 한부모가족"처럼 문맥에 따라 사전 단어를 놓친다 (한부모 334/384건)
URL_RE = re.compile(r"(?:https?://|www\.)[^\s()<>\[\]]+"
                    r"|[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+"
                    r"|\b[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.(?:kr|com|net|org)\b(?:/[^\s()<>\[\]]*)?")
DOT_RE = re.compile(r"(?<=\w)[ㆍ‧･•](?=\w)")
# 칸 전체가 정보 없는 안내 문구면 BM25 글에서 빈 값으로 본다. 문장 속 "없음"(소득기준 없음 등)은 건드리지 않는다.
NO_INFO_RE = re.compile(
    r"(?:(?:선정기준은|자세한 내용은)\s*)?지원\s*대상(?:의\s*내용|,\s*신청방법의\s*내용)?을\s*참고해\s*주시기\s*바랍니다\.?"
    r"|[*\s]*(?:해당\s*(?:사항\s*)?|급여\s*서비스\s*)?없음\.?"
    r"|['\"]?선정기준['\"]?\s*내용\s*참고|붙임\s*참조|상동\.?|[\W_]*")


def _read_lines(path):
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [w.strip() for w in lines if w.strip() and not w.lstrip().startswith("#")]


def load_stopwords(path=STOPWORDS_PATH):
    return set(_read_lines(path))


def load_user_dict(path=USER_DICT_PATH, abbrev_path=ABBREV_PATH):
    """사용자 사전·줄임말을 등록한 Kiwi와 {합성어: [부분 명사들]}를 돌려준다."""
    kiwi = Kiwi()
    rows = [line.split() for line in _read_lines(path)]
    # Kiwi는 단어를 등록한 뒤 첫 분석 때 모델을 다시 만든다(수 초). 등록과 분석이 번갈아 일어나지 않게 모아서 한다.
    for word, tag, *_ in rows:
        kiwi.add_user_word(word, tag, USER_WORD_SCORE)
    for form, *tokens in (line.split() for line in _read_lines(abbrev_path)):
        while True:
            try:
                kiwi.add_pre_analyzed_word(form, [(t, "NNG") for t in tokens], USER_WORD_SCORE)
                break
            except ValueError as e:  # Kiwi가 모르는 결과 단어(이미용, 모부자)는 먼저 등록해야 쓸 수 있다
                unknown = re.search(r"morpheme (\S+)/NNG", str(e))
                if not unknown or unknown.group(1) not in tokens:
                    raise
                kiwi.add_user_word(unknown.group(1), "NNG")
    parts = {}
    for word, tag, *given in rows:
        if given:
            if given[0] != "-":
                parts[word] = given[0].split("+")
            continue
        sub = kiwi.tokenize(word, blocklist=MorphemeSet(kiwi, [(word, tag)]))
        if len(sub) > 1 and all(t.tag in PREFIX_NOUN_TAGS for t in sub) and "".join(t.form for t in sub) == word:
            parts[word] = [t.form for t in sub]
    return kiwi, parts


STOPWORDS = load_stopwords()
_kiwi, COMPOUND_PARTS = load_user_dict()


def _attached(a, b):
    return a.start + a.len == b.start


def normalize(text):
    """Kiwi에 넣기 전 정리: URL·이메일 지우기, 가운뎃점 통일."""
    return DOT_RE.sub("·", URL_RE.sub(" ", text or ""))


def tokenize(text, verb_mode=None):
    verb_mode = verb_mode or VERB_MODE
    if verb_mode not in VERB_MODES:
        raise ValueError(f"verb_mode는 {VERB_MODES} 중 하나: {verb_mode}")
    toks = _kiwi.tokenize(normalize(text))
    out, i = [], 0

    def noun(form, last, j):
        """명사 하나를 내보내고 다음 위치를 돌려준다. 붙은 생·권·인·형이 있으면 붙인 토큰을 먼저 낸다."""
        suf = toks[j] if j < len(toks) else None
        if suf is not None and suf.tag == "XSN" and suf.form in NOUN_SUFFIXES and _attached(last, suf):
            out.append(form + suf.form)
            j += 1
        out.append(form)
        out.extend(COMPOUND_PARTS.get(form, ()))
        return j

    while i < len(toks):
        t = toks[i]
        nxt = toks[i + 1] if i + 1 < len(toks) else None
        if t.tag == "XPN" and nxt is not None and nxt.tag in PREFIX_NOUN_TAGS and _attached(t, nxt):
            i = noun(t.form + nxt.form, nxt, i + 2)
            continue
        if t.tag == "SN":
            j = i + 1
            while (j < len(toks) and toks[j].tag in NUMBER_UNIT_TAGS and toks[j].form not in NUMBER_STOP_WORDS
                   and _attached(toks[j - 1], toks[j])):
                j += 1
            if j > i + 1:
                out.append("".join(x.form for x in toks[i:j]))
                out.extend(x.form for x in toks[i + 2:j] if x.tag.startswith(KEEP_TAGS))
                i = j
                continue
        if t.tag in PREFIX_NOUN_TAGS:
            i = noun(t.form, t, i + 1)
            continue
        if t.tag.startswith(KEEP_TAGS) and not (verb_mode == "drop" and t.tag.startswith(VERB_TAGS)):
            out.append(t.form)
        i += 1
    return [w for w in out if w not in STOPWORDS]


def build_texts(df, fields=BM25_FIELDS):
    """BM25 색인용 텍스트. 열마다 반복 횟수만큼 이어 붙인다. 표에 없는 열은 경고를 남기고 건너뛴다.
    칸 전체가 안내 문구(NO_INFO_RE)면 빈 값으로 본다. 원본 표는 바꾸지 않는다."""
    missing = [c for c, _ in fields if c not in df]
    if missing:
        log.warning("BM25 검색 칸이 표에 없어 건너뜀: %s", ", ".join(missing))
    cols = [(c, n) for c, n in fields if c in df]
    if not cols:
        return [""] * len(df)
    values = [["" if NO_INFO_RE.fullmatch(v.strip()) else v for v in df[c].fillna("").astype(str)] for c, _ in cols]
    return [" ".join(" ".join([v] * n) for v, (_, n) in zip(row, cols)) for row in zip(*values)]


CACHE_VERSION = 2  # 2: 사전에서 '은둔' 제외, 캐시 키에 kiwipiepy_model 버전 추가
SETTING_FILES = (USER_DICT_PATH, STOPWORDS_PATH, ABBREV_PATH, Path(__file__))


def cache_key(ids, texts, verb_mode=None):
    """토큰화 결과를 바꿀 수 있는 모든 것을 묶은 해시."""
    h = hashlib.sha256()
    for part in (repr((CACHE_VERSION, verb_mode or VERB_MODE, BM25_FIELDS, KIWI_VERSIONS)).encode(),
                 *(Path(p).read_bytes() for p in SETTING_FILES),
                 "\x1f".join(map(str, ids)).encode(),
                 "\x1e".join(texts).encode()):
        h.update(hashlib.sha256(part).digest())
    return h.hexdigest()


def tokenize_docs(ids, texts, verb_mode=None, cache_path=None):
    """문서 토큰 목록. cache_path가 있으면 키가 같을 때 파일에서 읽고, 아니면 토큰화해서 저장한다."""
    texts = list(texts)
    key = cache_key(ids, texts, verb_mode) if cache_path else None
    if cache_path and os.path.exists(cache_path):
        try:
            with open(cache_path, "rb") as f:
                saved = pickle.load(f)
            if saved.get("key") == key:
                return saved["tokens"]
            log.info("BM25 토큰 캐시가 설정·데이터와 맞지 않아 다시 만듦: %s", cache_path)
        except Exception as e:  # 깨진 파일이면 다시 만든다
            log.warning("BM25 토큰 캐시를 읽지 못해 다시 만듦 (%s): %s", e, cache_path)
    tokens = [tokenize(t, verb_mode) for t in texts]
    if cache_path:
        tmp = f"{cache_path}.tmp"
        with open(tmp, "wb") as f:
            pickle.dump({"key": key, "tokens": tokens}, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, cache_path)
    return tokens


class BM25Index:
    def __init__(self, ids, texts, verb_mode=None, cache_path=None):
        self.ids = list(ids)
        if len(set(self.ids)) != len(self.ids):  # DenseIndex와 같은 검사. 중복이면 RRF 결합에서 순위가 섞인다
            raise ValueError("서비스 ID가 중복되었습니다")
        self.verb_mode = verb_mode
        self.doc_tokens = tokenize_docs(self.ids, texts, verb_mode, cache_path)
        self.bm25 = BM25Okapi([t or ["_"] for t in self.doc_tokens])

    def search(self, query, k=100):
        """점수 높은 순. 점수가 같으면 넣은 순서(ids 순서)대로. 점수 0인 문서는 내지 않는다."""
        scores = self.bm25.get_scores(tokenize(query, self.verb_mode))
        order = np.argsort(-scores, kind="stable")[:k]
        return [self.ids[i] for i in order if scores[i] > 0]
