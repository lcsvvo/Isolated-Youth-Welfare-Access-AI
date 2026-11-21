"""2단계 검색 쿼리 생성: 확인 세션(stage2-v1) → BM25 쿼리, Dense 쿼리.

query_source_text
- original_text를 문장으로 나눈 뒤 제약 evidence, activity_evidence, 접근 의도(want) 문장,
  삭제·변경된 영역의 evidence 문장을 뺀다. 회피 절만으로 근거를 제거할 수 있으면 남은 필요 내용을 쓴다.
  위치는 core.utils.normalize(공백·부호 무시)로 찾는다.
- 다음이면 원문을 쓰지 않는다(template): 삭제·변경 영역 evidence가 남은 영역 evidence와 겹침,
  빼야 할 evidence가 원문 전체, 위치를 못 찾음. 영역 삭제·변경이 있으면 Dense는 항상 template.

BM25 = dedup(main 핵심어 + sub 핵심어 + 확장어 + tokenize(query_source_text) + 접근 긍정어)
- 영역명 문자열을 기계적으로 붙이지 않는다. 마스터 근거로 고른 핵심어는 영역명과 겹쳐도 쓴다(취업, 주거 등).
- 나이, 지역은 넣지 않는다. 사전 토큰에서 banned·hold는 뺀다. banned는 원문 토큰에서도 뺀다.
- 현재 BM25Index.search(문자열)와 호환되게 '사전 토큰. 원문 문장. 접근 긍정어'로 잇는다.
  재토큰화 때문에 원문 단어를 빼지 않는다. 의도한 토큰과 검색 시 토큰을 모두 기록한다.
- 삭제·변경 전 영역의 핵심어·확장어 토큰은 어디서 왔든 뺀다. 그래도 남으면 template로 바꾸고 기록한다.
- 회피 제약 채널 토큰(방문·대면, 전화·통화)은 어떤 접근 변형에서도 넣지 않는다.

회피 접근(방문·전화 불가·부담)은 쿼리 글이 아니라 access 필드로만 전달한다(BM25·Dense 공통)
- 어휘 감지(strip_avoid): 채널어(방문, 주민센터, 전화 등. '면접'은 취업 욕구어라 넣지 않음)와 부담어(부담·무서·어렵·힘들·싫·못·불가 등)가
  같은 절에 있으면 그 절을 뺀다. '전화 말고 문자나 카톡으로'처럼 말고/보다/대신 앞 채널 구절만 뺀다(부정어만 지우지 않음).
  남은 필요 내용('일자리를 찾고 싶은데')은 유지한다. 추출기 evidence가 없어도(확인 화면에서 사용자가 추가한 제약,
  추출기 누락) 원문에서 직접 감지한다.
- 제약·활동 evidence가 문장 전체를 덮어도 원문 전체를 버리지 않는다. 그 구절만 빼고 문장 안 욕구 evidence(없으면 나머지)를 쓴다.
  남는 내용이 3자 미만이면 이전처럼 template.
- 원문 전체를 덮지 않아도 활동 문장·욕구 evidence 문장은 통째로 버리지 않고 그 구절만 뺀다(남는 상황 서술은 검색 단서).
  제약만 있는 문장은 남은 조각에 접근 방식 말이 섞이기 쉬워 원문이 다 빠질 때만 살린다.
- 제약 evidence 문장도 감지되면 통째로 버리지 않고 회피 절만 빼고 남은 내용을 쓴다. 감지 못하면 이전처럼 문장 통째로 뺀다.
- '불가능한 건 아니다'는 회피 제약으로 세지 않으며, 뒤에 별도 회피 선호가 있으면 그 표현을 판단한다.
- 감지된 회피 채널 토큰은 BM25 excluded에 더하고, Dense 문장에서도 채널어를 뺀다. 감지 결과는 query_expansion.lexical_avoid에 기록한다.
  어휘 감지만으로 access 항목을 만들지는 않는다(사용자 확정 제약만 access).

접근 변형(access_mode): A 없음 / B1 want 문장만 긍정어(이용 우선, 상한) / B2 B1 + 회피 제약 유도어(실험용) /
C 쿼리는 A와 같음. access 필드는 모든 변형에서 채운다. B2 추론분은 access_inferred로 따로 둔다.

잠정 규칙(사전 v0.2 대비, 근거 없는 초기 설계)
- BM25 대체 쿼리: 사전 핵심어·확장어와 원문 문장 토큰이 모두 비면 확정 영역(main, sub 순)의 label_plain 토큰을 쓴다.
  banned·hold·삭제 영역·회피 채널 제외 규칙은 그대로 적용하고 접근 긍정어는 필요 내용으로 세지 않음.
  쓰면 query_mode.bm25는 'fallback', 사유는 'empty_need_tokens', 기록은 query_expansion.bm25_fallback.
- Dense 전용 확장: 확장 항목에 bm25_allowed: false와 dense_phrase가 있으면 add_terms는 BM25에 넣지 않고
  dense_phrase를 Dense 문장에 넣음(영역당 최대 1개, 부정 문장 매칭이면 제외). 기록은 query_expansion.dense_only_expansions.
- 교육훈련 확정 시 외출·접근 구절과 함께 빠진 학습 요청을 두 쿼리에 살리고, 과목의 동의 표기를 BM25에 추가한다.

확인 화면 메모(actions의 memo)는 읽지 않는다. 쿼리 재료는 original_text, 확정 영역 evidence, 욕구 질문 답뿐이다.
"""
from __future__ import annotations

import copy
import os
import re

import yaml

from core.utils import normalize
from query.communication import communication_tokens
from query.preservation import physical_context, request_trace
from extract.postprocess import (
    ACCESS_CHANNEL_WORDS, ELIDED_ACCESS_BURDEN, EXTRA_ACCESS_BURDEN,
    access_clauses, mask_negated_cues, negates_cue, sentences as split_sentences,
)
from retrieve.benefit_context import DECLINED, EDU_MODES, REQUEST, SUBJECT_ALIASES, request_clauses

ACCESS_MODES = ("A", "B1", "B2", "C")
DEFAULT_CAPS = {"main_core": 4, "sub_core": 2, "main_exp": 3, "sub_exp": 1, "access": 2}
CHANNEL_TOKENS = {"방문": ("방문", "대면"), "전화": ("전화", "통화")}
# block_if를 찾는 범위: 같은 절에서 패턴 바로 뒤 글자 수.
BLOCK_WINDOW = 12
# 부정 표현. Dense 원문 문장은 접근 채널어와 함께 있을 때만 뺀다("방문 없이", "전화 말고" 등).
# Dense 전용 확장 매칭은 채널어와 무관하게 부정 문장이면 제외('잠을 못 자요'로 상담 문장을 붙이지 않음)
NEGATION = re.compile(r"(없이|말고|말곤|못 ?|않|싫|(^|\s)안 ?(돼|되|가|나|해|하))")
# 회피 어휘 감지(잠정 휴리스틱). 채널어 + 부담어가 같은 절에 있으면 회피 절로 본다.
_CH_ANY = "(?:" + "|".join(ACCESS_CHANNEL_WORDS.values()) + ")"
_CH_RE = {ch: re.compile(v) for ch, v in ACCESS_CHANNEL_WORDS.items()}
_BURDEN = re.compile(r"부담|무섭|무서|어렵|어려|힘들|힘드|싫|불가(?:능)?|못|않|"
                     r"수(?:가)?\s*없|(?:^|\s)안\s*(?:돼|되|가|나|해|하|받|갈|할)|" + EXTRA_ACCESS_BURDEN)
# '전화 말고', '주민센터 직접 가지 말고', '전화보다', '전화 대신'은 그 구절만 뺀다
_INSTEAD = re.compile(_CH_ANY + r"[^,.;!?]{0,12}?(말고|말곤|보다는?|대신)\s*")
_PARTICLE = r"(?:은|는|이|가|을|를|도|으로|로|에서|에|와|과|만)?"
_REPORTED_ABILITY = re.compile(r"(?:받을\s*수|할\s*수|가능)\s*(?:있다고|하다고)\s*(?:들었|하던|하더|했)")
_EDUCATION_DETAIL = re.compile(r"자격증|자격시험|검정고시|공부|학업|수강|배워|배울|배우(?:고|는|려)")
TERMS_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs", "query_terms.yaml")


def load_terms(path: str = TERMS_FILE) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _default_tokenize():
    from retrieve.bm25 import tokenize  # kiwipiepy 필요. 모듈 최상단에서 import하지 않는다

    return tokenize


def query_cfg(cfg: dict) -> dict:
    q = (cfg.get("stage2") or {}).get("query") or {}
    return {"access_mode": q.get("access_mode", "B1"), "caps": {**DEFAULT_CAPS, **(q.get("caps") or {})}}


# ── 문장 위치 ─────────────────────────────────────────────────────
class TextMap:
    """원문 문장과 정규화 위치. evidence가 덮는 문장 번호를 찾는다."""

    def __init__(self, text: str):
        self.sents = [s for s in split_sentences(text, split_commas=False) if re.search(r"[가-힣A-Za-z0-9]", s)]
        self.norm = normalize(text)
        self.spans, cur = [], 0
        for s in self.sents:
            ns = normalize(s)
            pos = self.norm.find(ns, cur)
            self.spans.append((pos, pos + len(ns)) if pos >= 0 and ns else None)
            if pos >= 0:
                cur = pos + len(ns)

    def locate(self, evidence: str | None):
        """evidence가 겹치는 문장 번호 집합. 못 찾으면 None."""
        ne = normalize(evidence or "")
        pos = self.norm.find(ne) if ne else -1
        if pos < 0:
            return None
        lo, hi = pos, pos + len(ne)
        return {i for i, sp in enumerate(self.spans) if sp and sp[0] < hi and lo < sp[1]}

    def is_whole(self, idx: set) -> bool:
        return bool(self.sents) and idx >= set(range(len(self.sents)))


# ── 회피 어휘 감지 ────────────────────────────────────────────────
def _channels_in(text: str) -> set:
    return {ch for ch, r in _CH_RE.items() if r.search(text or "")}


def strip_avoid(sent: str) -> tuple[str, list[str], set]:
    """회피 표현과 제약을 부정한 접근 절을 빼고, 실제 회피를 감지한 채널만 반환한다."""
    out, cut, chans = sent, [], set()
    for m in list(_INSTEAD.finditer(out)):
        cut.append(m.group(0).strip())
        chans |= _channels_in(m.group(0))
    out = _INSTEAD.sub("", out)
    clauses = access_clauses(out)
    drop = set()
    previous = set()
    for i, c in enumerate(clauses):
        channels = _channels_in(c)
        masked = mask_negated_cues(c, _BURDEN)
        if channels and (masked != c or _BURDEN.search(c)):
            drop.add(i)
            if _BURDEN.search(masked):
                chans |= channels
        elif not channels and previous and ELIDED_ACCESS_BURDEN.match(masked):
            drop |= {i - 1, i}  # '전화는 못 하는 건 아닌데 되도록 피하고 싶어요'
            chans |= previous
        previous = channels
    for i in sorted(drop):
        cut.append(clauses[i].strip())
    out = " ".join(c.strip() for i, c in enumerate(clauses) if i not in drop).strip()
    if cut and len(re.findall(r"[가-힣A-Za-z0-9]", out)) < 3:
        out = ""
    return (out if cut else sent), cut, chans


def _phrase_re(phrase: str):
    chars = [re.escape(c) for c in normalize(phrase)]
    return re.compile(r"[^가-힣A-Za-z0-9]*".join(chars)) if chars else None


def _need_remainder(sent: str, need_evs, phrases) -> str:
    """제약·활동 구절을 뺀 문장의 욕구 내용. 문장 안 욕구 evidence가 있으면 그것을, 없으면 구절을 뺀 나머지를 쓴다.
    욕구 evidence를 품은 구절은 빼지 않는다. 회피 절은 strip_avoid로 한 번 더 뺀다. 남은 글자가 3자 미만이면 빈 문자열."""
    ns = normalize(sent)
    pn = {normalize(ph) for ph in phrases}
    inside = [e for e in need_evs if normalize(e) and normalize(e) in ns and normalize(e) not in pn]  # 구절과 같은 evidence는 욕구 구분 불가
    if not inside:
        # 욕구 evidence가 없으면 글자 단위로 잘라 깨진 조각을 남기지 않도록 절 단위로 뺀다.
        def touches(c, ph):
            nc, np_ = normalize(c), normalize(ph)
            r = _phrase_re(ph)
            return bool(nc and np_) and (nc in np_ or np_ in nc or bool(r and r.search(c)))
        out = " ".join(c for c in access_clauses(sent) if not any(touches(c, ph) for ph in phrases))
        out = strip_avoid(out)[0] if out else ""
        return out if len(re.findall(r"[가-힣A-Za-z0-9]", out)) >= 3 else ""
    out = inside[0]
    for ph in phrases:
        if any(normalize(e) in normalize(ph) and normalize(e) != normalize(ph) for e in inside):
            continue
        r = _phrase_re(ph)
        if r:
            out = r.sub(" ", out)
    out = re.sub(r"\s{2,}", " ", out).strip(" ,.;")
    out = strip_avoid(out)[0] if out else ""
    return out if len(re.findall(r"[가-힣A-Za-z0-9]", out)) >= 3 else ""


def strip_channel_words(sent: str, channels) -> str:
    """Dense 문장에서 회피 채널어(와 붙은 조사)를 뺀다."""
    for ch in sorted(channels):
        sent = re.sub(ACCESS_CHANNEL_WORDS[ch] + _PARTICLE + r"\s*", "", sent)
    return re.sub(r"\s{2,}", " ", sent).strip()


# ── 접근 의도(want) ───────────────────────────────────────────────
def _hit(text: str, patterns, block_if) -> str | None:
    blockers = re.compile("|".join(map(re.escape, block_if))) if block_if else None
    for p in patterns or []:
        for m in re.finditer(re.escape(p), text or ""):
            if blockers is None:
                return p
            after = re.split(r"[.!?;,]", text[m.end():], maxsplit=1)[0]
            after = next(iter(access_clauses(after)), "")
            if not any(b.start() < BLOCK_WINDOW and not negates_cue(after[b.end():]) for b in blockers.finditer(after)):
                return p
    return None


def detect_want(texts: list[tuple[str, str]], terms: dict) -> list[dict]:
    """texts = [(문장, source)]. 패턴 + want_markers가 같은 문장에 있고 block_if가 패턴 뒤에 없을 때."""
    markers = terms.get("want_markers") or []
    found = []
    for text, source in texts:
        marker_text = _REPORTED_ABILITY.sub("", text)
        if not any(mk in marker_text for mk in markers):
            continue
        for a in terms.get("access_terms") or []:
            if a.get("polarity") != "want":
                continue
            p = _hit(text, a.get("patterns"), a.get("block_if"))
            if p:
                found.append({"id": a["id"], "scope": a.get("scope"), "pattern": p, "text": text, "source": source})
    return found


def _education_request_text(text, *, is_answer=False):
    for clause in request_clauses(text):
        if DECLINED.search(clause) or clause.rstrip().endswith("말고"):
            continue
        clause = strip_avoid(clause)[0].strip()
        if not (clause and (is_answer or REQUEST.search(clause))) or _channels_in(clause):
            continue
        if _EDUCATION_DETAIL.search(clause) or any(pattern.search(clause) for pattern, _ in EDU_MODES.values()):
            yield clause


def _access_entry(a: dict, scope_src: dict) -> dict:
    return {"polarity": "want", "channel": None, "level": None, "scope": a.get("scope"),
            "mode": (a.get("positive_terms") or [None])[0], "pattern_id": a["id"], "source": scope_src["source"]}


# ── 메인 ─────────────────────────────────────────────────────────
def build_queries(session: dict, terms: dict, cfg: dict, tokenize=None, access_mode: str | None = None) -> dict:
    tokenize = tokenize or _default_tokenize()
    qc = query_cfg(cfg)
    mode = access_mode or qc["access_mode"]
    if mode not in ACCESS_MODES:
        raise ValueError(f"access_mode: {mode}")
    caps = qc["caps"]
    ext = session["extracted"]
    prof = session["confirmed_profile"]
    needs = prof["needs"]
    if not needs:
        raise ValueError("검색 사용 영역이 없음 (explore_only)")
    tdom = terms.get("domains") or {}
    tm = TextMap(session["original_text"])
    reasons_bm25, reasons_dense, removed = [], [], []

    # 1) 접근 의도 문장: 원문 문장 + 욕구 질문 자유 답
    answers = [q["answer"] for q in session.get("questions", [])
               if q.get("slot") == "need" and q.get("answer_type") == "자유" and q.get("answer")]
    wants = detect_want([(s, "original_text") for s in tm.sents] + [(a, "question") for a in answers], terms)
    want_idx = {i for i, s in enumerate(tm.sents) if any(w["text"] == s and w["source"] == "original_text" for w in wants)}

    # 2) 빼야 할 문장
    def drop(evidence, reason):
        idx = tm.locate(evidence)
        if idx is None:
            reasons_bm25.append(f"evidence_not_found:{reason}")
            return set()
        if tm.is_whole(idx):
            reasons_bm25.append(f"evidence_is_whole_text:{reason}")
        removed.extend({"sentence": tm.sents[i], "reason": reason} for i in sorted(idx))
        return idx

    drop_idx = set()
    cons_ev = {c.get("evidence") for c in ext.get("constraints", []) + prof.get("constraints", []) if c.get("evidence")}
    cons_idx, cons_whole = set(), False
    for ev in sorted(cons_ev):
        idx = tm.locate(ev)
        if idx is None:
            reasons_bm25.append("evidence_not_found:constraint")
            continue
        cons_whole = cons_whole or tm.is_whole(idx)
        cons_idx |= idx
    activity_ev = ext.get("activity_evidence")
    activity_idx = drop(activity_ev, "activity") if activity_ev else set()
    for i in sorted(want_idx):
        removed.append({"sentence": tm.sents[i], "reason": "access_want"})

    changes = session.get("need_changes") or []
    active = {n["domain"] for n in needs}
    kept_ev_idx = set()
    for n in needs:
        if n.get("evidence"):
            kept_ev_idx |= tm.locate(n["evidence"]) or set()
    for c in changes:
        if not c.get("evidence"):
            continue
        idx = drop(c["evidence"], "changed_need")
        if idx & kept_ev_idx:
            reasons_bm25.append(f"changed_evidence_overlaps:{c['domain']}")
        drop_idx |= idx
    if changes:
        reasons_dense.append("need_changed")

    # 2-1) 회피 접근 어휘 감지: evidence 유무와 관계없이 원문에서 직접 본다
    cons_removed = [{"sentence": tm.sents[i], "reason": "constraint"} for i in sorted(cons_idx)]
    lex_channels, lex_clauses, cleaned, avoid_removed = set(), [], {}, []
    activity_restored = set()
    for i, sent in enumerate(tm.sents):
        text_i, cut, chans = strip_avoid(sent)
        lex_channels |= chans
        if i in drop_idx or not cut:
            continue
        lex_clauses.append({"sentence": sent, "removed": cut})
        if i in activity_idx:
            # 외출 제약과 같은 절이 활동 근거인 경우, 이미 그 절을 제거했으므로 나머지를 살린다.
            if not text_i or not any(normalize(activity_ev) in normalize(c) for c in cut):
                continue
            activity_restored.add(i)
        if text_i:
            cleaned[i] = text_i
            avoid_removed.extend({"sentence": c, "reason": "avoid_clause"} for c in cut)
        elif i not in cons_idx | want_idx:
            drop_idx.add(i)
            avoid_removed.append({"sentence": sent, "reason": "avoid_clause"})
    # 2-2) 제약·활동 evidence가 문장을 덮어도 같은 문장의 욕구 내용은 살린다: 문장 전체 대신 제약·활동 구절만 뺀다.
    phrases = [e for e in (list(cons_ev) + ([activity_ev] if activity_ev else [])) if e]
    need_evs = [n["evidence"].strip("'\" ") for n in needs if n.get("evidence")]
    gone_all = set(range(len(tm.sents))) <= (cons_idx | activity_idx | want_idx | drop_idx) - cleaned.keys()
    restore = cons_whole or "evidence_is_whole_text:activity" in reasons_bm25 or gone_all  # template가 될 때
    # 활동 문장·욕구 evidence 문장은 항상 구절만 뺀다(같은 문장의 상황 서술은 검색 단서).
    # 제약만 있는 문장은 남은 조각에 접근 방식 말이 섞이기 쉬워 template가 될 때만 살린다.
    targets = (cons_idx | activity_idx) - cleaned.keys() - want_idx - drop_idx
    keep = targets if restore else targets & ((activity_idx - cons_idx) | kept_ev_idx)
    for i in sorted(keep):
        rest = _need_remainder(tm.sents[i], need_evs, phrases)
        if rest:
            cleaned[i] = rest
            avoid_removed.append({"sentence": tm.sents[i], "reason": "constraint_activity_phrase_only"})
    drop_idx |= (cons_idx | want_idx | activity_idx) - cleaned.keys()
    if cons_whole and cons_idx - cleaned.keys():
        reasons_bm25.append("evidence_is_whole_text:constraint")
    if activity_idx and activity_idx <= cleaned.keys() and "evidence_is_whole_text:activity" in reasons_bm25:
        reasons_bm25.remove("evidence_is_whole_text:activity")
    restored_sents = {tm.sents[i] for i in cleaned}
    removed = [r for r in cons_removed + removed if r["sentence"] not in restored_sents] + avoid_removed

    kept = [cleaned.get(i, s) for i, s in enumerate(tm.sents) if i not in drop_idx]
    bm25_mode = "template" if reasons_bm25 else "text"
    if bm25_mode == "text" and not kept:
        bm25_mode, reasons_bm25 = "template", ["no_text_left"]
    source_text = " ".join(kept) if bm25_mode == "text" else ""

    # Situation information is diagnostic context, not an inferred request for
    # bathing, meals or medical care. Keep both production search texts intact.
    context_only = physical_context(session, _channels_in)

    education_details, education_aliases = [], []
    education = next((n for n in needs if n["domain"] == "교육훈련"), None)
    if education and not changes:
        clauses = []
        evidence = education.get("evidence")
        education_idx = tm.locate(evidence)
        if education_idx:
            excluded_phrases = sorted(cons_ev | ({activity_ev} if activity_ev else set()),
                                      key=lambda ev: (-len(ev), ev))
            for text in [evidence] + [tm.sents[i] for i in sorted(education_idx)]:
                for ev in excluded_phrases:
                    text = text.replace(ev, " ")
                clauses.extend(_education_request_text(text))
                if clauses:
                    break
        for answer in answers:
            clauses.extend(_education_request_text(answer, is_answer=True))
        present = normalize(source_text)
        for clause in dict.fromkeys(clauses):
            if normalize(clause) not in present:
                education_details.append(clause)
            for pattern, aliases in SUBJECT_ALIASES:
                if pattern.search(clause):
                    education_aliases.extend(word for word in aliases if word not in education_aliases)

    # 3) 강제 제외: 삭제·변경 전 영역(다시 고르지 않은 것)의 핵심어·확장어
    gone = {c["domain"] for c in changes} - active
    forced_terms = []
    for d in sorted(gone):
        forced_terms += (tdom.get(d) or {}).get("core", [])
        forced_terms += [t for e in terms.get("expansions") or [] if e["domain"] == d for t in e.get("add_terms", [])]
    forced_tokens = {t for w in forced_terms for t in tokenize(w)}

    banned = set(terms.get("banned_terms") or [])
    hold = set(terms.get("hold_terms") or [])
    avoid_channels = {a["channel"] for a in session.get("access", []) if a.get("polarity") == "avoid"}
    channel_tokens = {t for ch in avoid_channels | lex_channels for t in CHANNEL_TOKENS.get(ch, ())}
    excluded = banned | forced_tokens | channel_tokens

    def dict_tokens(words):
        """사전 단어의 토큰. banned·hold는 뺀다. 영역명과 겹치는 핵심어(취업, 주거 등)는 그대로 둔다
        (영역명 문자열을 기계적으로 붙이지 않을 뿐, 마스터 근거로 고른 핵심어는 쓴다)."""
        out = []
        for w in words:
            out += [t for t in tokenize(w) if t not in banned | hold and t not in out]
        return out

    def dict_words(words, cap):
        """상한은 사전 단어(표기) 개수 기준."""
        return [w for w in words if dict_tokens([w])][:cap]

    # 4) 확장 (확정 영역 안에서만)
    main = next(n for n in needs if n["level"] == "main")
    subs = [n for n in needs if n["level"] != "main"]
    want_texts = [w["text"] for w in wants]
    applied = []
    dense_only = []

    def expansions_for(n, cap):
        texts = [clean for t in [n.get("evidence")] + answers + want_texts for s in split_sentences(t, split_commas=False)
                 if (clean := strip_avoid(s)[0])]
        words = []
        for e in terms.get("expansions") or []:
            if e["domain"] != n["domain"] or e.get("apply_when", "domain_confirmed") != "domain_confirmed":
                continue
            if e.get("bm25_allowed", True) is False:
                # Dense 전용: BM25 토큰에 넣지 않는다. 영역당 1개, 부정 문장 매칭은 제외(Dense 문장 규칙과 동일)
                phrase = e.get("dense_phrase")
                if (phrase
                        and not any(d["domain"] == n["domain"] for d in dense_only)
                        and phrase not in [d["phrase"] for d in dense_only]):
                    pos = [t for t in texts if not NEGATION.search(t)]
                    hit = next(((t, p) for t in pos for p in [_hit(t, e.get("patterns"), e.get("block_if"))] if p), None)
                    if hit:
                        dense_only.append({"id": e["id"], "domain": n["domain"], "phrase": phrase})
                continue
            hit = next(((t, p) for t in texts for p in [_hit(t, e.get("patterns"), e.get("block_if"))] if p), None)
            if hit:
                applied.append({"id": e["id"], "domain": n["domain"], "pattern": hit[1]})
                words += [w for w in e.get("add_terms", []) if w not in words]
        return dict_words(words, cap)

    core_words = dict_words((tdom.get(main["domain"]) or {}).get("core", []), caps["main_core"])
    for s in subs:
        core_words += dict_words((tdom.get(s["domain"]) or {}).get("core", []), caps["sub_core"])
    exp_words = expansions_for(main, caps["main_exp"])
    for s in subs:
        exp_words += expansions_for(s, caps["sub_exp"])
    core, exp = dict_tokens(core_words), dict_tokens(exp_words)
    education_tokens = dict_tokens(education_aliases)

    # 5) 접근 긍정어
    by_id = {a["id"]: a for a in terms.get("access_terms") or []}
    want_ids = []
    for w in sorted(wants, key=lambda w: 0 if w["scope"] == "이용" else 1):
        if w["id"] not in want_ids:
            want_ids.append(w["id"])
    derived_ids = [i for ch in sorted(avoid_channels) for i in (terms.get("avoid_derived") or {}).get(ch, [])
                   if i not in want_ids]
    # B1 = 사용자가 직접 요청한 접근(requested). B2는 회피 제약에서 추론한 접근(inferred)을 따로 더한다.
    requested_ids = want_ids if mode in ("B1", "B2") else []
    inferred_ids = derived_ids if mode == "B2" else []
    use_ids = requested_ids + inferred_ids

    def access_tokens(ids, taken):
        toks = []
        for i in ids:
            a = by_id[i]
            if a.get("bm25_allowed"):
                toks += [t for w in a.get("positive_terms", []) for t in dict_tokens([w])
                         if t not in toks and t not in taken]
        return toks[: caps["access"]]

    # 6) BM25 조립 — 현재 BM25Index.search(문자열) 인터페이스와 호환되게 임시 연결
    # - 원문은 문장 그대로 넣는다(재토큰화 때문에 원문 단어를 빼지 않음).
    #   제외 규칙(banned, 삭제 영역 단어, 회피 채널 토큰)에 걸린 문장만 그 문장 토큰에서 해당 토큰을 빼고 잇는다.
    # - 사전 토큰(명사 위주)은 원문 앞에 두고 ". "으로 구분한다. 원문과 겹치는 사전 토큰은 넣지 않는다.
    # - bm25_tokens(의도한 토큰 목록)와 bm25_search_tokens(검색 시 다시 나눈 결과)를 모두 남긴다.
    #   토큰 목록을 직접 받는 검색 함수는 3번 담당 협의 사항.
    def text_segment(sents):
        parts, fallback = [], []
        for s in sents:
            toks = tokenize(s)
            if set(toks) & excluded:
                parts.append(" ".join(t for t in toks if t not in excluded))
                fallback.append({"sentence": s, "excluded": sorted(set(toks) & excluded)})
            else:
                parts.append(s)
        return " ".join(p for p in parts if p), fallback

    # 대체 토큰(잠정): 확정 영역 label_plain. 사전 토큰과 같은 제외 규칙
    fb_domains, fb_tokens = [], []
    for n in [main] + subs:
        lab = (tdom.get(n["domain"]) or {}).get("label_plain")
        new = [t for t in dict_tokens([lab]) if t not in excluded and t not in fb_tokens] if lab else []
        if new:
            fb_domains.append(n["domain"])
            fb_tokens += new

    def assemble(sents, use_fb=False):
        text_str, fallback = text_segment(sents)
        text_tok = tokenize(text_str) if text_str else []
        dict_tok = []
        for t in (fb_tokens if use_fb else core + exp + education_tokens):
            if t not in text_tok and t not in dict_tok and t not in excluded:
                dict_tok.append(t)
        req = access_tokens(requested_ids, set(text_tok) | set(dict_tok) | excluded)
        inf = access_tokens(inferred_ids, set(text_tok) | set(dict_tok) | set(req) | excluded)
        segs = [" ".join(dict_tok), text_str, " ".join(req + inf)]
        query = ". ".join(s for s in segs if s)
        intended = []
        for t in dict_tok + text_tok + req + inf:
            if t not in intended:
                intended.append(t)
        return query, intended, req, inf, fallback

    def has_need(sents):
        text_str, _ = text_segment(sents)
        text_tok = tokenize(text_str) if text_str else []
        return bool(text_tok) or any(t not in excluded for t in core + exp + education_tokens)

    fb_used = False

    def assemble_fb(sents):
        nonlocal fb_used
        fb_used = False
        if not has_need(sents) and fb_tokens:
            fb_used = True
            return assemble(sents, True)
        return assemble(sents)

    bm25_sents = (kept if bm25_mode == "text" else []) + education_details
    bm25_query, bm25_tokens, acc_req, acc_inf, fallback = assemble_fb(bm25_sents)
    search_tokens = tokenize(bm25_query)
    if set(search_tokens) & excluded:
        reasons_bm25.append("forced_exclusion_remains")
        bm25_mode, source_text = "template", ""
        bm25_sents = []
        bm25_query, bm25_tokens, acc_req, acc_inf, fallback = assemble_fb(bm25_sents)
        search_tokens = tokenize(bm25_query)
    acc_tok = acc_req + acc_inf
    if not fb_used and not has_need(bm25_sents):
        fb_empty = True  # 대체 토큰도 없음: 그대로 두고 기록
    else:
        fb_empty = False
    bm25_template = bm25_mode == "template"   # 대체 쿼리로 표시가 바뀌어도 Dense 판정은 원래 전환 여부를 따름
    if fb_used or fb_empty:
        bm25_mode = "fallback" if fb_used else bm25_mode
        reasons_bm25.append("empty_need_tokens")
    # C 실험에서 검증한 BM25 전용 보완. Dense와 원문 삭제 규칙은 유지한다.
    # 확인된 관계 욕구 + 명시적인 관계 공간 희망이 함께 있어야 하며,
    # 삭제·변경 욕구와 기존 제외/보류 토큰 정책은 우선한다.
    communication_added = [] if changes else [
        t for t in communication_tokens(search_tokens, session['original_text'], active, tokenize)
        if t not in excluded | hold
    ]
    if communication_added:
        bm25_query = '. '.join(s for s in (bm25_query, ' '.join(communication_added)) if s)
        bm25_tokens += [t for t in communication_added if t not in bm25_tokens]
        search_tokens = tokenize(bm25_query)
    dups = sorted({t for t in search_tokens if search_tokens.count(t) > 1})
    interface = {"match": search_tokens == bm25_tokens,
                 "missing": [t for t in bm25_tokens if t not in search_tokens],
                 "extra": sorted(set(search_tokens) - set(bm25_tokens)),
                 "search_duplicates": dups, "sentence_token_fallback": fallback}
    dropped = sorted({t for f in fallback for t in f["excluded"]})

    # 7) Dense
    dense_mode = "template" if (reasons_dense or bm25_template or fb_used) else "text"
    labels = [(tdom.get(n["domain"]) or {}).get("label_plain", n["domain"]) for n in [main] + subs]
    dense = f"{labels[0]}에 대한 도움을 찾고 있습니다."
    if labels[1:]:
        dense += f" {', '.join(labels[1:])}도 필요합니다."
    if dense_mode == "text":
        # 접근 채널어와 부정어가 함께 있는 문장('방문 없이', '전화는 못 해요')만 뺀다.
        # '가스비를 못 냈어', '월급을 못 받았어요'처럼 어려움을 말하는 부정 문장은 욕구 내용이라 유지
        pos_sents = [strip_channel_words(s, avoid_channels | lex_channels) for s in kept
                     if not (NEGATION.search(s) and _channels_in(s))]
        pos_sents = [s for s in pos_sents if s]
        if pos_sents:
            dense += " " + " ".join(pos_sents)
    if education_details:
        dense += " " + " ".join(education_details)
    for d in dense_only:
        dense += " " + d["phrase"]
    # 긍정 문장 1개: 사용자가 요청한 접근을 먼저, 없으면(B2만) 추론한 접근
    phrase, phrase_source = None, None
    for ids, src in ((requested_ids, "requested"), (inferred_ids, "inferred")):
        phrase = next((by_id[i]["dense_phrase"] for i in ids if by_id[i].get("dense_phrase")), None)
        if phrase:
            phrase_source = src
            dense += " " + phrase
            break
    forced_in_dense = [w for w in forced_terms if w in dense]
    if forced_in_dense:
        reasons_dense.append("forced_exclusion_in_dense")

    # 8) access: 회피(세션) + 사용자가 요청한 접근(want). 모든 변형에서 채운다.
    #    B2에서 추론한 접근은 access에 넣지 않고 access_inferred로 따로 둔다(사용자 요청 정보와 구분).
    access = copy.deepcopy([a for a in session.get("access", []) if a.get("polarity") == "avoid"])
    seen = set()
    for w in wants:
        if w["id"] not in seen:
            seen.add(w["id"])
            access.append(_access_entry(by_id[w["id"]], w))
    access_inferred = [{"polarity": "inferred", "channel": None, "level": None, "scope": by_id[i].get("scope"),
                        "mode": (by_id[i].get("positive_terms") or [None])[0], "pattern_id": i,
                        "source": "avoid_derived"} for i in derived_ids]

    return {
        "bm25": bm25_query,
        "bm25_tokens": bm25_tokens,
        "bm25_search_tokens": search_tokens,
        "dense": dense,
        "query_mode": {"bm25": bm25_mode, "dense": dense_mode, "bm25_reasons": reasons_bm25,
                       "dense_reasons": reasons_dense},
        "access_mode": mode,
        "access": access,
        "access_inferred": access_inferred,
        "query_expansion": {
            "terms_version": terms.get("version"), "terms_status": terms.get("status"),
            "query_source_text": source_text, "removed_sentences": removed,
            "core_words": core_words, "expansion_words": exp_words,
            "core_tokens": core, "expansion_tokens": exp, "expansions_applied": applied,
            "communication_tokens_added": communication_added,
            "request_preservation": {"context_only": context_only,
                "trace": request_trace(session, source_text, search_tokens, dense, tokenize, normalize, excluded)},
            "education_request_text": education_details, "education_aliases": education_aliases,
            "access_tokens": acc_tok, "access_tokens_requested": acc_req, "access_tokens_inferred": acc_inf,
            "access_ids": use_ids, "access_requested_ids": requested_ids, "access_inferred_ids": inferred_ids,
            "bm25_fallback": {"used": fb_used, "source": "label_plain",
                              "domains": fb_domains if fb_used else [], "tokens": fb_tokens if fb_used else []},
            "dense_only_expansions": dense_only,
            "dense_phrase_source": phrase_source, "wants": wants, "want_rule": "잠정 휴리스틱",
            "lexical_avoid": {"channels": sorted(lex_channels), "clauses": lex_clauses},
            "dropped_tokens": dropped, "forced_exclusion_terms": forced_terms,
            "forced_in_dense": forced_in_dense, "caps": caps, "bm25_interface": interface,
        },
    }
