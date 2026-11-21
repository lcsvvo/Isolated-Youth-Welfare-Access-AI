"""SSIS 중앙·지자체 복지서비스를 서비스 1행의 services_master로 만든다.

입력: data/raw/ssis/{central,local}_{list,detail_full}.csv (읽기 전용)
출력: data/processed/services_master.csv

라벨 컬럼 <속성>마다 <속성>_evidence(근거 원문 문장), <속성>_method(field/rule/llm/human),
<속성>_review(검수 필요)를 둔다. 이 단계는 field와 rule만 채운다.
값 표기: 숫자 라벨은 문자열 '0'~'3', 근거 없음은 'N', 해당 없음은 'NA'.
age_min/age_max는 한 쌍으로 age_* 근거 컬럼을 공유하며, 'NA'는 그쪽 경계가 없음(예: 65세 이상의 상한)을 뜻한다.

실행: python -m src.preprocess.ssis_master
"""
from __future__ import annotations

import html
import json
import re

import pandas as pd

from pathlib import Path

from ..collect._common import PROCESSED, RAW, ROOT
from . import keywords as K
from .review_overrides import TO_N

SSIS = RAW / "ssis"
COLLECTED_AT = "2026-10-30"
EXPECTED_ROWS = 5288
YOUTH_MIN, YOUTH_MAX = 19, 34
N, NA = "N", "NA"

NESTED = {
    "inqplCtadrList": "contact",
    "inqplHmpgReldList": "homepage",
    "baslawList": "law",
    "basfrmList": "form",
}
NESTED_FIELDS = {
    "central": ("servSeCode", "servSeDetailNm", "servSeDetailLink"),
    "local": ("wlfareInfoDtlCd", "wlfareInfoReldNm", "wlfareInfoReldCn"),
}
TEXT_RULE_ATTRS = ["apply_barrier", "use_barrier", "use_outing", "repeat_burden", "duration", "cost", "contact_burden", "job_relevance"]
CODE_FILES = ["src/preprocess/ssis_master.py", "src/preprocess/keywords.py", "src/preprocess/review_overrides.py",
              "src/collect/_common.py"]
GS_COLS = ["gs_A_summary", "gs_B_benefit", "gs_C_external", "gs_C_source", "gs_C_year", "gs_C_region",
           "gs_D_target_group", "gs_final_status"]


# ------------------------------------------------------------------ 로딩·병합
def read_raw(name: str) -> pd.DataFrame:
    return pd.read_csv(SSIS / f"{name}.csv", dtype=str, encoding="utf-8-sig", keep_default_na=False, na_values=[""])


def load_merged() -> dict[str, pd.DataFrame]:
    """목록·상세를 servId로 1:1 병합. 상세에만 있는 컬럼과 목록의 겹치는 컬럼은 _det 접미사로 남긴다."""
    out = {}
    for src in ("central", "local"):
        lst, det = read_raw(f"{src}_list"), read_raw(f"{src}_detail_full")
        m = lst.merge(det, on="servId", how="outer", suffixes=("", "_det"), validate="one_to_one", indicator=True)
        if (m["_merge"] != "both").any():
            raise ValueError(f"{src}: 목록·상세 1:1 불일치 {m['_merge'].value_counts().to_dict()}")
        out[src] = m.drop(columns="_merge")
    return out


# ------------------------------------------------------------------ 정제
def clean_text(s) -> str | None:
    if s is None or (isinstance(s, float) and pd.isna(s)) or s is pd.NA:
        return None
    s = html.unescape(str(s)).replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"[ \t ]+", " ", s)
    s = re.sub(r"\n\s*\n+", "\n", s).strip()
    return s or None


def norm_tags(s) -> str | None:
    s = clean_text(s)
    if not s:
        return None
    return ", ".join(t for t in re.split(r"\s*,\s*", s) if t)


def _code(v) -> str:
    v = str(v)
    return f"{int(float(v)):03d}" if re.fullmatch(r"\d+(\.0+)?", v) else v


def nested_items(row: pd.Series, group: str, src: str) -> list[dict]:
    """평탄화 컬럼(1개)과 JSON 컬럼(2개 이상)을 합친 [{code,name,content}]."""
    fc, fn, fv = NESTED_FIELDS[src]
    items = []
    raw = row.get(group)
    if isinstance(raw, str) and raw.strip():
        for it in json.loads(raw):
            items.append({"code": _code(it.get(fc)), "name": clean_text(it.get(fn)), "content": clean_text(it.get(fv))})
    flat = {f: row.get(f"{group}_{f}") for f in (fc, fn, fv)}
    if any(isinstance(v, str) and v for v in flat.values()):
        items.append({"code": _code(flat[fc]) if isinstance(flat[fc], str) else None,
                      "name": clean_text(flat[fn]), "content": clean_text(flat[fv])})
    return items


def apply_items(raw) -> str | None:
    """중앙 applmetList에서 '신청기관연락처목록' 항목만, 중복 제거 후 줄바꿈으로 잇는다."""
    if not isinstance(raw, str) or not raw.strip():
        return None
    seen = []
    for it in json.loads(raw):
        if it.get("servSeDetailNm") == "신청기관연락처목록":
            t = clean_text(it.get("servSeDetailLink"))
            if t and t not in seen:
                seen.append(t)
    return "\n".join(seen) or None


def build_base(merged: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for src, m in merged.items():
        c = src == "central"
        b = pd.DataFrame({
            "servId": m["servId"],
            "source": src,
            "servNm": m["servNm"].map(clean_text),
            "org": m["jurMnofNm"] if c else m["bizChrDeptNm"],
            "org_dept": m["jurOrgNm"] if c else None,
            "region_sido": None if c else m["ctpvNm"],
            "region_sgg": None if c else m["sggNm"],
            "detail_url": m["servDtlLink"],
            "collected_at": COLLECTED_AT,
            "last_mod": m["crtrYr_det" if "crtrYr_det" in m else "crtrYr"] if c else m["lastModYmd"],
            "enfc_end_ymd": None if c else m["enfcEndYmd"],
            # (B) 원문 *_raw 원본
            "summary_text_raw": m["servDgst"],
            "target_text_raw": m["tgtrDtlCn"] if c else m["sprtTrgtCn"],
            "criteria_text_raw": m["slctCritCn"],
            "benefit_text_raw": m["alwServCn"],
            "apply_text_raw": m["applmetList"] if c else m["aplyMtdCn"],
            "overview_text_raw": m["wlfareInfoOutlCn"] if c else None,
            "apply_cat": None if c else m["aplyMtdNm"].map(norm_tags),
            "online_bokjiro": m["onapPsbltYn"] if c else None,
            "provision_type": m["srvPvsnNm"].map(norm_tags),        # 목록 기준 (중앙 상세는 45건 결측)
            "support_cycle": m["sprtCycNm"],
            # 태그 기준 파일: 목록 (중앙 관심주제는 목록 결측 12 < 상세 56)
            "life_tags": m["lifeArray" if c else "lifeNmArray"].map(norm_tags),
            "target_tags": m["trgterIndvdlArray" if c else "trgterIndvdlNmArray"].map(norm_tags),
            "theme_tags": m["intrsThemaArray" if c else "intrsThemaNmArray"].map(norm_tags),
            "inq_num": m["inqNum"],
        })
        for col in ("summary_text", "target_text", "criteria_text", "benefit_text", "overview_text"):
            b[col] = b[f"{col}_raw"].map(clean_text)
        b["apply_text"] = m["applmetList"].map(apply_items) if c else b["apply_text_raw"].map(clean_text)
        for group, name in NESTED.items():
            b[f"{name}_json"] = [json.dumps(nested_items(r, group, src), ensure_ascii=False) for _, r in m.iterrows()]
        rows.append(b)
    df = pd.concat(rows, ignore_index=True)
    if len(df) != EXPECTED_ROWS or df["servId"].duplicated().any():
        raise ValueError(f"행 수 {len(df)} (기대 {EXPECTED_ROWS}), servId 중복 {int(df['servId'].duplicated().sum())}")
    return df


# ------------------------------------------------------------------ 근거 문장
_SPLIT = re.compile(r"\n+|(?<=[.!?。])\s+|(?=[■○●◦▶※□•]\s?)|(?<=\))\s*(?=-)|\s-\s")


def sentences(text: str | None) -> list[str]:
    if not isinstance(text, str) or not text:
        return []
    return [s.strip(" -·") for s in _SPLIT.split(text) if s and s.strip(" -·")]


def find(texts: list[str | None], include: str, exclude: str = "", mask: str = "") -> str | None:
    """include가 맞고 같은 문장에 exclude가 없는 첫 문장.
    mask가 있으면 판정 전에 그 표현만 가린다(근거로는 원문 문장을 그대로 돌려준다)."""
    inc = re.compile(include, re.I)
    exc = re.compile(exclude, re.I) if exclude else None
    msk = re.compile(mask, re.I) if mask else None
    for t in texts:
        for s in sentences(t):
            s2 = msk.sub("〔자격〕", s) if msk else s
            if inc.search(s2) and not (exc and exc.search(s2)):
                return s
    return None


def find_all(texts: list[str | None], rules: list[tuple[int, str]]) -> dict[int, str]:
    """값별로 처음 맞은 근거 문장."""
    hits = {}
    for val, pat in rules:
        if val in hits:
            continue
        ev = find(texts, pat)
        if ev:
            hits[val] = ev
    return hits


def _ptypes(s) -> set[str]:
    return set(re.split(r"\s*,\s*", s)) if isinstance(s, str) and s else set()


def _set(rec: dict, attr: str, val, ev=None, method=None, review=False) -> None:
    rec[attr] = val
    rec[f"{attr}_evidence"] = ev
    rec[f"{attr}_method"] = method
    rec[f"{attr}_review"] = bool(review)


# ------------------------------------------------------------------ (C) 자격
_AGE = r"(?<!\d)(\d{1,2})\s*세(?!대)"
_RANGE = re.compile(rf"(?:만\s*)?(?<!\d)(\d{{1,2}})\s*(?:세)?\s*(?:이상)?\s*~\s*(?:만\s*)?{_AGE}\s*(이하|미만|까지)?")
_RANGE2 = re.compile(rf"(?:만\s*)?{_AGE}\s*(이상|초과|부터)\s*,?\s*(?:만\s*)?{_AGE}\s*(이하|미만|까지)")
_ONE = re.compile(rf"(?:만\s*)?{_AGE}\s*(이상|초과|부터|이하|미만|까지|이전)")


def _age_context_ok(text: str, start: int, end: int, legacy: bool = False) -> bool:
    before = text[max(0, start - 8):start]
    after = text[end:end + 14]
    ok = not re.search(K.AGE_CTX_BEFORE, before) and not re.search(K.AGE_CTX_AFTER, after)
    if legacy or not ok:
        return ok
    # v3: 제외·예외 조항('18세 미만자 제외', '65세 미만이더라도')의 연령은 자격 범위로 쓰지 않는다
    return not re.search(K.AGE_EXCLUSION_AFTER, text[end:end + 16]) and not re.search(K.AGE_EXCLUSION_BEFORE, text[max(0, start - 12):start])


def extract_age(texts: list[str | None], legacy: bool = False) -> tuple[str, str, str | None, bool, list]:
    """(age_min, age_max, 근거 문장, review, 개별 범위). 정보 없으면 ('N','N',None,False,[]).
    legacy=True는 v1 합집합(열린 경계를 무시)으로 계산한다. age_*_v1 비교용."""
    found, evs = [], []
    for t in texts:
        if not isinstance(t, str) or not t:
            continue
        tt = re.sub(r"[∼～〜–—]", "~", t)
        tt = re.sub(r"(?<=\d)\s*-\s*(?=\d)", "~", tt)
        used: list[tuple[int, int]] = []
        for rx, kind in ((_RANGE, "r"), (_RANGE2, "r2"), (_ONE, "o")):
            for m in rx.finditer(tt):
                if any(a < m.end() and m.start() < b for a, b in used):
                    continue
                if not _age_context_ok(tt, m.start(), m.end(), legacy):
                    continue
                if kind == "o" and not legacy:
                    # 문장 전체가 아니라 연령 바로 뒤 짧은 구간만 본다(같은 문장의 다른 제외 조항으로 정상 연령을 잃지 않게)
                    win = re.split(r"[.;\n]", tt[m.end():m.end() + K.AGE_EXCLUSION_WINDOW])[0]
                    if re.search(K.AGE_EXCLUSION_SENTENCE, win):
                        continue
                if kind == "r":
                    lo, hi, op = int(m.group(1)), int(m.group(2)), m.group(3)
                    hi = hi - 1 if op == "미만" else hi
                elif kind == "r2":
                    lo, op1, hi, op2 = int(m.group(1)), m.group(2), int(m.group(3)), m.group(4)
                    lo = lo + 1 if op1 == "초과" else lo
                    hi = hi - 1 if op2 == "미만" else hi
                else:
                    n, op = int(m.group(1)), m.group(2)
                    lo, hi = (n + (op == "초과"), None) if op in ("이상", "초과", "부터") else (None, n - (op in ("미만", "이전")))
                if lo is not None and hi is not None and lo > hi:
                    continue
                used.append((m.start(), m.end()))
                found.append((lo, hi))
                evs.append(next((s for s in sentences(tt) if m.group(0) in s), m.group(0)))
    if not found:
        return N, N, None, False, []
    if legacy:
        # v1: 열린 경계를 무시한 최소·최대 (예: '18세 이상' + '18~20세' → 18~20)
        los = [lo for lo, _ in found if lo is not None]
        his = [hi for _, hi in found if hi is not None]
        amin = str(min(los)) if los else NA
        amax = str(max(his)) if his else NA
    else:
        # v3: 개별 범위 전체를 덮는 경계. 한 범위라도 그쪽이 열려 있으면 NA
        amin = NA if any(lo is None for lo, _ in found) else str(min(lo for lo, _ in found))
        amax = NA if any(hi is None for _, hi in found) else str(max(hi for _, hi in found))
    uniq_ev = list(dict.fromkeys(evs))
    return amin, amax, " | ".join(uniq_ev[:3]), len(set(found)) > 1, list(dict.fromkeys(found))


def overlaps_youth(intervals: list[tuple[int | None, int | None]]) -> bool:
    """개별 범위 중 하나라도 19~34세와 겹치면 True. age_min~age_max는 범위 사이의 빈 구간을 표현하지 못하므로
    (예: 6~18세 또는 70세 이상) 겹침은 개별 범위로 판정한다(v3)."""
    return any((lo if lo is not None else 0) <= YOUTH_MAX and (hi if hi is not None else 200) >= YOUTH_MIN
               for lo, hi in intervals)


def label_eligibility(r: pd.Series) -> dict:
    rec: dict = {}
    who = [r.servNm, r.summary_text, r.target_text, r.criteria_text]
    elig = [r.target_text, r.criteria_text]
    allt = [r.servNm, r.summary_text, r.target_text, r.criteria_text, r.benefit_text]

    # 청년 근거 플래그 3개
    rec["youth_tag"] = "청년" in _ptypes(r.life_tags)
    ev_txt = find(who, K.YOUTH_TEXT)
    rec["youth_text"] = ev_txt is not None
    rec["youth_text_evidence"] = ev_txt

    age_texts = [r.servNm, r.target_text, r.criteria_text, r.summary_text]
    amin, amax, aev, arev, intervals = extract_age(age_texts)
    rec["age_min"], rec["age_max"] = amin, amax
    # v1(합집합에서 열린 경계 무시) 값을 비교용으로 남긴다
    v1min, v1max, _, _, _ = extract_age(age_texts, legacy=True)
    rec["age_min_v1"], rec["age_max_v1"] = v1min, v1max
    if v1min == N:
        rec["elig_youth_v1"] = N
    else:
        lo1 = int(v1min) if v1min != NA else 0
        hi1 = int(v1max) if v1max != NA else 200
        rec["elig_youth_v1"] = "예" if lo1 <= YOUTH_MAX and hi1 >= YOUTH_MIN else "아니오"
    rec["age_evidence"], rec["age_method"] = aev, ("rule" if aev else None)
    rec["age_review"] = arev
    if amin == N:
        rec["youth_age"] = None
        _set(rec, "elig_youth", N, None, None, rec["youth_tag"] or rec["youth_text"])
    else:
        ov = overlaps_youth(intervals)
        rec["youth_age"] = ov
        conflict = (not ov) and (rec["youth_tag"] or rec["youth_text"])
        _set(rec, "elig_youth", "예" if ov else "아니오", aev, "rule", arev or conflict)

    ev_none = find(elig, K.INCOME_NONE)
    ev_yes = find(elig, K.INCOME_YES)
    if ev_none and ev_yes:
        _set(rec, "elig_income", "있음", f"{ev_yes} | {ev_none}", "rule", True)
    elif ev_yes:
        _set(rec, "elig_income", "있음", ev_yes, "rule")
    elif ev_none:
        _set(rec, "elig_income", "없음", ev_none, "rule")
    else:
        _set(rec, "elig_income", N)

    ev_iso = find(allt, K.ISOLATION)
    _set(rec, "isolation_explicit", "예" if ev_iso else "아니오", ev_iso, "rule")
    ev_fc = find(allt, K.FAMILY_CARE)
    if ev_iso and ev_fc:
        # 여러 집단이 함께 적힌 서비스: 급여를 집단별로 나눌 수 없으면 공통 + 검수
        _set(rec, "target_group", "공통", f"{ev_iso} | {ev_fc}", "rule", True)
    elif ev_iso:
        # 노인·1인가구의 '사회적 고립'도 걸리므로 청년 근거가 없으면 검수 대상
        youth_any = rec["youth_tag"] or rec["youth_text"] or bool(rec["youth_age"])
        _set(rec, "target_group", "고립·은둔", ev_iso, "rule", not youth_any)
    elif ev_fc:
        _set(rec, "target_group", "가족돌봄", ev_fc, "rule")
    else:
        _set(rec, "target_group", "불명")
    return rec


# ------------------------------------------------------------------ (D) 접근성
def label_access(r: pd.Series) -> dict:
    rec: dict = {}
    pt = _ptypes(r.provision_type)
    apply_txt = [r.apply_text]
    use_txt = [r.benefit_text, r.summary_text]

    # apply_barrier: 구조화 필드 → 원문 키워드. 쉬운 경로(작은 값)를 택한다.
    cands: list[tuple[int, str, str]] = []
    if r.online_bokjiro == "Y":
        cands.append((0, "[field] onapPsbltYn=Y (복지로 온라인 신청 가능)", "field"))
    for tok in _ptypes(r.apply_cat):
        if tok in K.APPLY_TOKEN:
            cands.append((K.APPLY_TOKEN[tok], f"[field] aplyMtdNm={r.apply_cat}", "field"))
    for val, ev in find_all(apply_txt, K.APPLY_TEXT).items():
        cands.append((val, ev, "rule"))
    if cands:
        val, ev, method = min(cands, key=lambda x: (x[0], x[2] != "field"))
        review = val == 0 and bool(re.search(r"신청\s*(불필요|없이)|별도|자동", ev))
        ev_int = find([r.apply_text, r.criteria_text], K.APPLY_INTERVIEW)
        if ev_int and val == 2:
            val, ev, method = 3, f"{ev} | {ev_int}", "rule"
        elif ev_int:
            review = True
        if any(v == 1 and re.search(r"추천|직권", e) for v, e, _ in cands if v == val):
            review = True
        _set(rec, "apply_barrier", str(val), ev, method, review)
    else:
        _set(rec, "apply_barrier", N)

    # use_barrier
    hits = find_all(use_txt, K.USE_TEXT)
    if pt and pt <= K.HOME_PROVISION:
        _set(rec, "use_barrier", "0", f"[field] srvPvsnNm={r.provision_type}", "field")
    elif hits:
        hi = max(hits)
        review = (0 in hits and hi > 0) or bool(pt & K.HOME_PROVISION)
        _set(rec, "use_barrier", str(hi), hits[hi], "rule", review)
    elif pt & K.HOME_PROVISION:
        _set(rec, "use_barrier", "0", f"[field] srvPvsnNm={r.provision_type}", "field", True)
    elif pt & {"현물지급", "현물대여"} and find(use_txt, r"택배|배송|가정으로|우편"):
        _set(rec, "use_barrier", "0", find(use_txt, r"택배|배송|가정으로|우편"), "rule")
    else:
        _set(rec, "use_barrier", N, None, None, "시설입소" in pt)

    # repeat_burden
    cash_only = bool(pt) and pt <= K.HOME_PROVISION
    texts = [r.benefit_text, r.summary_text]
    ev_reg = find(texts, K.REPEAT_REGULAR, K.REPEAT_EXCLUDE)
    ev_cnt = find(texts, K.REPEAT_COUNT + r"\s*(참여|상담|교육|방문|이상|과정|진행|운영|실시)", K.REPEAT_EXCLUDE)
    ev_wk = find(texts, K.REPEAT_WEEKS, K.REPEAT_EXCLUDE)
    ev_mo = find(texts, K.REPEAT_MONTHS, K.REPEAT_EXCLUDE)
    if cash_only:
        if r.support_cycle == "1회성":
            _set(rec, "repeat_burden", "0", f"[field] sprtCycNm=1회성, srvPvsnNm={r.provision_type}", "field")
        else:
            # 지급 주기는 참여 횟수가 아니므로 값을 정하지 않는다
            _set(rec, "repeat_burden", N, f"[field] sprtCycNm={r.support_cycle} (지급 주기)", None, True)
    elif ev_reg or ev_mo and int(re.search(K.REPEAT_MONTHS, ev_mo).group(1)) >= 2:
        _set(rec, "repeat_burden", "2", ev_reg or ev_mo, "rule")
    elif ev_cnt or ev_wk:
        n = int(re.search(K.REPEAT_COUNT, ev_cnt).group(1)) if ev_cnt else 0
        w = int(re.search(K.REPEAT_WEEKS, ev_wk).group(1)) if ev_wk else 0
        val = "0" if (ev_cnt and n <= 1 and not ev_wk) else ("2" if n > 8 or w > 6 else "1")
        _set(rec, "repeat_burden", val, ev_cnt or ev_wk, "rule", True)
    elif r.support_cycle == "1회성":
        _set(rec, "repeat_burden", "0", f"[field] sprtCycNm=1회성", "field", True)
    else:
        _set(rec, "repeat_burden", N)

    # remote_complete: 신청 비대면 = apply_barrier 0·1, 이용 비대면 = use_barrier 0
    a, u = rec["apply_barrier"], rec["use_barrier"]
    if N in (a, u):
        _set(rec, "remote_complete", N)
    else:
        v = int(a in ("0", "1")) + int(u == "0")
        _set(rec, "remote_complete", str(v), f"apply_barrier={a}, use_barrier={u}", "rule",
             rec["apply_barrier_review"] or rec["use_barrier_review"])
    return rec


# ------------------------------------------------------------------ (E) 추천 요소
APPLY_TO_OUT = {"0": 0, "1": 1, "2": 1, "3": 2}
USE_TO_OUT = {"0": 0, "1": 1, "2": 2, "3": 2}


def label_reco(r: pd.Series, acc: dict) -> dict:
    rec: dict = {}
    pt = _ptypes(r.provision_type)
    texts = [r.benefit_text, r.summary_text, r.criteria_text]

    a, u = acc["apply_barrier"], acc["use_barrier"]
    parts = [APPLY_TO_OUT[a]] if a != N else []
    parts += [USE_TO_OUT[u]] if u != N else []
    if parts:
        _set(rec, "outing_burden", str(max(parts)), f"apply_barrier={a}, use_barrier={u}", "rule",
             acc["apply_barrier_review"] or acc["use_barrier_review"] or len(parts) == 1)
    else:
        _set(rec, "outing_burden", N)

    _set(rec, "duration", acc["repeat_burden"], acc["repeat_burden_evidence"], acc["repeat_burden_method"],
         acc["repeat_burden_review"])

    hits = find_all(texts, K.COST_TEXT)
    if pt & K.LOAN_PROVISION:
        _set(rec, "cost", "2", f"[field] srvPvsnNm={r.provision_type} (상환 의무)", "field")
    elif pt and pt <= K.CASH_ONLY_PROVISION:
        _set(rec, "cost", NA, f"[field] srvPvsnNm={r.provision_type} (현금지급형)", "field")
    elif hits:
        v = max(hits)
        _set(rec, "cost", str(v), hits[v], "rule", len(hits) > 1)
    else:
        _set(rec, "cost", N)

    hits = find_all([r.benefit_text, r.summary_text], K.CONTACT_TEXT)
    if pt and pt <= K.HOME_PROVISION:
        _set(rec, "contact_burden", "0", f"[field] srvPvsnNm={r.provision_type}", "field")
    elif hits:
        v = max(hits)
        _set(rec, "contact_burden", str(v), hits[v], "rule", len(hits) > 1 or bool(pt & K.HOME_PROVISION))
    else:
        _set(rec, "contact_burden", N)

    hits = find_all([r.servNm, r.summary_text, r.benefit_text], K.JOB_TEXT)
    if hits:
        v = max(hits)
        _set(rec, "job_relevance", str(v), hits[v], "rule", len(hits) > 1)
    else:
        # 0(무관)은 근거 문장으로 확인할 수 없으므로 규칙으로 채우지 않는다
        _set(rec, "job_relevance", N)
    rec["job_relevance_signal"] = job_signal(r, rec["job_relevance"])
    return rec


def job_signal(r: pd.Series, value: str) -> str:
    """job_relevance N의 원인 후보. 값이 아니라 사람 검수·LLM 처리 순서를 정하기 위한 신호다.
    keyword_present: 넓은 직업 어휘가 있으나 규칙이 판정하지 못함 (규칙 누락 후보)
    text_no_keyword: 직업 어휘 없음, 지원내용 50자 이상 (실제 무관 후보, 0은 근거 문장 필요)
    short_text     : 직업 어휘 없음, 지원내용 50자 미만 (원문 정보 부족 후보)
    """
    if value != N:
        return "rule_labeled"
    text = " ".join(t for t in (r.servNm, r.summary_text, r.benefit_text) if isinstance(t, str))
    if re.search(K.JOB_BROAD, text):
        return "keyword_present"
    blen = len(r.benefit_text) if isinstance(r.benefit_text, str) else 0
    return "text_no_keyword" if blen >= K.JOB_SIGNAL_MIN_LEN else "short_text"


# ------------------------------------------------------------------ (F) 영역
def label_area(r: pd.Series) -> dict:
    texts = [r.servNm, r.summary_text, r.benefit_text]
    areas, evs, methods = [], [], []
    for code, rule in K.NEED_AREA_RULES.items():
        ev = find(texts, rule["include"], rule["exclude"])
        if ev:
            areas.append(code)
            evs.append(f"{code}: {ev}")
            methods.append("rule")
    pt = _ptypes(r.provision_type)
    if "q21_10" not in areas and pt & K.ECONOMIC_PROVISION:
        areas.append("q21_10")
        evs.append(f"q21_10: [field] srvPvsnNm={r.provision_type}")
        methods.append("field")
    areas.sort()
    rec = {
        "need_area_q21": "|".join(areas) if areas else N,
        "need_area_q21_evidence": " || ".join(evs) or None,
        "need_area_q21_method": ",".join(sorted(set(methods))) or None,
        # 원문 키워드 후보는 정밀도 미검증이므로 검수 대상, 제공유형 필드만으로 붙은 경우는 제외
        "need_area_q21_review": "rule" in methods,
    }

    dom, dev = [], []
    for code in areas:
        for d in K.APP_DOMAIN_MAP.get(code, []):
            dom.append(d)
            dev.append(f"{d}<-{code}")
    if "q21_10" in areas:
        ev_lab = find(texts, K.LABOR_PATTERN)
        d = "노동" if ev_lab else "경제(생활비)"
        dom.append(d)
        dev.append(f"{d}<-q21_10" + (f" ({ev_lab})" if ev_lab else ""))
    if {"q21_06", "q21_07"} & set(areas):
        ev_fut = find(texts, K.FUTURE_PATTERN)
        if ev_fut:
            dom.append("사회인식 및 미래설계")
            dev.append(f"사회인식 및 미래설계<-q21_06/07 ({ev_fut})")
    dom = [d for d in K.APP_DOMAINS if d in dom]
    rec.update({
        "app_domain": "|".join(dom) if dom else N,
        "app_domain_evidence": " || ".join(dev) or None,
        "app_domain_method": "rule" if dom else None,
        # 대응표가 확정되지 않았으므로(팀 확인 필요) 후보 값은 모두 검수 대상
        "app_domain_review": bool(dom),
    })
    return rec


# ------------------------------------------------------------------ v2 개정 (2026-11-02)
# apply_barrier·remote_complete·app_domain은 개정하고 개정 전 값을 *_v1로 남긴다.
# use_barrier·outing_burden은 그대로 두고, 새 개념은 use_outing·outing_need로 따로 만든다.
V1_ATTRS = ["apply_barrier", "remote_complete", "app_domain"]


def label_apply_v2(r: pd.Series) -> dict:
    """신청 장벽 v2. 중앙은 '신청기관연락처목록' 항목(apply_text), 지자체는 aplyMtdNm + aplyMtdCn.
    apply_interview: 없음 / 현장 / 비대면 / 불명(면접 형태 미기재, 신청 쪽 외출을 추정하지 않는다)."""
    rec: dict = {"apply_interview": "없음", "apply_channel": N}
    cands: list[tuple[int, str, str]] = []
    if r.online_bokjiro == "Y":
        cands.append((0, "[field] onapPsbltYn=Y (복지로 온라인 신청 가능)", "field"))
    for tok in _ptypes(r.apply_cat):
        if tok in K.APPLY_V2_TOKEN:
            cands.append((K.APPLY_V2_TOKEN[tok], f"[field] aplyMtdNm={r.apply_cat}", "field"))
    for val, pat in K.APPLY_V2_TEXT:
        ev = find([r.apply_text], pat, K.APPLY_V2_NEGATION)
        if ev:
            cands.append((val, ev, "rule"))
    ev_list = find([r.apply_text], K.APPLY_V2_LISTING, K.APPLY_V2_NEGATION + "|문의")
    for val, words in (K.APPLY_V2_LISTING_CHANNELS if ev_list else []):
        if re.search(words, ev_list, re.I):
            cands.append((val, ev_list, "rule"))
    if not cands:
        # 신청 정보가 없거나('직접 지원'만 있는 경우 포함) 신청 가능 문장으로 확인되지 않음
        _set(rec, "apply_barrier", N, None, None, isinstance(r.apply_text, str))
        return rec
    val, ev, method = min(cands, key=lambda x: (x[0], x[2] != "field"))
    if (val == 2 and method == "field" and find([r.apply_text], K.APPLY_V2_AGENCY)
            and not find([r.apply_text], K.APPLY_V2_USER_VISIT)):
        # 기관 주도 선정이고 사용자 방문이 원문에 없음: 분류값 '방문'만으로 2를 확정하지 않는다
        _set(rec, "apply_barrier", N, None, None, True)
        return rec
    review = method == "rule" and val == 0 and (bool(re.search(K.APPLY_V2_NO_APPLY, ev)) or bool(re.search(K.APPLY_V2_PARTIAL, ev)))
    req = find([r.apply_text], K.APPLY_VISIT_REQUIRED, K.APPLY_VISIT_REQUIRED_EXCLUDE)
    if req and val < 2:
        # 온라인 접수 후 방문 제출처럼 모든 단계를 거쳐야 하면 2 (Notion). 경로 최솟값 0·1을 올리고 검수
        val, ev, method, review = 2, (ev if req == ev else f"{ev} | {req}"), "rule", True
    rec["apply_channel"] = str(val)
    ev_int = find([r.apply_text], K.APPLY_V2_INTERVIEW_REQUIRED, K.APPLY_V2_INTERVIEW_EXCLUDE)
    ev_extra = find([r.apply_text], K.APPLY_V2_EXTRA, K.APPLY_V2_NEGATION)
    if ev_int:
        rec["apply_interview"] = "비대면" if re.search(K.APPLY_V2_INTERVIEW_REMOTE, ev_int) else (
            "현장" if re.search(r"현장|대면|방문", ev_int) else "불명")
        val, ev, method, review = 3, f"{ev} | {ev_int}", "rule", review or rec["apply_interview"] == "불명"
    elif ev_extra:
        rec["apply_interview"] = "현장"
        val, ev, method = 3, f"{ev} | {ev_extra}", "rule"
    elif find([r.apply_text], K.APPLY_V2_INTERVIEW):
        review = True        # 면접 언급은 있으나 필수 절차로 확인되지 않음
    if find([r.apply_text], K.APPLY_V2_AGENCY):
        review = True        # 추천·직권 등 기관 주도 선정 (경계 사례)
    _set(rec, "apply_barrier", str(val), ev, method, review)
    return rec


def label_use_outing(r: pd.Series) -> dict:
    """서비스 이용 외출 조건(0~3/N). 현금형 0, 이용권형 1(field). 그 밖에는 원문 키워드 중 가장 높은 값."""
    rec: dict = {}
    pt = _ptypes(r.provision_type)
    if "시설입소" in pt:
        _set(rec, "use_outing", N, None, None, True)    # 시설입소형은 0~3 척도로 잴 수 없다
        return rec
    field_val = 1 if pt & K.USE_OUTING_VOUCHER_PROVISION else (0 if pt & K.USE_OUTING_HOME_PROVISION else None)
    field_ev = f"[field] srvPvsnNm={r.provision_type}"
    if pt and pt <= K.USE_OUTING_HOME_PROVISION | K.USE_OUTING_VOUCHER_PROVISION:
        med = find([r.benefit_text, r.summary_text], K.USE_OUTING_MEDICAL_COST) if field_val == 0 else None
        if med:
            # 현금 수령은 집에서 가능하지만 진료 방문이 전제되므로 0/1 확정값으로 두지 않는다. 값은 사람 검수 전 바꾸지 않는다
            _set(rec, "use_outing", str(field_val), f"{field_ev} | [의료 이용 전제·확인 필요] {med}", "field", True)
        else:
            _set(rec, "use_outing", str(field_val), field_ev, "field")
        return rec
    hits = {}
    for v, inc, exc in K.USE_OUTING_TEXT:
        ev = find([r.benefit_text, r.summary_text], inc, exc)
        if ev and v not in hits:
            hits[v] = ev
    cands = [(v, e, "rule") for v, e in hits.items()] + ([(field_val, field_ev, "field")] if field_val is not None else [])
    if cands:
        levels = {c[0] for c in cands}
        val, ev, method = max(cands, key=lambda x: (x[0], x[2] == "field"))
        if len(levels) >= 2:
            choice = [c for c in cands if c[0] == 0 and c[2] == "rule" and re.search(K.USE_OUTING_CHOICE, c[1], re.I)]
            if choice:
                # 온라인·방문 중 고를 수 있으면 집에서 가능한 경로 0 (검수)
                val, ev, method = 0, "[선택 경로] " + choice[0][1], "rule"
            else:
                ev = "[혼재·확인 필요] " + ev
        _set(rec, "use_outing", str(val), ev, method, True)    # 현금·이용권 외 제공유형이 섞인 값은 모두 검수
    else:
        _set(rec, "use_outing", N)
    return rec


def outing_need(apply_v: str, use_v: str) -> tuple[str, str]:
    """(outing_need, outing_need_min). 신청 환산 0→0, 1→0, 2→1, 3→2. outing_need = max(신청 환산, use_outing).
    한쪽이 N이면 N이지만, 아는 쪽이 이미 최댓값(3)이면 3. outing_need_min = 확인된 값들의 최댓값(하한), 둘 다 N이면 N."""
    parts = ([K.APPLY_TO_OUTING_V2[apply_v]] if apply_v != N else []) + ([int(use_v)] if use_v != N else [])
    if not parts:
        return N, N
    low = max(parts)
    full = low if len(parts) == 2 or low == K.OUTING_NEED_MAX else None
    return (str(full) if full is not None else N), str(low)


def app_domain_v2(r: pd.Series, need_area: str, econ_candidate: bool = False) -> dict:
    """확정 대응만 app_domain에, 미확정 대응은 app_domain_pending에 둔다.
    ⑩ candidate(축하금·보상·용도 미상 현금 등)는 '경제(생활비)'를 확정하지 않고 pending에 둔다."""
    areas = set(need_area.split("|")) - {N}
    if econ_candidate and "q21_10" in areas:
        areas.discard("q21_10")
        extra_pend = ["경제(생활비)"]
    else:
        extra_pend = []
    texts = [r.servNm, r.summary_text, r.benefit_text]
    dom, pend, dev = [], [], []
    for q, d, status, cond, _ in K.APP_DOMAIN_MAP_V2:
        if q not in areas:
            continue
        ev_cond = find(texts, K.APP_DOMAIN_CONDITIONS[cond]) if cond else None
        if cond and not ev_cond:
            continue
        label = d or "(대응 없음)"
        (dom if status == "확정" else pend).append(label)
        dev.append(f"{label}<-{q}" + (f" ({ev_cond})" if ev_cond else "") + ("" if status == "확정" else " [미확정]"))
    dom = [d for d in K.APP_DOMAINS if d in dom]
    for d in extra_pend:
        pend.append(d)
        dev.append(f"{d}<-q21_10 [⑩ candidate, 미확정]")
    pend = list(dict.fromkeys(pend))
    return {"app_domain": "|".join(dom) if dom else N, "app_domain_evidence": " || ".join(dev) or None,
            "app_domain_method": "rule" if dom else None, "app_domain_review": bool(pend),
            "app_domain_pending": "|".join(pend) or None}


def app_domain_v3(r: pd.Series, need_area: str, econ_candidate: bool = False) -> dict:
    """핵심·부분 대응을 구분하고 ⑩ candidate만 미확정으로 남긴다."""
    areas = set(need_area.split("|"))
    pending = econ_candidate and "q21_10" in areas
    if pending:
        areas.remove("q21_10")

    texts = [r.servNm, r.summary_text, r.benefit_text]
    core, partial = set(), set()
    for q, domain, level, condition, _ in K.APP_DOMAIN_MAP_V3:
        if q not in areas:
            continue
        # 조건부 대응은 V2와 동일한 패턴·원문을 사용한다. 그 근거 문장은
        # 기존 app_domain_evidence에 대응 당시의 V2 영역 이름으로 남는다.
        if condition and not find(texts, K.APP_DOMAIN_CONDITIONS[condition]):
            continue
        (core if level == "핵심" else partial).add(domain)
    if K.APP_DOMAIN_V3_AUTO_ASSIGN_PARTIAL:
        core.update(partial)
    return {
        "app_domain": "|".join(d for d in K.APP_DOMAINS_V3 if d in core) or N,
        "app_domain_partial": "|".join(d for d in K.APP_DOMAINS_V3 if d in partial) or None,
        "app_domain_pending": "경제" if pending else None,
    }


def label_v2(r: pd.Series, rec: dict) -> dict:
    out = label_apply_v2(r)
    out.update(label_use_outing(r))
    for attr in TO_N.get(r.servId, {}):
        _set(out, attr, N, None, None, True)          # 검토 결과 근거 없는 값 (review_overrides.py)
    a, u_old, u = out["apply_barrier"], rec["use_barrier"], out["use_outing"]
    # 집에서 받음(use_outing 0) ≠ 본인 외출 없음 ≠ 비대면 완결을 구분한다.
    # 바우처·지역화폐(use_outing=1)는 use_barrier=0이어도 비대면 이용이 아니고, 가정 방문(use_outing=0)은 v1 use_barrier=1이라 비대면이 아니다.
    # 대리 신청(1)은 대리인이 방문하므로 비대면 신청으로 세지 않는다 (같은 문장에 다른 원격 경로가 있으면 제외)
    ev_a = out["apply_barrier_evidence"] or ""
    proxy_only = (a == "1" and out["apply_barrier_method"] == "rule" and "대리" in ev_a
                  and not re.search(r"전화|우편|팩스|이메일|e-?mail|메일", ev_a, re.I))
    iv, ch = out["apply_interview"], out["apply_channel"]
    if a in ("0", "1"):
        a_rem = not proxy_only
    elif a == "2":
        a_rem = False
    elif a == "3":
        a_rem = True if (iv == "비대면" and ch in ("0", "1")) else (False if iv == "현장" else None)
    else:
        a_rem = None
    if u_old == "0" and u == "0":
        u_rem = True
    elif u_old in ("1", "2", "3") or u in ("1", "2", "3"):
        u_rem = False
    else:
        u_rem = None
    if a_rem is None or u_rem is None:
        _set(out, "remote_complete", N)
    else:
        _set(out, "remote_complete", str(int(a_rem) + int(u_rem)),
             f"apply_barrier={a}{' (대리, 비대면 아님)' if proxy_only else ''}, use_barrier={u_old}, use_outing={u}",
             "rule", out["apply_barrier_review"] or rec["use_barrier_review"] or out["use_outing_review"])
    # 비대면 면접은 면접 전 신청 경로 값으로 환산하고, 면접 형태가 불명이면 신청 쪽 외출을 추정하지 않는다
    if out["apply_interview"] == "비대면":
        a_out = out["apply_channel"]
    elif out["apply_interview"] == "불명":
        a_out = N
    else:
        a_out = a
    full, low = outing_need(a_out, u)
    if out["apply_interview"] == "불명" and out["apply_channel"] != N:
        low = str(max(int(low) if low != N else 0, K.APPLY_TO_OUTING_V2[out["apply_channel"]]))
    _set(out, "outing_need", full, f"apply_barrier={a}(외출 환산 {K.APPLY_TO_OUTING_V2.get(a_out, N)}, 면접 {out['apply_interview']}), use_outing={u}" if full != N else None,
         "rule" if full != N else None, out["apply_barrier_review"] or out["use_outing_review"])
    out["outing_need_min"] = low
    out.update(app_domain_v2(r, rec["need_area_q21"], rec.get("q21_10_tier") == "candidate"))
    return out


# ------------------------------------------------------------------ v3 보완 (2026-11-04)
V3_ATTRS = ["need_area_q21"]
K_YOUTH_MIN = YOUTH_MIN          # 개정 전 값을 *_v1로 남긴다
V3_NEW_COLS = ["need_area_q21_v1", "q21_10_basis", "age_min_v1", "age_max_v1", "elig_youth_v1",
               "youth_scope", "youth_scope_basis", "youth_scope_evidence", "youth_scope_method", "youth_scope_review",
               "target_condition", "target_condition_evidence", "target_condition_method", "target_condition_review"]


def label_area_v3(r: pd.Series) -> dict:
    """need_area_q21 v3: v1 규칙과 같되 자격 조건 언급('의료급여 수급자' 등)을 가린 뒤 판정한다.
    q21_10_basis: ⑩ 경제적 지원의 근거 종류(text = 원문 키워드, field_only = 제공유형 필드만, 빈칸 = ⑩ 아님)."""
    texts = [r.servNm, r.summary_text, r.benefit_text]
    areas, evs, methods = [], [], []
    for code, rule in K.NEED_AREA_RULES.items():
        ev = find(texts, rule["include"], rule["exclude"], K.ELIG_MENTION_MASK)
        if ev:
            areas.append(code)
            evs.append(f"{code}: {ev}")
            methods.append("rule")
    basis = "text" if "q21_10" in areas else None
    pt = _ptypes(r.provision_type)
    if "q21_10" not in areas and pt & K.ECONOMIC_PROVISION:
        areas.append("q21_10")
        evs.append(f"q21_10: [field] srvPvsnNm={r.provision_type}")
        methods.append("field")
        basis = "field_only"
    areas.sort()
    return {
        "need_area_q21": "|".join(areas) if areas else N,
        "need_area_q21_evidence": " || ".join(evs) or None,
        "need_area_q21_method": ",".join(sorted(set(methods))) or None,
        "need_area_q21_review": "rule" in methods,
        "q21_10_basis": basis,
    }


def target_conditions(r: pd.Series) -> dict:
    """특정 집단 자격 조건 표시(다중). 제외 기준이 아니라 사용자 확인이 필요한 조건 목록이다."""
    texts = [r.servNm, r.target_text, r.criteria_text]
    hits, evs = [], []
    for name, pat in K.TARGET_CONDITIONS.items():
        ev = find(texts, pat)
        if ev:
            hits.append(name)
            evs.append(f"{name}: {ev}")
    return {"target_condition": "|".join(hits) or None, "target_condition_evidence": " || ".join(evs) or None,
            "target_condition_method": "rule" if hits else None, "target_condition_review": bool(hits)}


def youth_scope(r: pd.Series, rec: dict) -> dict:
    """서비스 쪽 청년 범위 A/B/U/C와 근거 종류(youth_scope_basis).
    사용자 개인의 자격 충족은 여기서 판정하지 않는다(src/recommend/eligibility.py)."""
    named = find([r.servNm], K.YOUTH_TEXT, K.YOUTH_TEXT_NEG)
    in_target = find([r.target_text], K.YOUTH_TEXT, K.YOUTH_TEXT_NEG)
    mention = rec["youth_text_evidence"] if rec["youth_text"] else None
    tag, ov = rec["youth_tag"], rec["youth_age"]
    amin, amax, aev = rec["age_min"], rec["age_max"], rec["age_evidence"]
    band = (ov is True and amin not in (N, NA) and amax not in (N, NA)
            and K.YOUTH_BAND_LO[0] <= int(amin) <= K.YOUTH_BAND_LO[1] and K.YOUTH_BAND_HI[0] <= int(amax) <= K.YOUTH_BAND_HI[1])
    band = band and not find([r.servNm, r.target_text], K.AGE_BAND_MIXED)   # 2026-11-05 품질점검
    ctx_texts = [r.servNm, r.summary_text, r.target_text, r.criteria_text]
    below = amax not in (N, NA) and int(amax) < K_YOUTH_MIN
    above = amin not in (N, NA) and int(amin) > YOUTH_MAX
    dependent = ((below and bool(find(ctx_texts, K.AGE_DEPENDENT_CTX)))
                 or (above and bool(find(ctx_texts, K.AGE_ELDER_DEPENDENT_CTX))))
    household = bool(find(ctx_texts, K.AGE_HOUSEHOLD_CTX))
    alt_group = find([r.target_text, r.criteria_text], K.AGE_ALT_GROUP)
    life = _ptypes(r.life_tags)
    if ov is False and (named or in_target or mention or tag):
        val, basis, ev, review = "U", "conflict", " | ".join(x for x in (aev, named or in_target or mention, "[field] lifeArray=청년" if tag else None) if x), True
    elif named:
        val, basis, ev, review = "A", "name_text", named, rec["age_review"]
    elif band:
        val, basis, ev, review = "A", "age_band", aev, rec["age_review"]
    elif ov is True:
        val, basis, ev, review = "B", "age_overlap", aev, rec["age_review"]
    elif in_target:
        # 지원대상 문장에만 '청년'이 있으면 여러 집단 중 하나로 나열된 경우가 많아 특화(A)로 보지 않는다
        val, basis, ev, review = "B", "target_text_mention", in_target, True
    elif tag:
        val, basis, ev, review = "B", "tag_only", f"[field] lifeArray={r.life_tags}", True
    elif mention:
        val, basis, ev, review = "B", "text_mention", mention, True
    elif ov is False and dependent:
        val, basis, ev, review = "U", "age_may_refer_to_dependent", aev, True
    elif ov is False and alt_group:
        # '65세 이상 또는 장애인'처럼 연령 밖 신청 경로가 함께 적힘: 청년 비해당으로 확정하지 않는다
        val, basis, ev, review = "U", "age_partial", f"{aev} | {alt_group}", True
    elif ov is False:
        val, basis, ev, review = "C", "age_excludes", aev, rec["age_review"] or household
    elif life:
        # 생애주기 태그는 있으나 청년이 없음: 비청년 대상일 가능성이 높지만 연령 근거가 없어 U로 둔다
        val, basis, ev, review = "U", "tag_non_youth", f"[field] lifeArray={r.life_tags}", False
    else:
        val, basis, ev, review = "U", "no_evidence", None, False
    out: dict = {"youth_scope_basis": basis}
    method = None if basis == "no_evidence" else ("field" if basis in ("tag_only", "tag_non_youth") else "rule")
    _set(out, "youth_scope", val, ev, method, review)
    return out


# ------------------------------------------------------------------ v3 외출 환산 (2026-11-05)
OUTING_V3_COLS = ["apply_outing", "apply_outing_basis", "outing_need_v2", "outing_need_min_v2"]


def apply_outing(r: pd.Series, out: dict) -> tuple[str, str, str | None]:
    """신청 쪽 외출 (값 0/2/N, 근거 수준, 근거). 잠정 라벨이다: visit_only_listed를 '방문 필수'로 해석하지 않는다.
    visit_required_text : 원문에 방문 필수 또는 '온라인 신청 후 방문 제출' 같은 순차 절차 → 2 (신청 장벽이 0·1이어도)
    interview_onsite    : 현장 면접 → 2 / interview_unknown: 면접 형태 불명 → N
    remote_route        : 온라인·전화·우편·대리 경로 확인 → 0
    visit_only_listed   : 방문 경로만 안내됨(다른 경로 미확인, 대부분 신청방법 분류값 '방문') → 2 + 검수
    no_apply_info       : 신청 정보 없음 → N"""
    a, ch, iv = out["apply_barrier"], out["apply_channel"], out["apply_interview"]
    req = find([r.apply_text], K.APPLY_VISIT_REQUIRED, K.APPLY_VISIT_REQUIRED_EXCLUDE)
    if a == N:
        if req and isinstance(r.apply_text, str):
            return "2", "visit_required_text", req
        return N, "no_apply_info", None
    if req:
        return "2", "visit_required_text", req
    if iv == "현장":
        return "2", "interview_onsite", out["apply_barrier_evidence"]
    if iv == "불명":
        return N, "interview_unknown", out["apply_barrier_evidence"]
    path = ch if iv == "비대면" else a
    if path == N:
        return N, "no_apply_info", None
    v = K.APPLY_OUTING_FROM_CHANNEL[path]
    return str(v), ("remote_route" if v == 0 else "visit_only_listed"), out["apply_barrier_evidence"]


def outing_v3(r: pd.Series, out: dict) -> dict:
    """외출필요도 v3 = max(apply_outing, use_outing). 한쪽이 N이면 N(아는 쪽이 3이면 3). v2 값은 *_v2로 남긴다."""
    rec = {"outing_need_v2": out["outing_need"], "outing_need_min_v2": out["outing_need_min"]}
    av, basis, aev = apply_outing(r, out)
    # 별도 근거 컬럼은 두지 않는다: 신청 근거는 apply_barrier_evidence, 방문 필수 문장만 outing_need_evidence에 붙인다
    rec.update({"apply_outing": av, "apply_outing_basis": basis})
    u = out["use_outing"]
    parts = ([int(av)] if av != N else []) + ([int(u)] if u != N else [])
    if out["apply_interview"] == "불명" and out["apply_channel"] in K.APPLY_OUTING_FROM_CHANNEL:
        low_parts = parts + [K.APPLY_OUTING_FROM_CHANNEL[out["apply_channel"]]]
    else:
        low_parts = parts
    low = str(max(low_parts)) if low_parts else N
    full = str(max(parts)) if parts and (len(parts) == 2 or max(parts) == K.OUTING_NEED_MAX) else N
    review = out["use_outing_review"] or basis in ("visit_only_listed", "interview_unknown") or out["apply_barrier_review"]
    ev = f"apply_outing={av}({basis}), use_outing={u}" + (f" | {aev}" if basis == "visit_required_text" else "")
    _set(rec, "outing_need", full, ev if full != N else None,
         "rule" if full != N else None, review)
    rec["outing_need_min"] = low
    return rec


# ------------------------------------------------------------------ ⑩ 경제적 지원 v4 (2026-11-05, 잠정)
ECON_V4_COLS = ["need_area_q21_v3", "q21_10_basis_v3", "q21_10_type", "q21_10_tier", "q21_10_evidence",
                "q21_07_beneficiary", "q21_07_beneficiary_evidence"]


def edu_beneficiary(r: pd.Series, areas: str) -> dict:
    """⑦ 교육 지원의 수혜자: dependent(자녀·아동·초중고생) / self(청년 본인·성인) / both / N. ⑦이 아니면 빈칸."""
    if "q21_07" not in areas.split("|"):
        return {"q21_07_beneficiary": None, "q21_07_beneficiary_evidence": None}
    texts = [r.servNm, r.target_text, r.criteria_text, r.summary_text, r.benefit_text]
    # 서비스명에 한쪽만 있으면 서비스명을 따른다(본문의 '만 35세 이상 한부모', '본인' 같은 신청인 조건을 수혜자로 보지 않기 위해)
    dep_n = find([r.servNm], K.EDU_DEPENDENT, "", K.ELIG_MENTION_MASK)
    slf_n = find([r.servNm], K.EDU_SELF, "", K.ELIG_MENTION_MASK)
    if bool(dep_n) != bool(slf_n):
        dep, slf = dep_n, slf_n
    else:
        dep = find(texts, K.EDU_DEPENDENT, "", K.ELIG_MENTION_MASK)
        slf = find(texts, K.EDU_SELF, "", K.ELIG_MENTION_MASK)
    if not dep and not slf:
        dep = find(texts, K.EDU_CHILD_GENERIC, "", K.ELIG_MENTION_MASK)
    val = "both" if dep and slf else "dependent" if dep else "self" if slf else N
    ev = " || ".join(x for x in (f"dependent: {dep}" if dep else "", f"self: {slf}" if slf else "") if x) or None
    return {"q21_07_beneficiary": val, "q21_07_beneficiary_evidence": ev}


def _econ_method(rec: dict, typ: str, tier, areas: set) -> str | None:
    """⑩ 재분류 뒤의 method. 제공유형만으로 붙은 ⑩(form_only)만 field, 나머지 영역은 원문 규칙."""
    if not areas:
        return None
    methods = set()
    if areas - {"q21_10"}:
        methods.add("rule")
    if "q21_10" in areas:
        methods.add("field" if typ == "form_only" else "rule")
    return ",".join(sorted(methods))


def econ_v4(r: pd.Series, rec: dict) -> dict:
    """⑩ 재분류. 지급 방식이 아니라 지원 목적으로 판단한다(자격 조건 언급은 가린다).
    q21_10_type: living / expense_relief / asset / lump_sum / compensation / unrestricted_cash / form_only /
                 earmarked(용도가 다른 영역) / org_target(기관·사업자 대상) / none
    q21_10_tier: main(Q1 본 분석 ⑩) / candidate(민감도 분석에서만 포함) / excluded / 빈칸(⑩ 신호 없음)
    need_area_q21에는 main·candidate를 ⑩으로 넣고, 용도가 교육·주거로 정해진 지급은 그 영역을 붙인다."""
    texts = [r.servNm, r.summary_text, r.benefit_text]
    mask = K.ELIG_MENTION_MASK + "|" + K.ECON_ELIG_MASK
    f = lambda pat: find(texts, pat, "", mask)
    areas = set(rec["need_area_q21"].split("|")) - {N, "q21_10"}
    pt = _ptypes(r.provision_type)
    name_only = lambda pat: find([r.servNm], pat, "", mask)
    earmark = {k: ev for k, pat in K.ECON_EARMARK.items() if (ev := f(pat))}
    name_earmark = {k: ev for k, pat in K.ECON_EARMARK.items() if (ev := name_only(pat))}
    # ⑩ 후보 신호: v3에서 ⑩이었거나(원문 키워드·제공유형) 제공유형이 현금·융자·지역화폐·감면. 생계 명시가 아니면 새로 ⑩을 만들지 않는다
    signal = rec["q21_10_basis"] is not None or bool(pt & K.ECONOMIC_PROVISION)
    nb = [r.servNm, r.benefit_text]
    # 판정 순서: 서비스명의 목적(생계 → 필수비용 → 보상 → 일시금 → 기관 대상) → 본문 생계 명시 → 서비스명의 다른 용도 → 기관 대상 → 본문
    steps = [
        # 생계·생활비 명시: 신호 없어도 ⑩, 다른 용도와 함께면 다중 라벨.
        # 서비스명에 다른 용도가 있으면 요약문의 자격 문장을 피하려고 서비스명·지원내용에서만 찾는다
        ("living", lambda: name_only(K.ECON_LIVING), False),
        ("expense_relief", lambda: name_only(K.ECON_EXPENSE), True),
        ("compensation", lambda: name_only(K.ECON_COMP), True),
        ("lump_sum", lambda: name_only(K.ECON_LUMP), True),
        ("org_target", lambda: name_only(K.ECON_ORG_NAME), True),
        ("living", lambda: find(nb, K.ECON_LIVING_WITH_EARMARK, "", mask) if name_earmark else f(K.ECON_LIVING), False),
        ("earmarked", lambda: next((f"{k}: {v}" for k, v in name_earmark.items()), None), True),
        ("org_target", lambda: find(nb, K.ECON_ORG, "", mask), True),
        ("asset", lambda: f(K.ECON_ASSET), True),
        ("earmarked", lambda: next((f"{k}: {v}" for k, v in earmark.items()), None), True),
        ("expense_relief", lambda: f(K.ECON_EXPENSE), True),
        # 보상·예우는 서비스명에서만 본다(본문의 '국가유공자' 등은 대상 나열인 경우가 많다)
        ("lump_sum", lambda: f(K.ECON_LUMP), True),
        ("living", lambda: f(K.ECON_LIVING_WEAK), True),
        ("unrestricted_cash", lambda: f(K.ECON_CASH_WORD), True),
    ]
    typ, ev = ("none", None)
    for name, fn, need_signal in steps:
        if need_signal and not signal:
            continue
        if (hit := fn()):
            typ, ev = name, hit
            break
    else:
        if signal:
            typ, ev = "form_only", f"[field] srvPvsnNm={r.provision_type}"
    # 용도가 교육·주거·취업으로 정해진 지급은 그 영역을 붙인다(⑩ 신호가 있는 지급에만. 기관 대상은 제외)
    added = []
    if signal and typ != "org_target":
        for a in ("q21_11", "q21_07", "q21_08"):
            if a in earmark and a not in areas and (a in name_earmark or not (set(earmark) - {a})):
                areas.add(a)
                added.append(f"{a}: {earmark[a]}")
    tier = ("main" if typ in K.ECON_MAIN_TYPES else "candidate" if typ in K.ECON_CANDIDATE_TYPES
            else "excluded" if typ in ("earmarked", "org_target") else None)
    if tier in ("main", "candidate"):
        areas.add("q21_10")
    new_areas = "|".join(sorted(areas)) if areas else N
    evs = [e for e in (rec["need_area_q21_evidence"] or "").split(" || ") if e and not e.startswith("q21_10:")]
    if tier in ("main", "candidate"):
        evs.append(f"q21_10: {ev}")
    evs += added
    return {
        "need_area_q21_v3": rec["need_area_q21"], "q21_10_basis_v3": rec["q21_10_basis"],
        "need_area_q21": new_areas, "need_area_q21_evidence": " || ".join(evs) or None,
        "need_area_q21_review": bool(rec["need_area_q21_review"]) or tier == "candidate" or bool(added),
        "need_area_q21_method": _econ_method(rec, typ, tier, areas),
        "q21_10_type": typ, "q21_10_tier": tier, "q21_10_evidence": ev,
        **edu_beneficiary(r, new_areas),
        # q21_10_basis는 v4 ⑩ 기준으로 다시 맞춘다(원문 근거 / 제공유형만). v3 값은 q21_10_basis_v3
        "q21_10_basis": (None if tier not in ("main", "candidate") else "field_only" if typ == "form_only" else "text"),
    }


# ------------------------------------------------------------------ 조립
def build_master() -> pd.DataFrame:
    base = build_base(load_merged())
    recs = []
    for _, r in base.iterrows():
        rec = label_eligibility(r)
        acc = label_access(r)
        rec.update(acc)
        rec.update(label_reco(r, acc))
        rec.update(label_area(r))
        # 개정 전 값 보존 후 v2로 교체 (use_barrier·outing_burden·repeat_burden·cost·contact_burden·job_relevance는 v1 유지)
        for a in V1_ATTRS:
            rec[f"{a}_v1"] = rec[a]
        for a in V3_ATTRS:
            rec[f"{a}_v1"] = rec[a]
        rec.update(label_area_v3(r))
        rec.update(econ_v4(r, rec))            # ⑩ 목적 기준 재분류 (v3 영역은 need_area_q21_v3)
        rec.update(youth_scope(r, rec))
        rec.update(target_conditions(r))
        rec.update(label_v2(r, rec))
        rec["app_domain_v2"] = rec["app_domain"]
        rec.update(app_domain_v3(r, rec["need_area_q21"], rec["q21_10_tier"] == "candidate"))
        rec.update(outing_v3(r, rec))
        recs.append(rec)
    lab = pd.DataFrame(recs, index=base.index)
    # 원문 키워드로 정한 (D)(E) 값은 후보이므로 모두 검수 대상으로 둔다.
    # 구조화 필드([field])나 다른 라벨에서 계산한 값(apply_barrier=…)은 그 출처의 review를 따른다.
    for attr in TEXT_RULE_ATTRS:
        ev = lab[f"{attr}_evidence"].fillna("")
        text_rule = lab[f"{attr}_method"].eq("rule") & ~ev.str.startswith(("[field]", "apply_barrier="))
        lab.loc[text_rule, f"{attr}_review"] = True
    df = pd.concat([base, lab], axis=1)
    for c in GS_COLS:
        df[c] = None
    # 개정으로 추가한 컬럼은 기존 컬럼 순서를 바꾸지 않도록 맨 뒤에 둔다
    tail = V3_NEW_COLS + OUTING_V3_COLS + ECON_V4_COLS + ["app_domain_partial", "app_domain_v2"]
    return df[[c for c in df.columns if c not in tail] + tail]


def save_master(df: pd.DataFrame) -> Path:
    """services_master.csv와 메타(services_master.meta.json)를 함께 쓴다."""
    import hashlib
    import subprocess
    from datetime import datetime, timezone

    from .raw_manifest import current_hashes

    PROCESSED.mkdir(parents=True, exist_ok=True)
    out = PROCESSED / "services_master.csv"
    # 원문 *_raw에 단독 CR(\r)이 있다. 줄 끝을 CRLF로 고정해야 csv 모듈이 그 칸을 따옴표로 감싸므로
    # OS와 관계없이 같은 파일이 나온다(LF로 쓰면 Linux·macOS에서 행이 깨진다).
    df.to_csv(out, index=False, encoding="utf-8-sig", lineterminator="\r\n")
    git = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    meta = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "rows": len(df), "cols": df.shape[1], "columns": list(df.columns),
        "inputs_sha256": {k: v for k, v in current_hashes().items() if "/ssis/" in k},
        "code_commit": git("rev-parse", "--short", "HEAD"),
        # 마스터를 만드는 추적 파일에 커밋되지 않은 수정이 있는지만 본다(관련 없는 미추적 파일은 제외)
        "code_dirty": bool(git("status", "--porcelain", "--untracked-files=no", "--", *CODE_FILES)),
        "code_sha256": {f: hashlib.sha256((ROOT / f).read_bytes()).hexdigest() for f in CODE_FILES},
        "spec": "docs/services_master_spec.md",
    }
    (PROCESSED / "services_master.meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def main() -> None:
    df = build_master()
    out = save_master(df)
    print(f"services_master {len(df):,}행 × {df.shape[1]}열 → {out}")


if __name__ == "__main__":
    main()
