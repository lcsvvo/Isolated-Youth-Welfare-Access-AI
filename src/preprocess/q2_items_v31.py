"""Q2 정보 항목 명시 수준 규칙 v3.1 (코드북 v3.1 경계 결정 + v3 검증 미해결 이슈 반영, 검증 전 잠정).

q2_items_v3.py 는 고치지 않는다. v3 모듈의 정규식·함수를 가져와 v3.1 에서 달라지는 부분만 덮어쓰거나 보탠다.
덮어쓰기는 judge31()/run31() 안에서만 일시적으로 적용되고(patched), 끝나면 v3 모듈은 원래대로 돌아온다.
정의는 q2_codebook.py(RULES, RULES_31)가 정한다. 이 모듈은 그 정의를 규칙으로 옮긴 것이다.

출력: data/processed/q2_items_v3_1.csv (v3 열 + rule_version=3.1, core_metric), q2_items_v3_1.meta.json,
      logs/validation/q2_item_rules/v31_rule_changes.csv
실행: python -m src.preprocess.q2_items_v31
"""
from __future__ import annotations

import contextlib
import json
import re
from datetime import datetime, timezone

import pandas as pd

from ..collect._common import PROCESSED, ROOT
from . import q2_codebook as CB
from . import q2_items_v3 as V

RULE_VERSION = "3.1"
OUT = PROCESSED / "q2_items_v3_1.csv"
OUT_META = PROCESSED / "q2_items_v3_1.meta.json"
CHANGES = V.RULES_DIR / "v31_rule_changes.csv"
ITEMS = V.ITEMS
H, clauses, ev = V.H, V.clauses, V.ev

# ------------------------------------------------------------------ 신청 불필요 (v3 이슈: '신청을 통한 지원 아님')
NOT_NEEDED31 = re.compile(V.NOT_NEEDED.pattern + r"|신청(?:을|에)\s?(?:통한|의한)\s?지원\s?(?:이\s?)?아[닙님]")

# ------------------------------------------------------------------ 신청 행위 (코드북 3.1 c: 예약은 필수 접수일 때만)
_ACT_BASE = r"(?:신청|접수|제출|청구|신고)(?!\s?(?:문의|안내|상담|자격|대상|기간|기한|일시|불필요|없))"
RES_REQUIRED = (r"예약\s?(?:이|은)?\s?필수|필수\s?(?:로\s?)?(?:사전\s?)?예약|반드시\s?(?:사전\s?)?예약"
                r"|(?:사전\s?)?예약\s?(?:을\s?)?(?:한\s?)?(?:후|뒤|하고|하여|해서)[^\n.]{0,12}?(?:신청|방문|접수|제출)")
ACT31 = re.compile(_ACT_BASE + "|" + RES_REQUIRED)

# ------------------------------------------------------------------ 연령
AGE_DOT_BIRTH = re.compile(
    r"\d{4}\s?[.\-/년]\s?\d{1,2}\s?(?:월|[.\-/]\s?\d{1,2}\s?일?)?\s?\.?\s?(?:이전|이후|까지|부터)?\s?(?:에\s?)?(?:출생|출산|태어난)")
AGE_ELAPSED = re.compile(
    r"(?:출생|출산|생후)\s?(?:일|후|이후)?\s?(?:로부터|으로부터|부터|기준)?\s?[^\n.]{0,10}?\d+\s?(?:일|주|개월|년|세)\s?(?:이내|미만|이하|까지|내)")
ELAPSED_SKIP = re.compile(r"신청|접수|청구|지급|기한|제출|신고|기간|거주|주민등록|주소|근무|재직|가입")   # 신청·지급 기한, 거주·지원 기간은 연령 경계가 아니다
BIRTH_CAT31 = re.compile(r"(?<!저)출산(?!율|휴가|전후)")             # 신생아·영아는 v3 AGE_CAT 이 이미 부분
BIRTH_WORD = re.compile(r"영아|신생아|사산아")
DEATH_WORD = re.compile(r"사망|장례|화장|장제|유골")                  # 사망 영아 정의 구절은 신청자 연령이 아니다
LEGAL_EXCL = re.compile(r"동반|보호자|법정\s?대리인|동의")             # 이용 규칙·절차 속 미성년
PIECE = re.compile(r"(?<![\d.])(?=\d{1,2}\s?\.\s?[가-힣「])|(?<=[가-힣])-\s|\s-\s|\s\*\s")


def age_text(F) -> str:
    t = "\n".join(F.get(k, "") for k in ("target", "criteria"))
    for c in clauses(t):
        for p in PIECE.split(c):
            p = p.strip()
            if p and BIRTH_WORD.search(p) and DEATH_WORD.search(p):
                t = t.replace(p, " ")
    return t


def j_age31(F) -> list:
    t = age_text(F)
    out = []

    def add(label, rule, pat):
        h = V.first_hit(t, [(label, rule, pat)])
        if h:
            out.append(h)

    for label, rule, pat in V.AGE_RULES:
        if label != "명시":
            break
        if rule != "age_legal":
            add(label, rule, pat)
    add("명시", "age_birth_dot", AGE_DOT_BIRTH)
    for c in clauses(t):
        if not ELAPSED_SKIP.search(c):
            m = AGE_ELAPSED.search(c)
            if m:
                out.append(H("명시", "age_birth_elapsed", ev(c, m)))
                break
    for c in clauses(t):
        m = V.AGE_LEGAL.search(c)
        if m and not LEGAL_EXCL.search(c):
            out.append(H("명시", "age_legal", ev(c, m)))
            break
    add("부분", "age_category", V.AGE_CAT)
    add("부분", "age_birth_category", BIRTH_CAT31)
    add("부분", "age_mention", V.AGE_WORD)
    return out


# ------------------------------------------------------------------ 소득 (코드북 3.1 e: 우선 선발·차등지원은 자격 아님)
PRIORITY_AFTER = re.compile(r"^[^\n.,，;·ㆍ/]{0,12}?(?:우선\s?(?:선발|선정|지원|배정)|차등\s?(?:지원|지급|적용))")
PRIORITY_BEFORE = re.compile(r"우선\s?(?:선발|선정|지원|배정)\s?(?:대상|순위)?\s?[:：(]?\s?[^\n.]{0,6}$")


def j_income31(F) -> list:
    cl = [c for k in ("target", "criteria") for c in clauses(F.get(k, "")) if not V.INCOME_DROP.search(c)]
    t = "\n".join(cl)
    out = []
    for label, rule, pat in [("명시", "income_explicit", V.INCOME_EXPLICIT), ("부분", "income_partial", V.INCOME_PARTIAL)]:
        for m in pat.finditer(t):
            ls = t.rfind("\n", 0, m.start()) + 1
            le = t.find("\n", m.end())
            le = len(t) if le < 0 else le
            if PRIORITY_AFTER.match(t[m.end():le]) or PRIORITY_BEFORE.search(t[ls:m.start()]):
                continue
            out.append(H(label, rule, ev(t, m)))
            break
    return out


# ------------------------------------------------------------------ 거주 (v3 이슈: 오탐·관내 누락)
RES_TOKEN31 = re.compile(r"거주|주민등록|전입|주소(?!득)지?|거소")
RES_BAD_LEFT31 = re.compile(
    V.RES_BAD_LEFT.pattern + r"|공작물\s?$|비\s?주택\s?$|(?:그룹홈|공동생활가정)\)?\s?(?:에서?|에)?\s?$"
    r"|(?:주택|건물|가옥)\s?(?:에서?|에)?\s?$|(?:독립|별도)\s?$|(?:기업|사업장|업체|법인|동일|건축물)\s?$")
RES_BAD_RIGHT31 = re.compile(V.RES_BAD_RIGHT.pattern + r"|^\s?(?:하는|하던|하고\s?있는|한)?\s?(?:주택|건물|가옥)")
KWAN_BARE = re.compile(r"관내\s?(?:에?\s?)?(?!\s*(?:분양|분만|소재|위치|설치|사업장|기업|업체|병원|의료|시설|학교|어린이집|기관))[^\n.]{0,10}?(?:재가|대상자|주민|세대|가구|노인|아동|청소년|청년|독거|저소득|취약|장애인|산모|임산부)")
_J_RESIDENCE = V.j_residence


def residence_generic31(t: str):
    for c in clauses(t):
        for m in RES_TOKEN31.finditer(c):
            left, right = c[max(0, m.start() - 8):m.start()], c[m.end():m.end() + 10]
            if RES_BAD_RIGHT31.match(right) or RES_BAD_LEFT31.search(left):
                continue
            if m.group(0) == "전입" and re.match(r"\s?신고", right):
                continue
            if m.group(0).startswith("주소") and re.match(r"\s?[:：]", right):     # '주소 : ○○시' 주소 표기 줄
                continue
            return H("부분", "res_generic", ev(c, m))
    return None


def j_residence31(F) -> list:
    out = list(_J_RESIDENCE(F))
    t = "\n".join(F.get(k, "") for k in ("target", "criteria"))
    m = KWAN_BARE.search(t)
    if m:
        out.append(H("부분", "res_kwan_bare", ev(t, m)))
    return out


# ------------------------------------------------------------------ 신청 경로 (코드북 3.1 d + v3 이슈: 긴 줄 방문 장소)
_J_CHANNEL = V.j_channel
ROUTE_BAD31 = re.compile(V.ROUTE_BAD.pattern + r"|예약")        # 선택적 예약 줄은 방법 줄이 아니다 (필수 예약은 ACT31)
LABEL_STRIP = re.compile(r"^(?:신청\s?방법|접수\s?방법|신청\s?경로|방법)\s*[:：]\s*")
BULLET = re.compile(r"^[-•·ᄋ○●■□▶▷◆◇※▪◦\s]+")
FILLER = re.compile(r"[\s,，/·ㆍ・&()\[\]*]|및|또는|혹은|과|와|이나|가능|으로|로|신청|접수|등|중|택일|선택")
ROUTE_TAIL = re.compile(r"(?:지참|구비|가지고|갖고|준비|첨부)\s?(?:하여|해서|하고|한\s?후|후)\s*([^\n]{0,70}?)(?:방문|내방|접수|제출)\s*(?:\([^)]{0,30}\))?\s*$")


def method_only(c: str):
    """방법 어휘만 있는 줄(방문 및 전화). 문의 줄(문의: 전화)이면 None."""
    c2 = V.INQUIRY_TAIL.sub("", c)
    c2 = BULLET.sub("", re.sub(r"\([^)]*\)|\[[^\]]*\]", "", c2))
    c2 = LABEL_STRIP.sub("", c2)
    ms = list(V.METHOD.finditer(c2))
    if not ms:
        return None
    rest = V.METHOD.sub("", c2)
    return ms[0] if not FILLER.sub("", rest) else None


def j_channel31(F) -> list:
    hits = list(_J_CHANNEL(F))
    t = F.get("apply", "")
    for c in clauses(t):
        m = method_only(c)
        if m:
            hits.append(H("명시", "channel_method_only", ev(c, m)))
            break
    for c in clauses(t):
        m = ROUTE_TAIL.search(c)
        if m and (V.PLACE.search(m.group(1)) or V.METHOD.search(m.group(1))):
            hits.append(H("명시", "channel_route_tail", ev(c, m)))
            break
    return hits


# ------------------------------------------------------------------ 신청 기간 (코드북 3.1 f: 내용 우선, 지급·처리 기한 제외)
ELIG_DATE = re.compile(r"출생(?:아|자)?\s?(?:이후|이전|부터)|(?:이후|이전)\s?출생|출생(?:일)?\s?기준|입양(?:아)?\s?(?:이후|이전)|\d\s?월\s?(?:이후\s?)?출생")
DEADLINE_NUM = re.compile(r"\d+\s?(?:일|개월|주|년)\s?(?:이내|내|까지)")
PAY_AFTER = re.compile(r"^\s?(?:이내)?\s?(?:에|으로|로|내에)?\s?(?:이\s?)?(?:지급|처리|입금|송금|결정|통보|통지|발급)")


def pay_deadline(c: str, m: re.Match) -> bool:
    return bool(PAY_AFTER.match(c[m.end():]))


def j_period31(F) -> list:
    t = F.get("apply", "")
    hits = []
    for m in V.PERIOD_LABEL.finditer(t):
        c = m.group(1)
        if ELIG_DATE.search(c) and not DEADLINE_NUM.search(c):      # 라벨은 신청시기, 내용은 자격 기준일
            continue
        d = DEADLINE_NUM.search(c)
        if d and pay_deadline(c, d):
            continue
        if V.PARTIAL_PERIOD.search(c):
            hits.append(H("부분", "period_label_partial", ev(t, m)))
        elif V.PERIOD_ALWAYS.search(c):
            hits.append(H("명시", "period_always", ev(t, m)))
        elif re.search(r"\d|매년|매월|분기|반기|상반기|하반기|연초|연말|이내|로부터|기준|발생|이후|까지|부터|중|내", c):
            hits.append(H("명시", "period_label_date", ev(t, m)))
        elif re.search(r"참조|참고|확인|문의|별도", c):
            hits.append(H("부분", "period_label_partial", ev(t, m)))
    for c in clauses(t):
        if V.PERIOD_CTX.search(c):
            m = V.PERIOD_ALWAYS.search(c)
            if m:
                hits.append(H("명시", "period_always", ev(c, m)))
            m = V.PERIOD_BUDGET.search(c)
            if m:
                hits.append(H("명시", "period_budget", ev(c, m)))
            for m in V.PERIOD_EVENT.finditer(c):
                tail = DEADLINE_NUM.search(c, m.start())
                if tail and pay_deadline(c, tail):
                    continue
                hits.append(H("명시", "period_event", ev(c, m)))
                break
            m = V.PARTIAL_PERIOD.search(c)
            if m:
                hits.append(H("부분", "period_partial", ev(c, m)))
        if V.APPLY_WORD.search(c):
            for m in V.DATE.finditer(c):
                if V.DATE_BAD_AFTER.match(c[m.end():]) or V.DATE_BAD_BEFORE.search(c[:m.start()]):
                    continue
                hits.append(H("명시", "period_date", ev(c, m)))
                break
    return hits


# ------------------------------------------------------------------ 적용
TEXT_JUDGE31 = {**V.TEXT_JUDGE, "age": j_age31, "income": j_income31, "residence": j_residence31,
                "channel": j_channel31, "period": j_period31}
PATCH = {"ACT": ACT31, "ROUTE_BAD": ROUTE_BAD31, "NOT_NEEDED": NOT_NEEDED31, "RES_TOKEN": RES_TOKEN31, "RES_BAD_LEFT": RES_BAD_LEFT31,
         "RES_BAD_RIGHT": RES_BAD_RIGHT31, "residence_generic": residence_generic31, "TEXT_JUDGE": TEXT_JUDGE31,
         "j_age": j_age31, "j_income": j_income31, "j_residence": j_residence31, "j_channel": j_channel31,
         "j_period": j_period31}
_depth = 0


@contextlib.contextmanager
def patched():
    """q2_items_v3 의 전역을 v3.1 으로 바꾼 동안만 판정한다. 끝나면 반드시 되돌린다."""
    global _depth
    saved = {k: getattr(V, k) for k in PATCH}
    if _depth == 0:
        for k, v in PATCH.items():
            setattr(V, k, v)
    _depth += 1
    try:
        yield
    finally:
        _depth -= 1
        if _depth == 0:
            for k, v in saved.items():
                setattr(V, k, v)


def judge31(item: str, side: str, source: str, F: dict, scope: str) -> dict:
    with patched():
        return V.judge(item, side, source, F, scope)


def run31(pairs: pd.DataFrame, sf, gf) -> pd.DataFrame:
    with patched():
        df = V.run(pairs, sf, gf)
    df["rule_version"] = RULE_VERSION
    df["core_metric"] = df["item"].isin(CB.CORE_ITEMS)
    return df


# ------------------------------------------------------------------ 판단 가능성 파생 열 (CB.JUDGEABLE, 집계에는 쓰지 않는다)
OUT_JUDGE = PROCESSED / "q2_items_v3_1_judgeable.csv"


def _tri(vals: list, ok: set):
    """True/False, 입력에 판단불가가 있으면 None(빈 값)"""
    if "판단불가" in vals:
        return None
    return all(v in ok for v in vals)


def judgeability(df: pd.DataFrame) -> pd.DataFrame:
    """쌍·side·scope 마다 judge_eligibility(age, income 모두 명시/해당없음), judge_application(channel 명시, period 명시/해당없음).
    JA 구조화 조건은 쓰지 않는다. 값은 'True'/'False'/''."""
    d = df.assign(lab=V.label(df)).pivot_table(index=["servId", "gov_id", "side", "scope"], columns="item", values="lab", aggfunc="first")
    ok = {"명시", "해당없음"}
    el = [_tri([r["age"], r["income"]], ok) for _, r in d.iterrows()]
    ap = [_tri([r["channel"], r["period"]], ok) if r["channel"] == "명시" or r["channel"] == "판단불가" or r["period"] == "판단불가"
          else False for _, r in d.iterrows()]
    d = d.reset_index()[["servId", "gov_id", "side", "scope"]]
    fmt = lambda v: "" if v is None else str(v)
    d["judge_eligibility"], d["judge_application"] = [fmt(v) for v in el], [fmt(v) for v in ap]
    return d


# ------------------------------------------------------------------ 변경 기록
def _changes_spec() -> list:
    """(rule_id, change, reason, source, 예시 판별 함수(item, r3, r31, text) → bool)"""
    C, O = "codebook_v3.1", "v3_validation_open_issue"
    return [
        ("age_birth_category", "출산 자격 문구(경계 없음)는 연령 부분. 신생아·영아는 v3 age_category 로 이미 부분", "코드북 3.1 (a): 경계 없이 단어만으로 명시로 보지 않는다", C,
         lambda i, a, b, t: i == "age" and b["rule_id"] == "age_birth_category"),
        ("age_birth_elapsed", "출생·출산·생후 뒤 N일/개월/년 이내 같은 출생 후 기간은 연령 명시. 신청·지급 기한 절은 제외", "코드북 3.1 (a): 출생일·출생 후 기간 경계가 있을 때만 명시", C,
         lambda i, a, b, t: i == "age" and b["rule_id"] == "age_birth_elapsed"),
        ("age_legal", "미성년은 자격 대상의 연령일 때만 명시. 동반·보호자·법정대리인·동의 절은 제외", "코드북 3.1 (b)", C,
         lambda i, a, b, t: i == "age" and a["rule_id"] == "age_legal" and b["rule_id"] != "age_legal"),
        ("ACT(예약)", "예약은 예약 필수·사전예약 후 신청/방문일 때만 신청 행위. 선택적 예약은 제외", "코드북 3.1 (c)", C,
         lambda i, a, b, t: i == "channel" and "예약" in t and (a["level"], a["meta"]) != (b["level"], b["meta"])),
        ("channel_method_only", "신청 행위어 없이 방법 어휘만 있는 줄(방문 및 전화)은 명시. 문의 줄은 제외", "코드북 3.1 (d)", C,
         lambda i, a, b, t: i == "channel" and b["rule_id"] == "channel_method_only"),
        ("income_priority", "우선 선발·우선 지원·차등지원 문구가 붙은 소득 표현은 자격 근거에서 제외", "코드북 3.1 (e)", C,
         lambda i, a, b, t: i == "income" and (a["level"], a["meta"]) != (b["level"], b["meta"])),
        ("period_label_content", "신청시기 라벨 뒤 내용이 자격 기준일(출생아 이후 등)이면 신청 기간 근거에서 제외", "코드북 3.1 (f): 내용 우선", C,
         lambda i, a, b, t: i == "period" and a["rule_id"] == "period_label_date" and (a["level"], a["meta"]) != (b["level"], b["meta"])),
        ("period_pay_deadline", "N일 이내 지급·처리·입금·송금·결정·통보·통지·발급 같은 지급·처리 기한은 신청 기간 근거에서 제외", "코드북 3.1 (f)", C,
         lambda i, a, b, t: i == "period" and a["rule_id"] == "period_event" and (a["level"], a["meta"]) != (b["level"], b["meta"])),
        ("age_birth_dot", "마침표 날짜('2025.9.30. 이전 출생', '2025. 1. 1. 이후 출생아')를 출생일 경계로 인정해 연령 명시", "v3 검증 표본에서 연령 경계가 있는데 부분으로 판정됨", O,
         lambda i, a, b, t: i == "age" and b["rule_id"] == "age_birth_dot"),
        ("age_deceased_infant", "영아·신생아·출생과 사망·장례가 같은 절에 있으면(사망 영아 정의) 그 절을 연령 판정에서 제외", "v3 파일럿: 사망 영아의 정의가 자격 연령으로 판정됨. 보수적으로 절 단위 제외", O,
         lambda i, a, b, t: i == "age" and a["level"] != "" and b["level"] == "미명시" and b["rule_id"] == ""),
        ("res_token(주소득자)", "거주 토큰 '주소'가 '주소득자'에서 걸리지 않게 함", "v3 검증 표본 거주 오탐", O,
         lambda i, a, b, t: i == "residence" and "주소득" in t and a["rule_id"] == "res_generic" and b["rule_id"] != "res_generic"),
        ("res_bad_left(주거 형태)", "비닐간이공작물·비주택·주택·건물·그룹홈 거주는 주거 형태로 보고 거주 요건에서 제외", "v3 검증 표본 거주 오탐", O,
         lambda i, a, b, t: i == "residence" and re.search("공작물|그룹홈|비주택|주택", t) is not None and (a["level"], a["meta"]) != (b["level"], b["meta"])),
        ("res_bad_right(거주하는 주택)", "'거주하는 주택·건물'은 거주 요건이 아니라 주거 형태로 제외", "v3 검증 표본 거주 오탐(긴급복지 화재 문구)", O,
         lambda i, a, b, t: i == "residence" and "거주하는 주택" in t and (a["level"], a["meta"]) != (b["level"], b["meta"])),
        ("res_generic(주소 표기 줄)", "'주소 : ○○' 표기 줄, 신청기업·임차주택·동일 주소는 거주 요건에서 제외", "v3 res_generic 증거 점검: 주소 표기 줄 오탐", O,
         lambda i, a, b, t: i == "residence" and "주소" in t and (a["level"], a["meta"]) != (b["level"], b["meta"]) and "주소득" not in t),
        ("res_kwan_bare", "'관내' 뒤에 대상자 명사(재가노인 등)가 이어지고 거주 표현이 없으면 거주 부분", "v3 검증 표본: 관내 재가노인 대상자를 놓침", O,
         lambda i, a, b, t: i == "residence" and b["rule_id"] == "res_kwan_bare"),
        ("channel_not_needed(신청을 통한 지원 아님)", "'신청을 통한 지원 아님'을 신청 불필요로 인정(신청 경로 명시, 기간·서류 해당없음)", "v3 검증 표본 V-channel-gov-070", O,
         lambda i, a, b, t: "신청을 통한 지원" in t and a["rule_id"] != b["rule_id"]),
        ("channel_route_tail", "'서류 지참하여 … 보건소 방문' 처럼 50자를 넘는 줄도 끝이 방문 장소이면 명시", "v3 channel_bare_route 길이 제한(50자) 누락: V-channel-gov-068", O,
         lambda i, a, b, t: i == "channel" and b["rule_id"] == "channel_route_tail"),
    ]


def example_ids(df3: pd.DataFrame | None = None) -> dict:
    """표본 단위(파일럿·검증·item_rule_sample)를 v3 와 v3.1 로 다시 판정해 변경 사례를 찾는다."""
    _, _, sf, gf = V.load_inputs()
    units = []
    for name in ("v3_pilot_sample", "v3_validation_sample", "item_rule_sample"):
        p = V.RULES_DIR / f"{name}.csv"
        if p.exists():
            s = V.read(p)
            units += list(zip(s["sample_id"], s["item"], s["side"], s["servId"], s["gov_id"]))
    spec = _changes_spec()
    ex = {s[0]: [] for s in spec}
    src = V.read_source()
    for sid, item, side, servId, gov_id in units:
        F = V.side_fields(sf, gf, servId, gov_id, side)
        a = V.judge(item, side, src.get(servId, ""), F, "text")
        b = judge31(item, side, src.get(servId, ""), F, "text")
        txt = "\n".join(F.get(k, "") for k in set(CB.FIELDS[item]["text"]))
        for rule_id, _, _, _, pred in spec:
            if pred(item, a, b, txt) and sid not in ex[rule_id]:
                ex[rule_id].append(sid)
    return ex


def changes_table() -> pd.DataFrame:
    ex = example_ids()
    rows = [{"rule_id": r, "change": c, "reason": why, "source": s, "example_sample_ids": ";".join(ex[r][:6])}
            for r, c, why, s, _ in _changes_spec()]
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ 실행
def diff_vs_v3(df31: pd.DataFrame, df3: pd.DataFrame) -> pd.DataFrame:
    key = ["servId", "gov_id", "side", "scope", "item"]
    x = df3.merge(df31, on=key, suffixes=("_3", "_31"), validate="1:1")
    x["lab_3"], x["lab_31"] = V.label(x.rename(columns={"level_3": "level", "meta_3": "meta"})), \
        V.label(x.rename(columns={"level_31": "level", "meta_31": "meta"}))
    return x[x["lab_3"].ne(x["lab_31"]) | x["rule_id_3"].ne(x["rule_id_31"])]


def main() -> None:
    m, g, sf, gf = V.load_inputs()
    pairs = V.pairs_table()
    df = run31(pairs, sf, gf)
    df.to_csv(OUT, index=False, encoding="utf-8-sig")
    d = diff_vs_v3(df, V.read(V.OUT)) if V.OUT.exists() else pd.DataFrame()
    t = d[d["scope"].eq("text")] if len(d) else d
    meta = {"created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "rule_version": RULE_VERSION,
            "rows": {"q2_items_v3_1": len(df), "pairs": len(pairs)},
            "core_items": CB.CORE_ITEMS, "excluded_items": CB.EXCLUDED_ITEMS,
            "changed_units_vs_v3": {"all_scopes": int(len(d)), "text_scope": int(len(t)),
                                    "text_by_item": {k: int(v) for k, v in t.groupby("item").size().items()} if len(t) else {}},
            "inputs_sha256": {str(p.relative_to(ROOT)): V.sha256(p) for p in [OUT.parent / "q2_items_v3.csv", PROCESSED / "q2_match.csv"] if p.exists()},
            "code_sha256": {f"src/preprocess/{n}.py": V.sha256(ROOT / "src" / "preprocess" / f"{n}.py")
                            for n in ("q2_items_v31", "q2_items_v3", "q2_codebook")}}
    judgeability(df).to_csv(OUT_JUDGE, index=False, encoding="utf-8-sig")
    OUT_META.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    tab = changes_table()
    tab.to_csv(CHANGES, index=False, encoding="utf-8-sig")
    print(f"q2_items_v3_1 {len(df)}행 → {OUT.relative_to(ROOT)}; v3 대비 본문 범위 변경 {len(t)}건 {meta['changed_units_vs_v3']['text_by_item']}")
    print(f"변경 기록 {len(tab)}건 → {CHANGES.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
