"""Q2 정보 항목 명시 수준 규칙 v3 (코드북 v3 구현, 검증 전 잠정).

정의는 q2_codebook.py 가 정한다. 이 모듈은 그 정의를 규칙으로 옮긴 것이고, 정의를 바꾸지 않는다.
q2_match.py 의 v1/v2 규칙(ITEMS, ITEMS_V2)은 건드리지 않고 비교용으로 남긴다.

판정 단위: (쌍, 플랫폼 side, 범위 scope, 항목 item) → level(명시/부분/미명시) 또는 meta(해당없음/판단불가),
rule_id, evidence(일치 구간 앞뒤 20자, 80자 이내).
  - 범위 text = 두 플랫폼 공통 본문 필드만, all = 플랫폼 전용 필드(신청방법 분류·신청기한·구비서류·문의처) 추가
  - 원천에 필드가 없는 경우는 fields_absent 열에 따로 적고, 그 자체로 미명시·meta 로 바꾸지 않는다
  - 여러 근거는 가장 높은 수준을 취한다(명시 > 부분 > 미명시). meta 는 해당없음이거나, 기여한 값이 모두 판단불가일 때만 붙인다

실행: python -m src.preprocess.q2_items_v3
  → data/processed/q2_items_v3.csv, q2_items_v3.meta.json, logs/validation/q2_item_rules/gov24_default_check.csv
표본·검수 시트: python -m src.preprocess.q2_items_v3 pilot | validate | sheet | crosstab
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
import unicodedata
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from ..collect._common import PROCESSED, RAW, ROOT
from . import q2_codebook as CB
from . import q2_match as M

OUT = PROCESSED / "q2_items_v3.csv"
OUT_META = PROCESSED / "q2_items_v3.meta.json"
RULES_DIR = ROOT / "logs" / "validation" / "q2_item_rules"
MATCH_FINAL = ROOT / "logs" / "validation" / "q2_match" / "match_final.csv"
JUDGED_BY = "AI 판정(사람 검증 전)"
ITEMS = list(CB.FIELDS)
SEED_PILOT, SEED_VALID = 20261106, 20261107
NO_CONFIRMED = "매칭 확정 0건: 산출 보류"


# ------------------------------------------------------------------ 공통 도구
def nfkc(s) -> str:
    return unicodedata.normalize("NFKC", "" if s is None or (isinstance(s, float) and np.isnan(s)) else str(s))


# 문장·절 단위: 줄바꿈, 한글/영문/닫는 괄호 뒤 마침표(숫자 뒤 마침표는 날짜라 자르지 않는다), 기호 글머리
SPLIT = re.compile(r"\n+|(?<=[가-힣a-zA-Z)\]])[.;!?](?=\s|$)|[;!?]\s|[○●■□▶▷◆◇※▪◦]")


def clauses(t: str) -> list[str]:
    return [c.strip() for c in SPLIT.split(t or "") if c and c.strip()]


def ev(s: str, m: re.Match, pad: int = 20, cap: int = 80) -> str:
    a, b = max(0, m.start() - pad), min(len(s), m.end() + pad)
    return re.sub(r"\s+", " ", s[a:b]).strip()[:cap]


def first(pat: re.Pattern, s: str):
    m = pat.search(s)
    return (m, ev(s, m)) if m else (None, "")


# hit = (label, rule_id, evidence). label ∈ 명시 / 부분 / 해당없음 / 판단불가
def H(label, rule, evidence):
    return (label, rule, evidence)


# ------------------------------------------------------------------ 신청 불필요
NOT_NEEDED = re.compile(
    r"신청\s?불필요|별도(?:의)?\s?신청(?:\s?(?:절차|사항))?\s?(?:없이|없음|불필요)|신청\s?(?:절차\s?)?없이"
    r"|(?:개인\s?)?신청\s?절차\s?(?:가\s?)?없|신청(?:이|은)?\s?필요\s?없|신청하지\s?(?:않아도|않고)"
    r"|직권(?!\s?(?:취소|해지|말소))|자동(?:으로|적으로)?\s?(?:지급|제공|선정|지원)")
FALLBACK = re.compile(r"누락|희망|원하|추가\s?신청|못한|미수령|미지급|이외|그\s?외|제외")

# ------------------------------------------------------------------ 신청 경로
ACT = re.compile(r"(?:신청|접수|제출|청구|신고|예약)(?!\s?(?:문의|안내|상담|자격|대상|기간|기한|일시|불필요|없))")
METHOD = re.compile(
    r"방문(?!\s?(?:간호|요양|목욕|재활|교육|상담|진료|급식|서비스|돌봄|지도|점검|검진|조사|관리|치료|수업|활동|방역|설치|수리|배달|보건|가사|도우미|운동|물리|건강))"
    r"|내방|온라인|인터넷|홈페이지|누리집|웹사이트|웹\s?페이지|사이트|정부\s?24|복지로|보조금\s?24|우편|등기|팩스|e-?mail|이메일|전자우편"
    r"|전화(?!\s?번호)|모바일|어플|앱(?![가-힣])|콜센터|https?://|www\.|메뉴|접속|직접\s?(?:제출|신청|방문)|대리\s?신청", re.I)
PLACE = re.compile(
    r"주민센터|행정복지센터|동사무소|읍\s?[·ㆍ./,]?\s?면\s?[·ㆍ./,]?\s?동|읍면동|읍[·ㆍ/]면|시\s?[·ㆍ./,‧]\s?군\s?[·ㆍ./,‧]\s?구|시군구|시[·ㆍ‧]도"
    r"|시청|군청|구청|도청|보건소|보건지소|보건의료원|복지관|센터|사무소|사무실|접수처|공단|공사|재단|진흥원|협회|교육청|지원청|은행|농협|우체국|지사|지부"
    r"|병원|의원|의료원|조리원|조합|위원회|본부|관공서|학교|[가-힣]{2,8}(?:과|팀|실|국|부서)\s?(?:에|로|에서)")
GENERIC = re.compile(r"해당\s?(?:기관|부서|관청|지자체|시설)|관할\s?(?:기관|관청|부서)|담당\s?(?:기관|부서|자|과)|접수\s?기관|신청\s?기관|관련\s?기관|소관\s?(?:기관|부서)|각\s?(?:기관|지자체)|지자체|주관\s?기관")
NOTICE = re.compile(r"공고(?:문)?\s?(?:을\s?)?(?:참조|참고|확인)|별도\s?공고")
CH_LABEL = re.compile(r"(?:신청\s?(?:방법|장소|처|경로)|접수\s?(?:처|장소|방법|기관)|제출\s?(?:처|방법|장소)|신청\s?및\s?접수|신청)\s*[:：]\s*|(?:신청\s?(?:방법|장소|처)|접수\s?(?:처|장소))\s+")
METHOD_LABEL = re.compile(r"^[-•·ᄋ\s]*(?:방문|온라인|우편|인터넷|전화|팩스|이메일|모바일)(?:\s?(?:및|/|,|또는)\s?[가-힣]{2,6})*\s*[:：]")
ROUTE_BAD = re.compile(r"문의|상담|안내|교육|제공|지급|수령|이용|서비스|발급|홍보|확인|추천|대상")
INQUIRY_TAIL = re.compile(r"(?:문의|연락|상담)\s?(?:처|전화)?\s?[:：].*$")


def channel_act_hits(t: str):
    """신청 행위와 같은 절에 있는 방법·장소. [(절 번호, 절, match)]"""
    out = []
    for i, c in enumerate(clauses(t)):
        c2 = INQUIRY_TAIL.sub("", c)
        if not ACT.search(c2):
            continue
        m = METHOD.search(c2) or PLACE.search(c2)
        if m:
            out.append((i, c2, m))
    return out


def not_needed(t: str):
    """신청 불필요 문구. 같은 본문에 누락자 등 별도 신청 경로가 따로 있으면 신청 있는 서비스로 본다(period·documents 해당없음 보류)."""
    cl = clauses(t)
    for i, c in enumerate(cl):
        m = NOT_NEEDED.search(c)
        if not m:
            continue
        fallback = any(j != i or FALLBACK.search(c) for j, _, _ in channel_act_hits(t))
        return ev(c, m), fallback
    return None, False


def j_channel(F) -> list:
    t = F.get("apply", "")
    hits = []
    e, _ = not_needed(t)
    if e:
        hits.append(H("명시", "channel_not_needed", e))
    for _, c, m in channel_act_hits(t):
        hits.append(H("명시", "channel_act_method", ev(c, m)))
        break
    for c in clauses(t):
        m = METHOD_LABEL.match(c)
        if m:
            hits.append(H("명시", "channel_method_label", ev(c, m)))
            break
        if len(c) <= 50 and not ROUTE_BAD.search(c) and not ACT.search(c):
            mm = METHOD.search(c) or PLACE.search(c)
            if mm and len(c) - len(mm.group(0)) <= 40:
                hits.append(H("명시", "channel_bare_route", ev(c, mm)))
                break
    for m in CH_LABEL.finditer(t):
        tail = t[m.end():m.end() + 100]
        mm = METHOD.search(tail) or PLACE.search(tail)
        if mm and not re.match(r"\s*문의", tail):
            hits.append(H("명시", "channel_label", ev(tail, mm, pad=10)))
            break
    for c in clauses(t):
        c2 = INQUIRY_TAIL.sub("", c)
        if ACT.search(c2):
            mg = GENERIC.search(c2)
            hits.append(H("부분", "channel_generic" if mg else "channel_unspecified", ev(c2, mg or ACT.search(c2))))
            break
    m = NOTICE.search(t)
    if m:
        hits.append(H("부분", "channel_notice", ev(t, m)))
    return hits


# ------------------------------------------------------------------ 신청 기간
PARTIAL_PERIOD = re.compile(
    r"별도\s?(?:의\s?)?(?:공고|안내|통보|공지)|추후\s?(?:안내|공지|공고|통보)|공고\s?(?:시|후|에\s?따라|일정|를\s?통해|참조|참고)"
    r"|접수\s?기관\s?별?\s?상이|기관\s?별\s?상이|지자체\s?별\s?(?:상이|다)|(?:시군구|시·군·구)(?:청)?\s?에?\s?따라")
PERIOD_LABEL = re.compile(r"(?:신청|접수|모집|공모|청구|제출)\s?(?:기간|기한|일정|시기|마감|일시)\s*[:：\-]?\s*([^\n]{0,80})")
PERIOD_CTX = re.compile(r"신청|접수|모집|공모|청구|제출|신고")
PERIOD_ALWAYS = re.compile(r"상시|연중|(?<!접)수시|언제든")
PERIOD_BUDGET = re.compile(r"예산\s?(?:이\s?)?소진|예산\s?(?:범위|한도)\s?(?:내)?\s?(?:에서|까지)")
PERIOD_EVENT = re.compile(
    r"(?:신청|신고|청구|접수|제출)[^\n.]{0,40}?\d+\s?(?:일|개월|년|주)\s?(?:이내|내|까지)"
    r"|\d+\s?(?:일|개월|년|주)\s?(?:이내|내)(?:에|로)?[^\n.]{0,25}(?:신청|신고|청구|접수|제출)")
DATE = re.compile(
    r"\d{4}\s?[.\-년]\s?\d{1,2}|(?<![\d개])\d{1,2}\s?월(?![분급액초평])|(?<!\d)\d{1,2}\s?[./]\s?\d{1,2}(?!\d)|매년|매월|매분기|분기별|상반기|하반기|연초|연말|\d{1,2}\s?일\s?(?:까지|부터|~)")
DATE_BAD_AFTER = re.compile(r"^.{0,10}?(?:지급|입금|출생|이후\s?입학|지원\s?기간|지원됩니다)")
DATE_BAD_BEFORE = re.compile(r"(?:지원\s?기간|지급\s?(?:시기|일)|사업\s?기간|운영\s?기간|교육\s?기간|출생)[^\n]{0,8}$")
APPLY_WORD = re.compile(r"신청(?!서)|접수|모집|공모")
CONTRADICT = {"period_date", "period_label_date", "period_partial", "period_label_partial"}


def j_period(F) -> list:
    t = F.get("apply", "")
    hits = []
    for m in PERIOD_LABEL.finditer(t):
        c = m.group(1)
        if PARTIAL_PERIOD.search(c):
            hits.append(H("부분", "period_label_partial", ev(t, m)))
        elif PERIOD_ALWAYS.search(c):
            hits.append(H("명시", "period_always", ev(t, m)))
        elif re.search(r"\d|매년|매월|분기|반기|상반기|하반기|연초|연말|이내|로부터|기준|발생|이후|까지|부터|중|내", c):
            hits.append(H("명시", "period_label_date", ev(t, m)))
        elif re.search(r"참조|참고|확인|문의|별도", c):
            hits.append(H("부분", "period_label_partial", ev(t, m)))
    for c in clauses(t):
        if PERIOD_CTX.search(c):
            m = PERIOD_ALWAYS.search(c)
            if m:
                hits.append(H("명시", "period_always", ev(c, m)))
            m = PERIOD_BUDGET.search(c)
            if m:
                hits.append(H("명시", "period_budget", ev(c, m)))
            m = PERIOD_EVENT.search(c)
            if m:
                hits.append(H("명시", "period_event", ev(c, m)))
            m = PARTIAL_PERIOD.search(c)
            if m:
                hits.append(H("부분", "period_partial", ev(c, m)))
        if APPLY_WORD.search(c):
            for m in DATE.finditer(c):
                if DATE_BAD_AFTER.match(c[m.end():]) or DATE_BAD_BEFORE.search(c[:m.start()]):
                    continue
                hits.append(H("명시", "period_date", ev(c, m)))
                break
    return hits


# ------------------------------------------------------------------ 제출 서류
DOC_NAME = re.compile(
    r"신청서|신분증|주민등록\s?(?:등|초)본|등본|초본|통장(?:\s?사본)?|계좌\s?(?:사본|번호)|사본|증명서|확인서|동의서|진단서|소견서|영수증|청구서|계획서"
    r"|사업자\s?등록증|자격증|통지서|서약서|계약서|가족관계\s?증명|건강보험증|수급자증|장애인증|복지카드|위임장|인감|납부확인|각서|증빙\s?사진|사진\s?\d")
DOC_CTX = re.compile(r"제출|구비|첨부|지참|서류|필요|준비|갖추|동봉|함께|발급받|원본|사본")
DOC_GENERIC = re.compile(r"구비\s?서류|제출\s?서류|관련\s?서류|증빙\s?(?:서류|자료)|필요\s?서류|해당\s?서류|서류|공고(?:문)?\s?(?:을\s?)?(?:참조|참고)")
DOC_NONE = re.compile(r"(?:구비|제출|필요)?\s?서류\s?(?:는\s?|가\s?)?(?:없음|불필요|필요\s?없|없이)|구비\s?서류\s?없|서류\s?제출\s?(?:없|불필요)")
DOC_BROAD = re.compile(r"[가-힣A-Za-z0-9]{1,}(?:서|증|본|통장|카드|사진|자료)(?![가-힣])")
DOC_SPECIAL = re.compile(r"^\s*(?:해당\s?없음|없음)\s*[.]?\s*$")


def docs_in(t: str, strict: bool = True) -> list:
    hits = []
    sticky = 0   # 서류 문맥 절 뒤 몇 개 절은 목록으로 본다
    for c in clauses(t):
        ctx = bool(DOC_CTX.search(c) or ACT.search(c)) or sticky > 0
        sticky = 6 if re.search(r"서류|증빙|첨부|구비|제출|지참|준비물", c) else max(0, sticky - 1)
        m = DOC_NONE.search(c)
        if m:
            hits.append(H("명시", "documents_none", ev(c, m)))
            continue
        if not strict or ctx:
            m = DOC_NAME.search(c)
            if m and not (m.group(0) == "증명서" and "발급" in c and not re.search(r"제출|첨부|지참", c)):
                hits.append(H("명시", "documents_named", ev(c, m)))
                continue
            if not strict:
                m = DOC_BROAD.search(c)
                if m:
                    hits.append(H("명시", "documents_named", ev(c, m)))
                    continue
        m = DOC_GENERIC.search(c)
        if m:
            hits.append(H("부분", "documents_generic", ev(c, m)))
    return hits


def j_documents(F) -> list:
    return docs_in(F.get("apply", ""))


# ------------------------------------------------------------------ 연락처
PHONE = re.compile(
    r"(?<![\d-])(?:0\d{1,2}\s?[-)\s.]{1,2}\s?\d{3,4}\s?[-.\s]\s?\d{4}|0\d{9,10}|1[5-9]\d{2}\s?[-]\s?\d{4}|0\d{1,2}[-)\s.]{1,2}1\d{2})(?!\d)"
    r"|(?:☎|전화|콜센터|상담\s?(?:센터|전화)?|문의|call)\s?[:：]?\s?(1\d{2,3})(?![\d-])", re.I)
OFFICE_WORD = (r"(?:[가-힣]{2,}(?:과|팀|실|국|센터|사무소|부서|복지관|공단|기관|구청|시청|군청|주민센터|보건소|청|처)|담당\s?(?:부서|기관|자|과)|해당\s?(?:기관|부서)|주민센터)")
OFFICE_INQ = re.compile(OFFICE_WORD + r"[^\n.]{0,12}?(?:문의|연락)|(?:문의|연락)\s?(?:처)?\s?[:：]\s?[가-힣]{2,}(?:과|팀|실|국|센터|사무소|부서|복지관|공단|청|처|소)")


def j_contact(F) -> list:
    t = F.get("apply", "")
    m = PHONE.search(t)
    if m:
        return [H("명시", "contact_phone", ev(t, m))]
    m = OFFICE_INQ.search(t)
    return [H("부분", "contact_office_inquiry", ev(t, m))] if m else []


# ------------------------------------------------------------------ 연령
AGE_NUM = re.compile(r"(?<![\d.])(?:만\s?)?\d{1,3}\s?세(?![대반]|\s?환자)")
AGE_BIRTH = re.compile(r"(?:\d{4}|[’'`]\d{2})(?![\d])[^\n.]{0,15}출생(?!\s?신고)|\d{4}\s?[~\-]?\s?(?:\d{4})?\s?년\s?생|출생\s?(?:일|연도|년도)?[^\n.]{0,12}\d{4}")
AGE_MONTH = re.compile(r"(?:생후|만)\s?\d{1,3}\s?개월|\d{1,3}\s?개월\s?(?:이상|미만|이하|이내|~|까지)?\s?(?:의\s?)?(?:영아|유아|아동|아기|신생아|자녀|영유아|어린이)")
AGE_SCHOOL = re.compile(r"초등학생|초등학교|중학생|중학교|고등학생|고등학교|초\s?[·ㆍ,]\s?중|초중고|대학생|대학원생|미취학|취학\s?전|학령")
AGE_NONE = re.compile(r"(?:연령|나이)\s?(?:제한\s?)?(?:은\s?)?(?:없|무관)|(?:연령|나이)에?\s?관계\s?없|전\s?연령|모든\s?연령")
AGE_LEGAL = re.compile(r"미성년")
AGE_CAT = re.compile(r"노인(?!복지|시설|쉼터|수련|센터|병원|의료|장기요양|성|일자리|돌봄|양로|요양|대학|정책|보호|학대|인권|회관|여가|교실|맞춤|문화)|어르신|청년|영유아|영아|유아|신생아|아동(?!학대|복지|센터|시설)|청소년(?!수련|쉼터|시설|복지|센터)|고령자?|노년|중장년|장년|어린이(?!집|\s?집)|유소년|시니어")
AGE_WORD = re.compile(r"연령|나이")


def first_hit(text, pats):
    for label, rule, pat in pats:
        m = pat.search(text)
        if m:
            return H(label, rule, ev(text, m))
    return None


AGE_RULES = [("명시", "age_none", AGE_NONE), ("명시", "age_numeric", AGE_NUM), ("명시", "age_birth", AGE_BIRTH),
             ("명시", "age_month", AGE_MONTH), ("명시", "age_school", AGE_SCHOOL), ("명시", "age_legal", AGE_LEGAL),
             ("부분", "age_category", AGE_CAT), ("부분", "age_mention", AGE_WORD)]


def j_age(F) -> list:
    t = "\n".join(F.get(k, "") for k in ("target", "criteria"))
    out = []
    for label, rule, pat in AGE_RULES:
        h = first_hit(t, [(label, rule, pat)])
        if h:
            out.append(h)
    return out


# ------------------------------------------------------------------ 소득·재산
INCOME_DROP = re.compile(r"면제|감면|중복\s?(?:지원|수혜|지급|신청)|중복.{0,8}불가")
INCOME_EXPLICIT = re.compile(
    r"중위\s?소득|기준\s?중위|소득\s?인정액|(?:기초)?\s?생활\s?(?:보장\s?)?수급|기초\s?수급|기초\s?생계|수급자|수급권자|차상위|(?:의료|생계|주거|교육)\s?급여"
    r"|건강\s?보험료[^\n.]{0,25}(?:이하|미만|이상|이내|\d)|소득[^\n.]{0,25}\d[\d,.]*\s?(?:%|만\s?원|원|천원|백만)"
    r"|(?:\d[\d,.]*\s?(?:만\s?)?원|\d+\s?%)\s?(?:이하|미만)[^\n.]{0,10}소득|소득\s?하위\s?\d+|소득(?:\s?수준|\s?기준|\s?요건|\s?제한)?\s?[:：]?\s?(?:에\s?)?(?:관계|상관|제한)?\s?(?:없|무관)"
    r"|보험료[^\n.]{0,20}(?:이하|미만)|재산[^\n.]{0,20}\d[\d,.]*\s?(?:억|만\s?)?원|평균\s?소득|전국\s?가구\s?평균")
INCOME_PARTIAL = re.compile(
    r"저소득|취약\s?계층|취약\s?가구|생활이?\s?(?:어려|곤란)|경제적(?:으로)?\s?(?:어려|곤란|취약)|어려운\s?(?:가정|형편|환경|가구|계층)|형편|빈곤"
    r"|소득\s?(?:기준|요건|조건|수준)|소득이?\s?(?:낮|적)|위기\s?가구")


def j_income(F) -> list:
    cl = [c for k in ("target", "criteria") for c in clauses(F.get(k, "")) if not INCOME_DROP.search(c)]
    t = "\n".join(cl)
    out = []
    for label, rule, pat in [("명시", "income_explicit", INCOME_EXPLICIT), ("부분", "income_partial", INCOME_PARTIAL)]:
        h = first_hit(t, [(label, rule, pat)])
        if h:
            out.append(h)
    return out


# ------------------------------------------------------------------ 거주
RES_STRONG = re.compile(r"(?:주민등록|주소|거소|전입)(?:을|이|지를|지가|지)?\s?(?:를\s?)?(?:두|둔|등재|된|되어|하고|한)|주민등록상(?!\s?(?:나이|연령|생년|만\s?\d|\d))"
                       r"|(?:주민등록|주소지?|거소)(?:지)?(?:을|를)?\s?[가-힣]{2,8}(?:시|군|구|도)?에\s?(?:두|둔|등재)|(?:주민등록|주소지?|거소)(?:지)?(?:가|는|:)?\s?[가-힣]{2,8}(?:시|군|구)(?:인|이다|에)|주소(?:를|지를)?\s?(?:계속\s?)?두")
RES_REGION = re.compile(
    r"[가-힣]{2,8}(?:특별자치도|특별자치시|특별시|광역시|시|군|구|도|읍|면|동)(?:에|내|의|에서)?\s?(?:\d+\s?(?:년|개월)\s?(?:이상)?\s?)?(?:계속\s?)?(?:주민등록|거주|거소|주소|전입|살|사는|소재)")
RES_KWAN = re.compile(r"(?:도내|시내|군내|구내)\s?(?:에서?\s?)?(?:\d+\s?(?:년|개월)\s?(?:이상\s?)?)?(?:계속\s?)?(?:거주|주민등록|주소|거소)|관내[^\n.]{0,12}(?:거주|주민|주소|전입|거소|살)|(?:거주|주민등록|주소)[^\n.]{0,10}관내")
RES_PERIOD = re.compile(r"\d+\s?(?:년|개월)\s?이상\s?(?:계속\s?)?(?:거주|주민등록|전입)|(?:거주|주민등록)\s?\d+\s?(?:년|개월)\s?이상")
RES_NATION = re.compile(r"전국(?!\s?가구)|거주\s?지?\s?(?:제한\s?)?(?:없|무관)")
RES_TOKEN = re.compile(r"거주|주민등록|전입|주소지?|거소")
RES_BAD_RIGHT = re.compile(r"^상\s?(?:나이|연령|생년|만\s?\d|\d)|^\s?(?:지)?\s?(?:관할)?\s?(?:읍|면|동|주민센터|행정복지|시청|군청|구청|사무소|번호|등본|초본|증|표|사본|시설)|^\s?(?:지)?\s?(?:신고)")
RES_BAD_LEFT = re.compile(r"(?:혼자|독거|홀로|단독|1인)\s?(?:로\s?)?$|(?:함께|같이|동거)\s?$|시설\s?(?:에서?|에)?\s?$|전입\s?$")


def residence_generic(t: str):
    for c in clauses(t):
        for m in RES_TOKEN.finditer(c):
            left, right = c[max(0, m.start() - 8):m.start()], c[m.end():m.end() + 10]
            if RES_BAD_RIGHT.match(right) or RES_BAD_LEFT.search(left):
                continue
            if m.group(0) == "전입" and re.match(r"\s?신고", right):
                continue
            return H("부분", "res_generic", ev(c, m))
    return None


def j_residence(F) -> list:
    t = "\n".join(F.get(k, "") for k in ("target", "criteria"))
    out = []
    for rule, pat in [("res_period", RES_PERIOD), ("res_kwan", RES_KWAN), ("res_strong", RES_STRONG), ("res_region", RES_REGION), ("res_nation", RES_NATION)]:
        for c in clauses(t):
            m = pat.search(c)
            if m and not (RES_BAD_RIGHT.match(c[m.end():m.end() + 10]) and pat is RES_REGION) and not RES_BAD_LEFT.search(c[max(0, m.start() - 8):m.start()]):
                out.append(H("명시", rule, ev(c, m)))
                break
    g = residence_generic(t)
    if g:
        out.append(g)
    return out


TEXT_JUDGE = {"age": j_age, "income": j_income, "residence": j_residence, "channel": j_channel,
              "period": j_period, "documents": j_documents, "contact": j_contact}

# ------------------------------------------------------------------ 플랫폼 전용 필드
GOV_CAT_CHANNEL = re.compile(r"방문신청|온라인신청|정부24|기타\s?온라인")
SSIS_CAT_CHANNEL = re.compile(r"방문|인터넷|전화|우편|팩스|e-?mail|이메일|모바일", re.I)
ALWAYS_DEFAULT = re.compile(r"^\s*상시\s?신청\s*$")


def cat_tokens(s: str) -> list:
    return [x.strip() for x in s.split("||") if x.strip()]


def gov_cat_not_needed(cat: str) -> bool:
    tk = cat_tokens(cat)
    return "신청불필요" in tk and all(x in ("신청불필요", "직접입력") for x in tk)


def service_nn(F, scope_all: bool, side: str):
    """신청 없는 서비스 판정 근거. 본문 → (증거, 보류여부), 분류·기한은 보조금24 전용(all 범위)."""
    e, fb = not_needed(F.get("apply", ""))
    nn_text = (e, fb) if e else None
    nn_cat = bool(scope_all and side == "gov" and gov_cat_not_needed(F.get("apply_cat", "")))
    nn_dl = None
    if scope_all and side == "gov":
        m = NOT_NEEDED.search(F.get("deadline", ""))
        nn_dl = ev(F["deadline"], m) if m else None
    return nn_text, nn_cat, nn_dl


def gov_docs_hits(F, nn_any: bool) -> list:
    d = F.get("documents", "").strip()
    if not d:
        return []
    if DOC_SPECIAL.match(d):
        if nn_any:
            return [H("해당없음", "gov_docs_na_consistent", f"[필드:{d}] 신청 없는 서비스")]
        txt = docs_in(F.get("apply", "")) + docs_in(F.get("target", ""))
        if txt:
            lab, _, e = txt[0]
            return [H(lab, "gov_docs_field_contradicted", f"[필드:{d}] {e}"[:80])]
        return [H("판단불가", "gov_docs_ambiguous", f"[필드:{d}]")]
    if DOC_NONE.search(d):
        return [H("명시", "gov_docs_none", ev(d, DOC_NONE.search(d)))]
    if M.PLACEHOLDER.fullmatch(d):
        return [H("부분", "gov_docs_generic", d[:80])]
    h = docs_in(d, strict=False)
    return [(l, "gov_docs_named" if r == "documents_named" else "gov_docs_generic", e) for l, r, e in h]


def gov_deadline_hits(F, apply_hits: list) -> list:
    d = F.get("deadline", "").strip()
    if not d or M.PLACEHOLDER.fullmatch(d) and not ALWAYS_DEFAULT.match(d):
        return []
    m = NOT_NEEDED.search(d)
    if m:
        return [H("해당없음", "gov_deadline_na", ev(d, m))]
    if ALWAYS_DEFAULT.match(d):
        txt = [h for h in apply_hits if h[1] in CONTRADICT]
        if txt:
            return [H(txt[0][0], "gov_deadline_contradicted", f"[필드:{d}] {txt[0][2]}"[:80])]
        return [H("명시", "gov_deadline_default", d)]
    if PARTIAL_PERIOD.search(d) or re.search(r"상이|다를\s?수", d):
        return [H("부분", "gov_deadline_partial", d[:80])]
    return [H("명시", "gov_deadline_text", d[:80])]


def gov_branch(F: dict) -> dict:
    """보조금24 구비서류 '해당없음/없음'·신청기한 '상시신청'의 가지 (gov24_default_check 용)."""
    nn_text, nn_cat, nn_dl = service_nn(F, True, "gov")
    nn_any = bool(nn_text or nn_cat or nn_dl)
    out = {}
    d = F.get("documents", "").strip()
    if DOC_SPECIAL.match(d):
        h = gov_docs_hits(F, nn_any)[0]
        out["documents"] = {"gov_docs_na_consistent": "consistent", "gov_docs_field_contradicted": "contradicted",
                            "gov_docs_ambiguous": "ambiguous"}[h[1]]
    if ALWAYS_DEFAULT.match(F.get("deadline", "")):
        h = gov_deadline_hits(F, j_period(F))[0]
        out["deadline"] = "contradicted" if h[1] == "gov_deadline_contradicted" else "default"
    return out


# ------------------------------------------------------------------ 판정
def combine(hits: list) -> tuple:
    """(level, meta, rule_id, evidence). 같은 수준이면 기본값(_default) 근거를 뒤로, 모순 근거를 앞으로 둔다."""
    order = sorted(range(len(hits)), key=lambda i: (hits[i][1].endswith("_default"), not hits[i][1].startswith("gov_docs_na"), "contradicted" not in hits[i][1], i))
    hits = [hits[i] for i in order]
    for lab in ("해당없음",):
        for h in hits:
            if h[0] == lab:
                return "", lab, h[1], h[2]
    for lab in ("명시", "부분"):
        for h in hits:
            if h[0] == lab:
                return lab, "", h[1], h[2]
    for h in hits:
        if h[0] == "판단불가":
            return "", "판단불가", h[1], h[2]
    return "미명시", "", "", ""


def judge(item: str, side: str, source: str, F: dict, scope: str) -> dict:
    """한 단위의 판정. F = {필드명: NFKC 텍스트}"""
    allv = scope == "all"
    hits = list(TEXT_JUDGE[item](F))
    absent = []
    nn_text, nn_cat, nn_dl = service_nn(F, allv, side)
    if item in ("period", "documents"):
        if nn_text and not nn_text[1]:
            hits.append(H("해당없음", f"{item}_not_needed", nn_text[0]))
        if item == "period" and nn_cat:
            hits.append(H("해당없음", "gov_cat_na", F.get("apply_cat", "")[:80]))
    if allv:
        if item == "channel":
            cat = F.get("apply_cat", "")
            if side == "gov":
                tk = cat_tokens(cat)
                if any(GOV_CAT_CHANNEL.search(x) for x in tk):
                    hits.append(H("명시", "gov_cat_channel", cat[:80]))
                elif gov_cat_not_needed(cat):
                    hits.append(H("명시", "gov_cat_not_needed", cat[:80]))
            elif source == "central":
                absent.append("apply_cat")
            elif SSIS_CAT_CHANNEL.search(cat):
                hits.append(H("명시", "ssis_cat_channel", cat[:80]))
        elif item == "period":
            if side == "gov":
                hits += gov_deadline_hits(F, [h for h in hits])
            else:
                absent.append("deadline")
        elif item == "documents":
            if side == "gov":
                nn_any = bool((nn_text and not nn_text[1]) or nn_cat or nn_dl)
                hits += gov_docs_hits(F, nn_any)
                if nn_cat or nn_dl:
                    hits.append(H("해당없음", "gov_service_na", (F.get("apply_cat", "") if nn_cat else nn_dl)[:80]))
            else:
                absent.append("documents")
        elif item == "contact":
            t = F.get("contact", "")
            m = PHONE.search(t)
            if m:
                hits.append(H("명시", f"{side}_contact_phone", ev(t, m)))
            elif re.search(r"[가-힣]{2,}", t):
                hits.append(H("부분", f"{side}_contact_office", t[:80].replace("\n", " ")))
    lv, meta, rule, evd = combine(hits)
    return {"level": lv, "meta": meta, "rule_id": rule, "evidence": evd, "fields_absent": ",".join(absent)}


# ------------------------------------------------------------------ 입력·실행
def read(path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, encoding="utf-8-sig", keep_default_na=False)


def load_inputs():
    m, g = M.load_ssis(), M.load_gov24()
    sf = M.ssis_fields(m).set_index("servId")
    gf = M.gov_fields(g).set_index("gov_id")
    return m, g, sf, gf


def pairs_table() -> pd.DataFrame:
    q = read(PROCESSED / "q2_match.csv")
    q = q[q["tier"].ne("U") & q["gov_id"].ne("")].copy()
    conf = pd.Series(False, index=q.index)
    if MATCH_FINAL.exists():
        f = read(MATCH_FINAL)
        ok = set(zip(f.loc[f["final_same"].eq("동일"), "servId"], f.loc[f["final_same"].eq("동일"), "gov_id"]))
        conf = pd.Series([(a, b) in ok for a, b in zip(q["servId"], q["gov_id"])], index=q.index)
    q["match_confirmed"] = conf
    return q[["servId", "gov_id", "tier", "gov_shared", "source", "match_confirmed"]].reset_index(drop=True)


def side_fields(sf, gf, servId, gov_id, side) -> dict:
    row = sf.loc[servId] if side == "ssis" else gf.loc[gov_id]
    return {k: nfkc(v) for k, v in row.items()}


def run(pairs: pd.DataFrame, sf, gf) -> pd.DataFrame:
    rows = []
    for r in pairs.itertuples(index=False):
        for side in ("ssis", "gov"):
            F = side_fields(sf, gf, r.servId, r.gov_id, side)
            for scope in ("text", "all"):
                for item in ITEMS:
                    j = judge(item, side, r.source, F, scope)
                    rows.append({"servId": r.servId, "gov_id": r.gov_id, "tier": r.tier, "gov_shared": r.gov_shared,
                                 "source": r.source, "match_confirmed": r.match_confirmed, "side": side, "scope": scope,
                                 "item": item, **j})
    cols = ["servId", "gov_id", "tier", "gov_shared", "source", "match_confirmed", "side", "scope", "item", "level",
            "meta", "rule_id", "evidence", "fields_absent"]
    return pd.DataFrame(rows)[cols]


def sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def label(df: pd.DataFrame) -> pd.Series:
    """level 또는 meta 하나의 판정 라벨"""
    return df["meta"].where(df["meta"].ne(""), df["level"])


def gov24_default_check(g: pd.DataFrame) -> pd.DataFrame:
    gf = M.gov_fields(g)
    rec = []
    for r in gf.to_dict("records"):
        F = {k: nfkc(v) for k, v in r.items()}
        b = gov_branch(F)
        if "documents" in b:
            rec.append(("구비서류 해당없음/없음", b["documents"]))
        if "deadline" in b:
            rec.append(("신청기한 상시신청", b["deadline"]))
    t = pd.DataFrame(rec, columns=["field", "branch"]).value_counts().rename("n").reset_index()
    t["share_of_field"] = (t["n"] / t.groupby("field")["n"].transform("sum")).round(4)
    return t.sort_values(["field", "branch"]).reset_index(drop=True)


def summary_confirmed(df: pd.DataFrame) -> None:
    c = df[df["match_confirmed"]]
    if c.empty:
        print(NO_CONFIRMED)
        return
    t = c[c["scope"].eq("text")].assign(lab=label(c[c["scope"].eq("text")]))
    print(pd.crosstab([t["item"], t["side"]], t["lab"]).to_string())


def main() -> None:
    m, g, sf, gf = load_inputs()
    pairs = pairs_table()
    df = run(pairs, sf, gf)
    df.to_csv(OUT, index=False, encoding="utf-8-sig")
    inputs = [M.MASTER, M.GOV_LIST, M.GOV_DETAIL, PROCESSED / "q2_match.csv"] + ([MATCH_FINAL] if MATCH_FINAL.exists() else [])
    meta = {"created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "rows": {"q2_items_v3": len(df), "pairs": len(pairs), "match_confirmed_pairs": int(pairs["match_confirmed"].sum())},
            "inputs_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in inputs},
            "code_sha256": {"src/preprocess/q2_items_v3.py": sha256(ROOT / "src" / "preprocess" / "q2_items_v3.py"),
                            "src/preprocess/q2_codebook.py": sha256(ROOT / "src" / "preprocess" / "q2_codebook.py")}}
    OUT_META.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    RULES_DIR.mkdir(parents=True, exist_ok=True)
    dc = gov24_default_check(g)
    dc.to_csv(RULES_DIR / "gov24_default_check.csv", index=False, encoding="utf-8-sig")
    print(f"q2_items_v3 {len(df)}행 (쌍 {len(pairs)}) → {OUT.relative_to(ROOT)}")
    print(dc.to_string(index=False))
    summary_confirmed(df)


# ------------------------------------------------------------------ 표본 (파일럿·독립 검증)
SHEET_TEXT = ["apply", "target", "criteria", "benefit", "summary"]


def clip(s: str, n: int = 1000) -> str:
    return s[:n]


def population(df: pd.DataFrame) -> pd.DataFrame:
    """A·B 등급, 보조금24 공유 아님, 본문(text) 범위. 단위 = (쌍, side, 항목)"""
    p = df[df["tier"].isin(["A", "B"]) & df["gov_shared"].eq("False") & df["scope"].eq("text")].copy()
    p["lab"] = label(p)
    p["stratum"] = p["lab"].where(p["lab"].isin(["명시", "부분", "미명시"]), "meta")
    return p


def draw(pop: pd.DataFrame, seed: int, exclude: set, per_item_side: int = 10, per_stratum: int = 3) -> pd.DataFrame:
    """항목×side마다 수준별 층에서 번갈아 최대 per_stratum 씩, 합계 per_item_side 까지."""
    rng = np.random.default_rng(seed)
    parts = []
    for item in ITEMS:
        for side in ("ssis", "gov"):
            sub = pop[pop["item"].eq(item) & pop["side"].eq(side)]
            sub = sub[[(a, b, item, side) not in exclude for a, b in zip(sub["servId"], sub["gov_id"])]]
            pools = {}
            for st in ["부분", "미명시", "명시", "meta"]:
                x = sub[sub["stratum"].eq(st)].sort_values(["servId", "gov_id"])
                pools[st] = list(x.index[rng.permutation(len(x))])
            picked, taken = [], {k: 0 for k in pools}
            while len(picked) < per_item_side and any(pools[k] and taken[k] < per_stratum for k in pools):
                for k in pools:
                    if pools[k] and taken[k] < per_stratum and len(picked) < per_item_side:
                        picked.append(pools[k].pop(0))
                        taken[k] += 1
            parts.append(sub.loc[picked])
    return pd.concat(parts)


def sheet_frame(smp: pd.DataFrame, sf, gf, prefix: str) -> pd.DataFrame:
    rows = []
    for n, r in enumerate(smp.itertuples(index=False), 1):
        F = side_fields(sf, gf, r.servId, r.gov_id, r.side)
        d = {"sample_id": f"{prefix}-{r.item}-{r.side}-{n:03d}", "item": r.item, "side": r.side, "servId": r.servId,
             "gov_id": r.gov_id, "source": r.source}
        d["designated_fields"] = "+".join(CB.FIELDS[r.item]["text"])
        d["designated_text"] = clip("\n".join(f"[{k}] {F.get(k, '')}" for k in CB.FIELDS[r.item]["text"]))
        for k in SHEET_TEXT:
            d[k] = clip(F.get(k, ""))
        rows.append(d)
    return pd.DataFrame(rows)


def confusion(sheet: pd.DataFrame) -> pd.DataFrame:
    a = sheet.assign(v3=np.where(sheet["v3_meta"].ne(""), sheet["v3_meta"], sheet["v3_level"]),
                     ai=np.where(sheet["ai_meta"].ne(""), sheet["ai_meta"], sheet["ai_level"]))
    t = a.groupby(["item", "side", "v3", "ai"]).size().rename("n").reset_index()
    t["agree"] = t["v3"].eq(t["ai"])
    return t


def guard_overwrite(path, cols) -> None:
    if path.exists():
        prev = read(path)
        if any(c in prev and prev[c].ne("").any() for c in cols):
            raise ValueError(f"판정값이 있어 덮어쓰지 않습니다: {path}")


def crosstab_v2_v3(df: pd.DataFrame) -> pd.DataFrame:
    fc = read(PROCESSED / "q2_field_compare.csv")
    fc = fc[fc["tier"].isin(["A", "B"]) & fc["gov_shared"].eq("False")]
    rec = []
    for scope in ("text", "all"):
        for item in ITEMS:
            for side in ("ssis", "gov"):
                v2 = fc[["servId", "gov_id", f"{side}_item2_{scope}_{item}"]].rename(columns={f"{side}_item2_{scope}_{item}": "v2"})
                v3 = df[df["scope"].eq(scope) & df["item"].eq(item) & df["side"].eq(side)]
                x = v3.merge(v2, on=["servId", "gov_id"])
                x["v3"] = label(x)
                t = x.groupby(["v2", "v3"]).size().rename("n").reset_index()
                t.insert(0, "side", side)
                t.insert(0, "item", item)
                t.insert(0, "scope", scope)
                rec.append(t)
    return pd.concat(rec).rename(columns={"v2": "v2_flag", "v3": "v3_label"})


V3_COLS = ["v3_level", "v3_meta", "v3_rule_id", "v3_evidence"]
AI_COLS = ["ai_level", "ai_meta", "ai_evidence", "ai_note"]
HUMAN_COLS = ["h_level", "h_meta", "h_evidence", "h_note", "reviewer", "reviewed_at"]
DEF_SHORT = {  # 시트에 싣는 항목별 정의 (코드북 요약)
    "age": "명시: 연령 경계를 알 수 있음(만 N세, 출생연도, 초·중·고등학생, 연령 제한 없음). 부분: 노인·청년·영유아 등 경계 없는 범주어. 미명시: 없음. 지원기간·혼인기간 등 자격 아닌 숫자는 제외",
    "income": "명시: 기준 수치 또는 법정 지위(중위소득 N%, 수급자, 차상위, 건보료 기준), 소득 무관. 부분: 저소득층·취약계층 등 기준 불명. 미명시: 없음. 면제·중복지원 불가 문구는 제외",
    "residence": "명시: 거주·주민등록 지역을 알 수 있음(관내 거주, ○○시 주민등록, 전국). 부분: 거주 요건은 있으나 지역·기준 불명. 미명시: 없음. 거주지 주민센터(신청 장소)·혼자 거주(가구 형태)는 제외",
    "channel": "명시: 신청·접수·제출·청구와 연결된 방법·장소(방문 신청, 복지로 온라인 신청), 신청 불필요 명시. 부분: 신청은 필요하나 어디서·어떻게 불분명. 미명시: 없음. 기관명만·문의처·서비스 제공 방문은 제외",
    "period": "명시: 신청 가능 시기(날짜·월, 상시·연중, 사건 기준 기한, 예산 소진 시까지). 부분: 별도 공고·추후 안내. 미명시: 없음. 해당없음(meta): 신청 불필요. 지원기간·지급시기·자격 기준일은 제외",
    "documents": "명시: 서류를 하나 이상 이름으로 알 수 있음 또는 서류 없음 명시. 부분: 구비서류 지참·증빙서류 등 불분명. 미명시: 없음. 해당없음(meta): 신청 불필요",
    "contact": "명시: 연락 가능한 전화번호·창구번호. 부분: 기관·부서만 있고 번호 없음. 미명시: 없음",
}


def read_source() -> dict:
    q = read(PROCESSED / "q2_match.csv")
    return dict(zip(q["servId"], q["source"]))


def _unit_set(sheet: pd.DataFrame) -> set:
    return set(zip(sheet["servId"], sheet["gov_id"], sheet["item"], sheet["side"]))


def make_sample(seed: int, prefix: str, path, exclude: set) -> pd.DataFrame:
    df = read(OUT)
    smp = draw(population(df), seed, exclude)
    _, _, sf, gf = load_inputs()
    sh = sheet_frame(smp, sf, gf, prefix)
    for a, b in zip(["level", "meta", "rule_id", "evidence"], V3_COLS):
        sh[b] = smp[a].to_numpy()
    sh["stratum"] = smp["stratum"].to_numpy()
    sh["tier"] = smp["tier"].to_numpy()
    for c in AI_COLS + ["judged_by"]:
        sh[c] = ""
    guard_overwrite(path, AI_COLS)
    sh.to_csv(path, index=False, encoding="utf-8-sig")
    return sh


def review_sheet(valid: pd.DataFrame) -> pd.DataFrame:
    """사람 검수용. v3·ai 열을 넣지 않는다(앵커링 방지). 본문은 자르지 않고 원문(NFKC) 전체를 싣는다."""
    _, _, sf, gf = load_inputs()
    rows = []
    for r in valid.itertuples(index=False):
        F = side_fields(sf, gf, r.servId, r.gov_id, r.side)
        d = {"sample_id": r.sample_id, "item": r.item, "side": r.side, "servId": r.servId, "gov_id": r.gov_id,
             "designated_fields": "+".join(CB.FIELDS[r.item]["text"]),
             "designated_text": "\n".join(f"[{k}] {F.get(k, '')}" for k in CB.FIELDS[r.item]["text"])}
        for k in SHEET_TEXT:
            d[k] = F.get(k, "")
        d["codebook_definition"] = DEF_SHORT[r.item]
        rows.append(d)
    sh = pd.DataFrame(rows)
    for c in HUMAN_COLS:
        sh[c] = ""
    return sh.sample(frac=1, random_state=SEED_VALID).reset_index(drop=True)


def cli(cmd: str) -> None:
    if cmd == "pilot":
        sh = make_sample(SEED_PILOT, "P", RULES_DIR / "v3_pilot_sample.csv", set())
        print(f"파일럿 {len(sh)}건"); print(sh.groupby(["item", "side"]).size().to_string())
    elif cmd == "validate":
        pilot = read(RULES_DIR / "v3_pilot_sample.csv")
        sh = make_sample(SEED_VALID, "V", RULES_DIR / "v3_validation_sample.csv", _unit_set(pilot))
        print(f"독립 표본 {len(sh)}건")
    elif cmd == "sheet":
        out = RULES_DIR / "item_review_sheet.csv"
        guard_overwrite(out, HUMAN_COLS)
        sh = review_sheet(read(RULES_DIR / "v3_validation_sample.csv"))
        sh.to_csv(out, index=False, encoding="utf-8-sig")
        print(f"검수 시트 {len(sh)}건 → {out.relative_to(ROOT)}")
    elif cmd == "refit":   # 규칙 수정 후 표본 단위를 현재 규칙으로 다시 판정해 v3fix_ 열로 붙인다
        for name in ("pilot", "validation"):
            path = RULES_DIR / f"v3_{name}_sample.csv"
            if not path.exists():
                continue
            sh = read(path)
            _, _, sf, gf = load_inputs()
            res = []
            for r in sh.itertuples(index=False):
                F = side_fields(sf, gf, r.servId, r.gov_id, r.side)
                src = F.get("source", "") or read_source().get(r.servId, "")
                j = judge(r.item, r.side, src, F, "text")
                res.append((j["level"], j["meta"], j["rule_id"], j["evidence"]))
            for k, c in enumerate(["v3fix_level", "v3fix_meta", "v3fix_rule_id", "v3fix_evidence"]):
                sh[c] = [x[k] for x in res]
            sh.to_csv(path, index=False, encoding="utf-8-sig")
            print(name, "refit", len(sh))
    elif cmd == "crosstab":
        t = crosstab_v2_v3(read(OUT))
        t.to_csv(RULES_DIR / "v2_v3_crosstab.csv", index=False, encoding="utf-8-sig")
        print(f"v2×v3 {len(t)}행")
    elif cmd == "confusion":
        for name in ("pilot", "validation"):
            sh = read(RULES_DIR / f"v3_{name}_sample.csv")
            t = confusion(sh)
            t.to_csv(RULES_DIR / f"v3_{name}_confusion.csv", index=False, encoding="utf-8-sig")
            print(name, "일치", int(t.loc[t["agree"], "n"].sum()), "/", int(t["n"].sum()))
    else:
        raise SystemExit(f"알 수 없는 명령: {cmd}")


if __name__ == "__main__":
    main() if len(sys.argv) == 1 else cli(sys.argv[1])
