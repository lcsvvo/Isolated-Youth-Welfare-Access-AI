"""사업·모집 기간 규칙 검증. 사업이 이미 끝난 경우만 '제외'함.

범위 결정 (data/processed/services_master.csv 5,288행 실측)
- enfc_end_ymd: 99991231(종료일 없음) 4,703, 결측 461(중앙부처), 그 외는 실제 종료일
- support_cycle은 급여 지급 주기이므로 모집 기간 판단에 쓰지 않음
- 모집 기간 원문(apply_text, summary_text, criteria_text, target_text): '신청기간' 160행, '접수기간' 21행,
  '모집기간' 11행, '상시' 98행, '연중' 48행, '수시' 93행. 연도가 포함된 'YYYY.MM.DD ~ ...' 형태로 파싱되는 행은
  소수이고 대부분 '1년 이내', '매년 9월~10월'처럼 날짜로 환산되지 않는 서술임

판정 (profile["today"]: 명시한 date, 'YYYY-MM-DD' 또는 'YYYYMMDD', 생략 시 한국 기준 오늘)
- enfc_end_ymd < today → 제외 (사업 종료). 이것만 제외 근거임
- 상시·연중·수시 접수·신청·모집 표현 → 통과
- 파싱된 모집 기간(연도 포함)이 오늘을 포함 → 통과 (모집 중)
- 모집 기간이 오늘 이후 시작 → 확인 필요 (모집 시작 전, 시작일 표기)
- 파싱된 모집 기간이 모두 이미 끝남 → 확인 필요. 원문이 지난 공고일 수 있고 매년 반복되는 사업이 많아 종료 확정 불가
- 연도 없는 반복 서술('매년 3월', '신청기간 3.30 ~ 5.29') → 예상 기간만 안내하고 확인 필요
- 모집 정보가 전혀 없음 → 확인 필요, 사유는 PERIOD_NO_INFO_REASON. 호출 측에서 이 사유로 순위 감점 제외 가능
"""
from __future__ import annotations

import calendar
import re
from datetime import date, datetime, timedelta, timezone

PASS, CHECK, EXCLUDE = "통과", "확인 필요", "제외"
PERIOD_NO_INFO_REASON = "모집기간 원문 없음"   # 모집 정보 부재 사유. 순위 감점 제외 판별용

_KEY = re.compile(r"(모집|신청|접수|공모)\s*기간")
_Y = r"(?:(20\d\d)|'(\d\d))"
_START = re.compile(_Y + r"\s*[.\-년/]\s*(\d{1,2})\s*[.\-월/]\s*(\d{1,2})")
_END = re.compile(r"[\s.일]*(?:\([^)]*\))?\s*[~\-–]\s*" + _Y + r"?\s*(?:[.\-년/]\s*)?(?:(\d{1,2})\s*[.\-월/]\s*)?(\d{1,2})")
_OPEN = re.compile(r"(?:상시|연중|수시)\s*(?:접수|신청|모집|운영)|(?:접수|신청|모집)[^\n]{0,10}(?:상시|연중|수시)")
_MD = r"(\d{1,2})\s*[.월]\s*(\d{1,2})"
_MD_RANGE = re.compile(r"(?<![\d.])" + _MD + r"\s*[.일]?\s*(?:\([^)]*\))?\s*[~\-–]\s*(?:(\d{1,2})\s*[.월]\s*)?(\d{1,2})(?!\d)")
_MONTH_RANGE = re.compile(r"매년\s*(?:초|\d회)?\s*\(?(\d{1,2})\s*월(?:\s*(?:초|중|말))?\s*(?:[~\-–]\s*(\d{1,2})\s*월)?")
_NOT_APPLY = re.compile(r"지급|기준|출생")
_TEXT_KEYS = ("apply_text", "summary_text", "criteria_text", "target_text")
_KST = timezone(timedelta(hours=9), "Asia/Seoul")


def korea_today() -> date:
    return datetime.now(_KST).date()


def parse_reference_date(value) -> date:
    """평가 기준일의 형식과 실제 달력 날짜를 검증함."""
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, str) and re.fullmatch(r"[0-9]{8}|[0-9]{4}-[0-9]{2}-[0-9]{2}", value):
        try:
            return date.fromisoformat(f"{value[:4]}-{value[4:6]}-{value[6:]}" if len(value) == 8 else value)
        except ValueError:
            pass
    raise ValueError(f"기준일(as_of)은 유효한 YYYY-MM-DD 또는 YYYYMMDD 날짜여야 합니다: {value!r}")


def _s(v) -> str:
    return v if isinstance(v, str) else ""


def to_date(v) -> date | None:
    """date, datetime, 'YYYYMMDD', 'YYYY-MM-DD', 정수·실수형 YYYYMMDD를 date로 바꿈. 실패하면 None."""
    if isinstance(v, datetime):
        return v.date()
    if isinstance(v, date):
        return v
    if isinstance(v, float) and v.is_integer():
        v = int(v)
    try:
        return parse_reference_date(str(v).strip())
    except ValueError:
        return None


def _year(full, short):
    return int(full) if full else 2000 + int(short)


def _windows(text: str) -> list[tuple[date, date, str]]:
    out = []
    for km in _KEY.finditer(text):
        seg = text[km.end(): km.end() + 80]
        sm = _START.search(seg)
        if not sm:
            continue
        try:
            sy, smo, sd = _year(sm.group(1), sm.group(2)), int(sm.group(3)), int(sm.group(4))
            start = date(sy, smo, sd)
            em = _END.match(seg, sm.end())
            if not em:
                continue
            has_year = bool(em.group(1) or em.group(2))
            ey = _year(em.group(1), em.group(2)) if has_year else sy
            emo = int(em.group(3)) if em.group(3) else smo
            ed = int(em.group(4))
            if not has_year and (emo, ed) < (smo, sd):
                ey += 1
            end = date(ey, emo, ed)
        except ValueError:
            continue
        if end >= start:
            out.append((start, end, seg[: em.end()].strip()))
    return out


def _recurring(texts: dict[str, str]) -> list[tuple[int, int, int, int, str]]:
    """연도 없는 반복 기간을 (시작월, 시작일, 종료월, 종료일, 근거)로 추정함."""
    out = []
    for k, text in texts.items():
        for km in _KEY.finditer(text):
            seg = text[km.end(): km.end() + 60]
            if re.search(r"20\d\d|'\d\d", seg[:25]):
                continue
            m = _MD_RANGE.search(seg)
            if m:
                a, b = int(m.group(1)), int(m.group(2))
                c, d = (int(m.group(3)) if m.group(3) else a), int(m.group(4))
                if 1 <= a <= 12 and 1 <= c <= 12 and 1 <= b <= 31 and 1 <= d <= 31:
                    out.append((a, b, c, d, seg[: m.end()].strip()))
                    continue
            m = _MONTH_RANGE.search(seg)
            if m:
                a = int(m.group(1))
                c = int(m.group(2)) if m.group(2) else a
                if 1 <= a <= 12 and 1 <= c <= 12:
                    out.append((a, 1, c, 31, seg[: m.end()].strip()))
        if k == "apply_text":
            for m in _MONTH_RANGE.finditer(text):
                if _NOT_APPLY.search(text[max(0, m.start() - 8): m.start()] + text[m.end(): m.end() + 6]):
                    continue
                a = int(m.group(1))
                c = int(m.group(2)) if m.group(2) else a
                if 1 <= a <= 12 and 1 <= c <= 12:
                    out.append((a, 1, c, 31, m.group(0).strip()))
    return out


def _recurring_dates(item, today: date) -> tuple[date, date] | None:
    """오늘을 포함하는 반복 기간, 없으면 가장 가까운 다음 기간을 돌려줌."""
    a, b, c, d, _ = item
    cands = []
    for y in (today.year - 1, today.year, today.year + 1):
        ey = y if (c, d) >= (a, b) else y + 1
        try:
            cands.append((date(y, a, min(b, calendar.monthrange(y, a)[1])),
                          date(ey, c, min(d, calendar.monthrange(ey, c)[1]))))
        except ValueError:
            continue
    for s, e in cands:
        if s <= today <= e:
            return s, e
    return min((w for w in cands if w[0] > today), key=lambda w: w[0], default=None)


def parse_period(svc: dict) -> dict:
    """{'end_date': date|None(종료일 없음·결측), 'windows': [(시작, 종료, 근거)], 'open_ended': 근거|None, 'recurring': 연도 없는 반복 추정 [(시작월, 시작일, 종료월, 종료일, 근거)](windows가 없을 때만)}."""
    end = to_date(svc.get("enfc_end_ymd"))
    if end is not None and end.year >= 9999:
        end = None
    wins, open_ev, texts = [], None, {}
    for k in _TEXT_KEYS:
        t = _s(svc.get(k))
        texts[k] = t
        wins += _windows(t)
        m = _OPEN.search(t)
        if m and open_ev is None:
            open_ev = m.group(0)
    rec = [] if wins else _recurring(texts)
    return {"end_date": end, "windows": wins, "open_ended": open_ev, "recurring": rec}


def check_period(svc: dict, profile: dict) -> tuple[str, str]:
    today = parse_reference_date(profile["today"]) if "today" in profile else korea_today()
    info = parse_period(svc)
    end = info["end_date"]
    if end is not None and end < today:
        return EXCLUDE, f"사업 종료 ({end.isoformat()})"
    if info["open_ended"]:
        return PASS, f"상시·연중·수시 모집 표현 ({info['open_ended']})"
    wins = info["windows"]
    if wins:
        for w in wins:
            if w[0] <= today <= w[1]:
                return PASS, f"모집 중 ({w[0].isoformat()} ~ {w[1].isoformat()})"
        future = [w for w in wins if w[0] > today]
        if future:
            return CHECK, f"모집 시작 전, 시작일 {min(w[0] for w in future).isoformat()}"
        last = max(wins, key=lambda w: w[1])
        return CHECK, f"원문 모집기간 {last[1].isoformat()} 종료, 재모집 여부 확인 필요"
    if info["recurring"]:
        spans = [(_recurring_dates(it, today), it) for it in info["recurring"]]
        spans = [(span, it) for span, it in spans if span is not None]
        for (s0, e0), it in spans:
            if s0 <= today <= e0:
                return CHECK, f"매년 반복 추정 기간에 해당 ({it[4]}), 실제 모집 여부 확인 필요"
        if not spans:
            return CHECK, "매년 반복 추정, 실제 모집일 확인 필요"
        (s0, _), it = min(spans, key=lambda x: x[0][0])
        return CHECK, f"매년 반복 추정, 예상 다음 시작일 {s0.isoformat()} ({it[4]})"
    return CHECK, PERIOD_NO_INFO_REASON
