"""접근 방식(신청, 이용 경로) 대조. 사용자가 피하거나 원하는 접근과 서비스의 신청, 이용 경로를 맞춤.

원칙 (src/recommend/eligibility.py와 같은 의미, ai/는 src/를 import하지 않으므로 다시 구현함)
- 부담 != 불가. 부담은 어떤 경우에도 제외하지 않고 감점과 확인 필요로만 반영함
- 제외는 필수 요건이 확정되었고 사용자가 충족할 수 없을 때만 함. 경로 목록만 있는 것은 확정이 아님
- 서비스 값이 N이거나 누락이면 확인 필요임. 조건 없음으로 보지 않음
- 신청(apply)과 이용(use)을 나눠 봄

방문 필수 확정 기준 (제외 가능)
- 이용: use_outing >= 2이고 use_outing_evidence가 있으며 [혼재, 선택, 의료, 확인 필요] 검수 표시가 없음
- 신청: apply_outing_basis가 visit_required_text 또는 interview_onsite
- visit_only_listed(방문 경로만 안내됨)는 확정이 아니므로 확인 필요
방문 불가 제외 조건 (대체 경로 확인)
- 방문 필수가 확정된 범위(신청 또는 이용)마다 그 범위의 대체 경로(alternative_routes)가 없을 때만 제외함
- 신청 대체 경로는 이용 방문을 해소하지 못하고, 이용 대체 경로는 신청 방문을 해소하지 못함
- 대체 경로가 하나라도 있으면 확인 필요 + 감점(visit_unable_alternative). 제공자가 찾아오는 방문 서비스는 사용자 방문 필수가 아님
전화 필수 확정 기준: apply_cat이 전화뿐이고 온라인, 방문, 우편, 팩스 경로가 없으며 apply_text에 전화로만 같은 한정 표현이 있음.
전화뿐이라는 분류값만으로는 확정하지 않음(확인 필요).

profile 입력
- profile["access"]: [{polarity: avoid|want|inferred, channel, level, scope, mode, pattern_id}] (confirm.session의 access)
- access가 없으면 profile["constraints"]([{channel, level}] 또는 {channel: {level}})를 avoid로 봄
- inferred는 사용하지 않음(사용자가 밝힌 정보만 반영)
"""
import re

import pandas as pd

from rules import overlay

PASS, CHECK, EXCLUDE = "통과", "확인 필요", "제외"

# 감점 가중치. 모두 잠정값이며 근거 데이터가 없음. config로 덮어쓸 수 있도록 모듈 변수로 둠.
DEFAULT_WEIGHTS = {
    "visit_burden_confirmed": 1.0,    # 방문 부담 + 방문 필수 확정
    "visit_burden_likely": 0.5,       # 방문 부담 + 방문 경로만 안내됨(확정 아님)
    "visit_unable_unconfirmed": 1.0,  # 방문 불가 + 방문 필수 미확정이나 방문 경로만 안내됨(제외하지 않음)
    "visit_unable_alternative": 1.0,  # 방문 불가 + 방문 필수 확정이나 같은 범위에 대체 경로가 있음(제외하지 않음)
    "phone_burden_only": 0.5,        # 전화 부담 + 전화가 유일한 경로
    "phone_unable_unconfirmed": 1.0,  # 전화 불가 + 전화뿐이나 필수 미확정(제외하지 않음)
    "want_online_match": -0.5,        # 온라인 접근을 원하고 서비스에 비대면 경로가 있음(음수 = 가점)
    "visit_unable_verified_use": -1.0,
}

VISIT_CONFIRMED_BASIS = {"visit_required_text", "interview_onsite"}
ONLINE_TOKENS = {"인터넷", "모바일", "모바일앱", "E-mail"}
OTHER_REMOTE_TOKENS = {"우편", "팩스"}
REVIEW_MARKERS = ("[혼재", "[선택", "[의료", "확인 필요")
# 대체 경로 근거 (잠정 사전). 놓치면 제외 쪽으로 기울므로 넓게 잡고, 걸린 행은 제외 대신 확인 필요가 됨
APPLY_ALT_TOKENS = ONLINE_TOKENS | OTHER_REMOTE_TOKENS | {"전화"}
APPLY_ALT_RE = re.compile(
    r"대리\s*(?:신청|접수|인)|대리인|가족.{0,8}신청|신청.{0,8}가족|직권|담당\s*공무원|찾아가는|"
    r"온라인|인터넷|복지로|정부\s*24|우편|팩스|이메일|e-?mail|전화\s*(?:신청|접수)|비대면|모바일", re.I)
USE_ALT_RE = re.compile(
    r"비대면|온라인|화상|원격|재택|인터넷|찾아가는|가정\s*방문|방문\s*(?:서비스|재활|간호|요양|목욕|돌봄|급식|교육|지도|상담|진료|검진)|"
    r"(?:사업장|가정|자택|집|거주지|현장)\s*(?:을|를)?\s*(?:직접\s*)?방문|"
    r"출장|순회|이동\s*(?:식|형|버스|검진)|택배|배송|우편|대리\s*(?:수령|이용)|전화\s*상담")
ONLY_PHONE_RE = re.compile(r"(?:전화|유선)\s*(?:으로|로)?\s*만(?!\w)|전화\s*(?:신청|접수)\s*만")
ONLINE_WANT_RE = re.compile(r"온라인|비대면|인터넷|모바일|앱|online", re.I)
REMOTE_USE_RE = re.compile(
    r"(?:전화|유선|온라인|비대면|화상|원격)\s*(?:으로|로)?\s*"
    r"(?:상담|진료|교육|수업|프로그램|훈련)(?:을|를)?\s*(?:무료로\s*)?(?:제공|운영|실시|가능|진행)"
)
REMOTE_USE_UNCLEAR = re.compile(r"문의|예약|접수|(?:상담|교육)\s*신청|불가|않|미제공|중단|예정")


def _val(svc, k):
    v = svc.get(k)
    return None if v is None or (isinstance(v, float) and pd.isna(v)) else v


def _int(v):
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return None


def route_info(svc) -> dict:
    """서비스의 신청, 이용 방문 정보를 한 번에 정리함."""
    toks = {t.strip() for t in str(_val(svc, "apply_cat") or "").split(",") if t.strip()}
    basis = _val(svc, "apply_outing_basis")
    use_v = _int(_val(svc, "use_outing"))
    ev = str(_val(svc, "use_outing_evidence") or "").strip()
    online = bool(toks & ONLINE_TOKENS) or _val(svc, "online_bokjiro") == "Y"
    return {
        "tokens": toks,
        "online": online,
        "phone": "전화" in toks,
        "use_visit_confirmed": use_v is not None and use_v >= 2 and bool(ev) and not any(m in ev for m in REVIEW_MARKERS),
        "use_visit_unconfirmed": use_v is not None and use_v >= 2,   # 값은 2 이상이나 근거 부족 또는 검수 표시
        "use_unknown": use_v is None,
        "apply_visit_confirmed": basis in VISIT_CONFIRMED_BASIS,
        "apply_visit_listed": basis == "visit_only_listed",
        "apply_unknown": basis in (None, "no_apply_info", "interview_unknown"),
        "phone_only": toks == {"전화"} and not online,
    }


def _hits(rx, text, label):
    return [f"{label}:{m}" for m in dict.fromkeys(x.group(0) for x in rx.finditer(text))]


def alternative_routes(svc) -> dict:
    """범위(신청, 이용)별 방문 외 대체 경로 근거 목록. 빈 목록은 근거를 찾지 못한 것이며 없다는 확정은 아님."""
    toks = {t.strip() for t in str(_val(svc, "apply_cat") or "").split(",") if t.strip()}
    apply_ev = [f"apply_cat:{t}" for t in sorted(toks & APPLY_ALT_TOKENS)]
    if _val(svc, "online_bokjiro") == "Y":
        apply_ev.append("online_bokjiro:Y")
    if _val(svc, "apply_outing_basis") == "remote_route":
        apply_ev.append("apply_outing_basis:remote_route")
    apply_ev += _hits(APPLY_ALT_RE, str(_val(svc, "apply_text") or ""), "apply_text")
    use_text = " ".join(str(_val(svc, k) or "") for k in ("use_outing_evidence", "benefit_text", "summary_text"))
    use_ev = _hits(USE_ALT_RE, use_text, "use_text")
    if str(_val(svc, "remote_complete")) == "2":
        use_ev.append("remote_complete:2")
    return {"신청": apply_ev, "이용": use_ev}


def phone_required_confirmed(svc) -> bool:
    return route_info(svc)["phone_only"] and bool(ONLY_PHONE_RE.search(str(_val(svc, "apply_text") or "")))


def _avoid_items(profile) -> list:
    acc = profile.get("access")
    if acc:
        out = [a for a in acc if isinstance(a, dict) and a.get("polarity") == "avoid"]
    else:
        cons = profile.get("constraints") or []
        if isinstance(cons, dict):
            cons = [{"channel": k, **(v if isinstance(v, dict) else {"level": v})} for k, v in cons.items()]
        out = [c for c in cons if isinstance(c, dict)]
    return [a for a in out if a.get("channel") in ("방문", "전화") and a.get("level") in ("부담", "불가")]


def _want_items(profile) -> list:
    return [a for a in profile.get("access") or [] if isinstance(a, dict) and a.get("polarity") == "want"]


def _scopes(item):
    s = item.get("scope")
    return {s} if s in ("신청", "이용") else {"신청", "이용"}


def _confirmed_scopes(ri, scopes) -> list:
    out = []
    if "이용" in scopes and ri["use_visit_confirmed"]:
        out.append("이용")
    if "신청" in scopes and ri["apply_visit_confirmed"]:
        out.append("신청")
    return out


def _visit_state(ri, scopes) -> tuple[str, str]:
    """방문 필요 상태(confirmed, likely, unknown, none)와 이유."""
    if "이용" in scopes and ri["use_visit_confirmed"]:
        return "confirmed", "이용에 방문이 필요한 것으로 확인됨"
    if "신청" in scopes and ri["apply_visit_confirmed"]:
        return "confirmed", "신청에 방문이 필수인 것으로 확인됨"
    if "신청" in scopes and ri["apply_visit_listed"]:
        return "likely", "방문 신청 경로만 안내됨, 다른 경로가 없다는 확인은 아님"
    if "이용" in scopes and ri["use_visit_unconfirmed"]:
        return "likely", "이용 방문 값이 있으나 근거 부족 또는 검수 표시"
    if ("신청" in scopes and ri["apply_unknown"]) or ("이용" in scopes and ri["use_unknown"]):
        return "unknown", "신청 또는 이용 외출 정보 일부 없음"
    return "none", "방문 필수 근거 없음"


# 방문 필수·대체 불가를 원문이 명시하는 표현. 이 표현이 있을 때만 방문 불가 사용자를 제외함
MANDATORY_VISIT_RE = re.compile(
    r"반드시\s*(?:본인이?\s*)?(?:직접\s*)?(?:방문|내방|출석)|본인이?\s*직접\s*(?:방문|내방|출석)|직접\s*방문하여야|"
    r"방문\s*(?:신청|접수)만|대리\s*(?:신청|접수|수령)\s*(?:불가|안\s*됨)|내방\s*필수|방문\s*필수"
)
_MANDATORY_COLS = ("apply_text", "apply_text_raw", "use_outing_evidence", "benefit_text", "summary_text",
                   "target_text", "criteria_text")


def mandatory_visit_explicit(svc) -> bool:
    """원문에 방문 필수·대체 불가가 명시됐는지. 2026-11 마스터 기준 해당 0건."""
    text = " ".join(str(_val(svc, c) or "") for c in _MANDATORY_COLS)
    return bool(MANDATORY_VISIT_RE.search(text))


def _visit_unable_from_review(ov, scopes, w):
    """검수된 접근 조건으로 방문 불가 판정. 범위별로 제외, 통과, 확인 필요를 정하고 가장 강한 판정을 씀."""
    parts = []   # (status, text)
    if "신청" in scopes:
        if "서비스별 상이" in (ov["apply_visit"], ov["apply_alt"]):
            parts.append((CHECK, "신청 방식 서비스별 상이, 세부 서비스 확인 필요"))
        elif ov["apply_visit"] == "필수" and ov["apply_alt"] == "없음 명시":
            parts.append((EXCLUDE, "신청 방문 필수, 대체 경로 없음 명시"))
        elif ov["apply_visit"] == "불필요":
            parts.append((PASS, "신청 방문 불필요"))
        else:
            parts.append((CHECK, f"신청 방문 {ov['apply_visit']}, 대체 경로 {ov['apply_alt']}"))
    if "이용" in scopes:
        # 비용 수령 방식과 실제 활동을 분리한 경우, 실제 활동이 불명인 상태를 통과로 만들지 않는다.
        use_visit = ov.get("activity_visit") or ov["use_visit"]
        if "서비스별 상이" in (use_visit, ov["use_alt"]):
            parts.append((CHECK, "이용 방식 서비스별 상이, 세부 서비스 확인 필요"))
        elif use_visit == "사용자 방문 필수" and ov["use_alt"] == "없음 명시":
            parts.append((EXCLUDE, "이용 사용자 방문 필수, 대체 경로 없음 명시"))
        elif use_visit in ("담당자 방문", "불필요"):
            parts.append((PASS, f"이용 {use_visit}"))
        else:
            parts.append((CHECK, f"이용 방문 {use_visit}, 대체 경로 {ov['use_alt']}"))
    for rank in (EXCLUDE, CHECK):
        hit = [t for st, t in parts if st == rank]
        if hit:
            break
    else:
        rank, hit = PASS, [t for _, t in parts]
    why = "검수 반영: " + ", ".join(hit) + (f" ({ov['reason']})" if ov["reason"] else "")
    return rank, (w["visit_unable_unconfirmed"] if rank == CHECK else 0.0), why


def _eval_item(item, ri, svc, w):
    """회피 항목 하나 -> (status, penalty, reason)."""
    ch, lv, scopes = item["channel"], item["level"], _scopes(item)
    if ch == "방문":
        ov = overlay.get_access(_val(svc, "servId"))
        if ov:
            status, penalty, why = _visit_unable_from_review(ov, scopes, w)
            details = []
            if "신청" in scopes and ov.get("apply_route_note"):
                details.append("신청 안내: " + ov["apply_route_note"])
            if "이용" in scopes and ov.get("receipt_visit"):
                details.append("비용 수령: " + ov["receipt_visit"] + ", 실제 활동: " + (ov.get("activity_visit") or "불명"))
            if details:
                why += "; " + "; ".join(details)
            if lv == "부담" and status == EXCLUDE:
                return CHECK, w["visit_burden_confirmed"], why + ", 방문 부담"
            return status, penalty, why
    if ch == "방문":
        st, why = _visit_state(ri, scopes)
        if lv == "불가":
            if st == "confirmed":
                alts = alternative_routes(svc)
                conf = _confirmed_scopes(ri, scopes)
                open_scopes = [sc for sc in conf if not alts[sc]]
                if open_scopes and mandatory_visit_explicit(svc):
                    return EXCLUDE, 0.0, f"{why}, 원문에 방문 필수·대체 불가 명시, 사용자는 방문 불가"
                if open_scopes:
                    # 대체 경로를 찾지 못한 것과 대체 경로가 없는 것은 다름. 명시 근거 없으면 제외하지 않음
                    return CHECK, w["visit_unable_unconfirmed"], (
                        f"{why}, {'/'.join(open_scopes)} 대체 경로 근거 못 찾음(없다는 명시 없음), 사용자는 방문 불가. "
                        "온라인·대리 경로 확인 필요")
                ev = "; ".join(f"{sc} {', '.join(alts[sc][:3])}" for sc in conf)
                return CHECK, w["visit_unable_alternative"], f"{why}. 같은 범위에 대체 경로 근거 있음({ev}), 사용자는 방문 불가, 확인 필요"
            if st == "likely":
                return CHECK, w["visit_unable_unconfirmed"], f"{why}. 사용자는 방문 불가, 온라인 또는 대리 경로 확인 필요"
            if st == "unknown":
                return CHECK, 0.0, f"{why}. 사용자는 방문 불가, 확인 필요"
            return PASS, 0.0, why
        if st == "confirmed":
            return CHECK, w["visit_burden_confirmed"], f"{why}, 사용자는 방문 부담. 부담 조건으로 안내"
        if st == "likely":
            return CHECK, w["visit_burden_likely"], f"{why}, 사용자는 방문 부담"
        if st == "unknown":
            return CHECK, 0.0, f"{why}, 사용자는 방문 부담"
        return PASS, 0.0, why
    # 전화는 신청 경로에만 해당함(이용 중 전화 여부는 필드가 없음)
    if "신청" not in scopes:
        return PASS, 0.0, "이용 단계 전화 정보 없음"
    if not ri["tokens"]:
        return CHECK, 0.0, "신청 방법 분류 없음"
    if not ri["phone_only"]:
        return PASS, 0.0, "전화 외 신청 경로가 있음" if ri["phone"] else "전화 신청 경로 아님"
    if lv == "불가":
        if phone_required_confirmed(svc):
            return EXCLUDE, 0.0, "전화로만 신청 가능한 것으로 확인됨, 사용자는 전화 불가"
        return CHECK, w["phone_unable_unconfirmed"], "신청 방법이 전화뿐으로 분류됨, 필수 여부 미확정. 사용자는 전화 불가"
    return CHECK, w["phone_burden_only"], "신청 방법이 전화뿐으로 분류됨, 사용자는 전화 부담"


def check_access(svc, profile):
    """(status, reason). 제외는 방문 필수 또는 전화 필수가 확정되고 사용자가 불가일 때만."""
    items = _avoid_items(profile)
    if not items:
        ov = overlay.get_access(_val(svc, "servId"))
        if ov and ov.get("apply_route_note"):
            why = "신청 안내: " + ov["apply_route_note"]
            if ov.get("receipt_visit"):
                why += "; 비용 수령: " + ov["receipt_visit"] + ", 실제 활동: " + (ov.get("activity_visit") or "불명")
            return PASS, why
        return PASS, "회피 접근 정보 없음"
    ri = route_info(svc)
    results = [(_eval_item(it, ri, svc, DEFAULT_WEIGHTS), it) for it in items]
    for rank in (EXCLUDE, CHECK):
        hit = [(r, it) for r, it in results if r[0] == rank]
        if hit:
            return rank, "; ".join(f"{it['channel']} {it['level']}: {r[2]}" for r, it in hit)
    reviewed = [r[2] for r, _ in results if r[2].startswith("검수 반영")]   # 검수로 통과한 근거는 남김
    return PASS, "; ".join(reviewed) if reviewed else "접근 방식 충돌 확인되지 않음"


def access_penalty(svc, profile):
    """(penalty, reason). 양수 = 감점, 음수 = 가점. 제외 판정은 하지 않음."""
    ri = route_info(svc)
    total, reasons = 0.0, []
    avoid = _avoid_items(profile)
    for it in avoid:
        _, pen, why = _eval_item(it, ri, svc, DEFAULT_WEIGHTS)
        if pen:
            total += pen
            reasons.append(f"{it['channel']} {it['level']}: {why}")
    blocked_use = {it["channel"] for it in avoid if it["level"] == "불가" and "이용" in _scopes(it)}
    if profile.get("prioritize_verified_use") and blocked_use == {"방문"}:
        reviewed = overlay.get_access(_val(svc, "servId"))
        if reviewed:
            verified_use = (reviewed.get("activity_visit") or reviewed["use_visit"]) in ("불필요", "담당자 방문")
        else:
            passage = str(_val(svc, "benefit_text_raw") or _val(svc, "benefit_text") or "")
            verified_use = not ri["use_visit_unconfirmed"] and (
                _int(_val(svc, "remote_complete")) == 2 or any(
                    REMOTE_USE_RE.search(part) and not REMOTE_USE_UNCLEAR.search(part)
                    for part in re.split(r"[.。;\r\n]+", passage)
                )
            )
        if verified_use:
            total += DEFAULT_WEIGHTS["visit_unable_verified_use"]
            reasons.append("방문 불가: 비방문 이용 경로 확인됨")
    for it in _want_items(profile):
        if not ONLINE_WANT_RE.search(f"{it.get('mode') or ''} {it.get('pattern_id') or ''}"):
            continue
        scopes = _scopes(it)
        if ("신청" in scopes and ri["online"]) or ("이용" in scopes and str(_val(svc, "remote_complete")) == "2"):
            total += DEFAULT_WEIGHTS["want_online_match"]
            reasons.append("온라인 접근 희망, 서비스에 비대면 경로 있음")
    return total, "; ".join(reasons) if reasons else "접근 방식 조정 없음"
