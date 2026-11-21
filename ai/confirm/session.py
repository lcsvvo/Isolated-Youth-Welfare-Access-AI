"""2단계 사용자 확인 세션 (형식 stage2-v1).

- original_text, extracted(extract-v1)는 세션을 만들 때 복사해 두고 바꾸지 않는다.
  normalize()·postprocess()를 다시 돌리지 않는다.
- actions는 추가만 한다. confirmed_profile·provenance·conflicts·access·need_changes는
  extracted + profile 인자 + actions를 처음부터 다시 적용해 만든다(derive).
- 우선순위: 이번 세션 사용자 값 > profile 인자 > 추출.
- 사용자 값은 normalize를 쓰지 않고 직접 검증한다. 사용자 입력은 원문 근거(evidence)로 기록하지 않는다.
"""
from __future__ import annotations

import copy
import re
from datetime import datetime, timezone

from extract.schema import ACTIVITY_RANGES, AGE_RANGE, CONSTRAINT_CHANNELS, CONSTRAINT_LEVELS, SIDO
from rules.normalize import MERGED_SIDO, norm_age, valid_sgg_for

SCHEMA_VERSION = "stage2-v1"
STATUSES = ("crisis_stop", "awaiting_confirm", "awaiting_answer", "explore_only", "ready")
CONFIRM_ACTIONS = ("confirm_need", "edit_need", "delete_need", "add_need", "set_main",
                   "set_activity", "set_constraint", "set_age", "set_region", "set_region_sgg",
                   "finish_confirm")
# 질문 답, 확인 화면 메모, 자유 입력 위기 중단 기록
OTHER_ACTIONS = ("answer", "memo", "crisis_stop")
CONSTRAINT_VALUES = ("없음",) + tuple(CONSTRAINT_LEVELS)
PROFILE_KEYS = ("activity_range", "constraints", "age", "region_sido")
UNKNOWN_ACTIVITY = "판단 불가"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ── 사용자 값 검증 (normalize 미사용) ─────────────────────────────────
def valid_age(v):
    """10~99 정수만. 숫자 입력칸의 '23' 같은 숫자 문자열은 받는다. '23살'·bool·실수는 거부."""
    if isinstance(v, bool):
        return None
    if isinstance(v, str) and re.fullmatch(r"\d{1,2}", v.strip()):
        v = int(v.strip())
    return v if isinstance(v, int) and AGE_RANGE[0] <= v <= AGE_RANGE[1] else None


def valid_region(v):
    """시·도 약칭 또는 통합 시·도 이름을 확인한다."""
    return v if isinstance(v, str) and (v in SIDO or v in MERGED_SIDO) else None


def valid_activity(v):
    return v if v in ACTIVITY_RANGES else None


# ── 세션 생성 ─────────────────────────────────────────────────────
def new_session(text: str, crisis: dict, extracted: dict | None, profile: dict | None = None) -> dict:
    """위기 확인·추출 결과로 세션을 만든다. 위기면 extracted를 두지 않는다(1·2단계 미실행)."""
    s = {
        "schema_version": SCHEMA_VERSION,
        "status": "awaiting_confirm",
        "confirmation": "none",
        "original_text": text,
        "crisis": copy.deepcopy(crisis),
        "extracted": None,
        "profile": copy.deepcopy(profile) if profile else {},
        "actions": [],
        "questions": [],
        "confirmed_profile": {},
        "provenance": {},
        "access": [],
        "conflicts": [],
        "need_changes": [],
        "queries": {},
        "query_expansion": {},
    }
    if crisis.get("crisis"):
        s["status"] = "crisis_stop"
        return s
    s["extracted"] = copy.deepcopy(extracted)
    return derive(s)


def add_action(session: dict, type_: str, target=None, value=None, at: str | None = None, **extra) -> dict:
    """행동을 추가하고 파생 필드를 다시 만든다. 입력 세션은 바꾸지 않고 새 세션을 돌려준다."""
    if type_ not in CONFIRM_ACTIONS + OTHER_ACTIONS:
        raise ValueError(f"알 수 없는 action type: {type_}")
    s = copy.deepcopy(session)
    s["actions"].append({"type": type_, "target": target, "value": value, "at": at or now_iso(), **extra})
    if type_ == "finish_confirm":
        s["confirmation"] = "user"
    return derive(s) if s.get("extracted") is not None else s


def apply_confirm_action(session: dict, action: dict, domains: list[str]) -> dict:
    """확인 화면 조작. awaiting_confirm 상태에서만 받는다. 틀린 값은 기록만 하고 반영하지 않는다."""
    if session["status"] != "awaiting_confirm":
        raise ValueError(f"확인 단계가 아님: {session['status']}")
    t = action.get("type")
    if t not in CONFIRM_ACTIONS:
        raise ValueError(f"확인 화면 조작이 아님: {t}")
    target, value = action.get("target"), action.get("value")
    ok = _valid_confirm(t, target, value, domains)
    return add_action(session, t, target, value if ok else None, action.get("at"), accepted=ok)


def _valid_confirm(t, target, value, domains) -> bool:
    if t in ("confirm_need", "delete_need", "set_main"):
        return target in domains
    if t == "add_need":
        return value in domains
    if t == "edit_need":
        return target in domains and value in domains and target != value
    if t == "set_activity":
        return valid_activity(value) is not None
    if t == "set_constraint":
        return target in CONSTRAINT_CHANNELS and value in CONSTRAINT_VALUES
    if t == "set_age":
        return valid_age(value) is not None
    if t == "set_region":
        return valid_region(value) is not None
    if t == "set_region_sgg":
        return value is None or (isinstance(value, str) and bool(value.strip()))
    return True  # finish_confirm


# ── 파생 필드 계산 ─────────────────────────────────────────────────
def _known(field, v) -> bool:
    if field == "activity_range":
        return v not in (None, UNKNOWN_ACTIVITY)
    return v is not None


def derive(s: dict) -> dict:
    ext = s["extracted"]
    needs = [{"domain": n["domain"], "level": n["level"], "evidence": n.get("evidence"),
              "status": n.get("status"), "origin": "extract", "prov": "extract",
              "usable": n.get("status") == "명시"} for n in ext.get("needs", [])]
    changes, conflicts = [], []
    fields = {
        "activity_range": [ext.get("activity_range", UNKNOWN_ACTIVITY), "extract"],
        "age": [ext.get("age"), "extract"],
        "region_sido": [ext.get("region_sido"), "extract"],
        "region_sgg": [None, "extract"],
    }
    cons = {c["channel"]: {"channel": c["channel"], "level": c["level"], "evidence": c.get("evidence"),
                           "source": "extract"} for c in ext.get("constraints", [])}

    def set_field(name, value, prov):
        cur, cur_prov = fields[name]
        if _known(name, cur) and cur != value:
            conflicts.append({"field": name, "before": cur, "after": value, "source": prov,
                              "before_source": cur_prov})
        fields[name] = [value, prov]

    def set_cons(ch, level, prov):
        cur = cons.get(ch)
        if cur and cur["level"] != level:
            conflicts.append({"field": f"constraints:{ch}", "before": cur["level"], "after": level or "없음",
                              "source": prov, "before_source": cur["source"]})
        if level is None:
            cons.pop(ch, None)
        else:
            cons[ch] = {"channel": ch, "level": level, "evidence": None, "source": prov}

    # profile 인자 (이전 세션 값). 형식이 틀린 값은 버린다.
    prof = s.get("profile") or {}
    if valid_activity(prof.get("activity_range")) and prof["activity_range"] != UNKNOWN_ACTIVITY:
        set_field("activity_range", prof["activity_range"], "profile")
    if valid_age(prof.get("age")) is not None:
        set_field("age", valid_age(prof["age"]), "profile")
    if valid_region(prof.get("region_sido")):
        set_field("region_sido", prof["region_sido"], "profile")
    for c in prof.get("constraints") or []:
        if isinstance(c, dict) and c.get("channel") in CONSTRAINT_CHANNELS and c.get("level") in CONSTRAINT_LEVELS:
            set_cons(c["channel"], c["level"], "profile")

    def find(d):
        return next((e for e in needs if e["domain"] == d), None)

    def remove(e, how, to=None):
        needs.remove(e)
        changes.append({"domain": e["domain"], "evidence": e["evidence"], "how": how, "to": to,
                        "origin": e["origin"]})
        if e["origin"] == "extract":
            conflicts.append({"field": f"needs:{e['domain']}", "before": e["status"],
                              "after": "삭제" if how == "delete" else f"변경:{to}", "source": "user_edit",
                              "before_source": "extract"})

    def add(d, prov, level=None):
        e = find(d)
        if e:
            e["usable"], e["prov"] = True, prov
            return
        has_main = any(x["level"] == "main" and x["usable"] for x in needs)
        needs.append({"domain": d, "level": level or ("sub" if has_main else "main"), "evidence": None,
                      "status": None, "origin": "user", "prov": prov, "usable": True})

    for a in s["actions"]:
        t, tg, v = a["type"], a.get("target"), a.get("value")
        if a.get("accepted") is False:
            continue
        if t == "confirm_need" and find(tg):
            find(tg).update(usable=True, prov="user_confirm")
        elif t == "delete_need" and find(tg):
            remove(find(tg), "delete")
        elif t == "edit_need" and find(tg):
            old = find(tg)
            remove(old, "edit", v)
            add(v, "user_edit", old["level"])
        elif t == "add_need" and v:
            add(v, "user_edit")
        elif t == "set_main" and find(tg):
            for x in needs:
                if x["level"] == "main":
                    x["level"] = "sub"
            e = find(tg)
            e["level"] = "main"
            if not e["usable"]:
                e.update(usable=True, prov="user_confirm")
        elif t == "set_activity" and valid_activity(v):
            set_field("activity_range", v, "user_edit")
        elif t == "set_constraint" and tg in CONSTRAINT_CHANNELS and v in CONSTRAINT_VALUES:
            set_cons(tg, None if v == "없음" else v, "user_edit")
        elif t == "set_age" and valid_age(v) is not None:
            set_field("age", valid_age(v), "user_edit")
        elif t == "set_region" and valid_region(v):
            set_field("region_sido", v, "user_edit")
        elif t == "set_region_sgg":
            # 확정 시도에 실제로 있는 시군구만 받음. 시도가 없거나 바뀌어 짝이 안 맞으면 반영하지 않음
            sg = valid_sgg_for(fields["region_sido"][0], v)
            if sg or v is None:
                set_field("region_sgg", sg, "user_edit")
        elif t == "answer" and a.get("answer_type") == "선택":
            if tg == "need":
                for d in (v if isinstance(v, list) else [v]):
                    add(d, "question")
            elif tg == "region_sido" and valid_region(v):
                set_field("region_sido", v, "question")
            elif tg == "activity_range" and valid_activity(v):
                set_field("activity_range", v, "question")
            elif tg == "age" and valid_age(v) is not None:
                set_field("age", valid_age(v), "question")

    # 시도가 없거나 바뀌어 시군구가 그 시도에 없으면 비움
    if fields["region_sgg"][0] is not None and valid_sgg_for(fields["region_sido"][0], fields["region_sgg"][0]) is None:
        fields["region_sgg"] = [None, "extract"]

    # 검색 사용 영역: (명시 + 삭제 안 됨) 또는 사용자가 확인·추가. main은 하나만.
    usable = [dict(e) for e in needs if e["usable"]]
    main_seen = False
    for e in usable:
        if e["level"] == "main" and not main_seen:
            main_seen = True
        elif e["level"] == "main":
            e["level"] = "sub"
    if usable and not main_seen:
        usable[0]["level"] = "main"

    prov = {f"needs:{e['domain']}": e["prov"] for e in usable}
    for name, (val, p) in fields.items():
        prov[name] = p if _known(name, val) else ("unknown" if p == "extract" else p)
    for ch in CONSTRAINT_CHANNELS:
        prov[f"constraints:{ch}"] = cons[ch]["source"] if ch in cons else "unknown"

    constraints = [{k: cons[ch][k] for k in ("channel", "level", "evidence")}
                   for ch in CONSTRAINT_CHANNELS if ch in cons]
    s["confirmed_profile"] = {
        "needs": [{"domain": e["domain"], "level": e["level"], "evidence": e["evidence"]} for e in usable],
        "activity_range": fields["activity_range"][0],
        "constraints": constraints,
        "age": fields["age"][0],
        "region_sido": fields["region_sido"][0],
        "region_sgg": fields["region_sgg"][0],
    }
    s["provenance"] = prov
    s["conflicts"] = conflicts
    s["need_changes"] = changes
    # 회피(avoid) 접근 정보. 원하는 접근(want)은 쿼리 단계(query.build)가 덧붙인다.
    s["access"] = [{"polarity": "avoid", "channel": cons[ch]["channel"], "level": cons[ch]["level"],
                    "scope": None, "mode": None, "pattern_id": None, "source": cons[ch]["source"]}
                   for ch in CONSTRAINT_CHANNELS if ch in cons]
    return s


def usable_domains(session: dict) -> list[str]:
    return [n["domain"] for n in session.get("confirmed_profile", {}).get("needs", [])]


def confirm_view(session: dict) -> dict:
    """확인 화면에 보여줄 항목. 이미 추출된 값만 보여주고 새 정보는 묻지 않는다."""
    ext = session.get("extracted") or {}
    return {
        "needs": [{"domain": n["domain"], "level": n["level"], "status": n.get("status"),
                   "evidence": n.get("evidence")} for n in ext.get("needs", [])],
        "activity_range": ext.get("activity_range"),
        "constraints": [{"channel": c["channel"], "level": c["level"]} for c in ext.get("constraints", [])],
        "age": ext.get("age"),
        "region_sido": ext.get("region_sido"),
    }


def rules_view(confirmed_profile: dict, cfg: dict, access: list | None = None) -> dict:
    """규칙 모듈(rules.filter.apply_rules)에 넘길 사본.

    stage2.rules_pass.age / region_sido가 false면 해당 값을 None으로 바꿈.
    true면 나이는 rules.normalize.norm_age를 거친 int, 지역은 SIDO 약칭 그대로 넘김.
    access는 항상 복사해 넣음(check_access 초안용, 현재 CHECKS에서는 읽지 않음).
    """
    view = copy.deepcopy(confirmed_profile)
    rp = (cfg.get("stage2") or {}).get("rules_pass") or {}
    view["age"] = norm_age(view.get("age")) if rp.get("age", False) else None
    if not rp.get("region_sido", False):
        view["region_sido"] = None
    if not rp.get("region_sgg", False):
        view["region_sgg"] = None
    view["access"] = copy.deepcopy(access or [])
    return view
