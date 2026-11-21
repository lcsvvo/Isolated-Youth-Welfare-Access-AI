"""2단계 질문과 상태 진행.

- 순서 기본값: 욕구 → 지역 → 활동범위 → 나이. 세션 전체 최대 max_total개(기본 2).
- 확인 화면은 질문 수에 넣지 않는다. 이미 아는 항목(추출·profile·확인 화면 입력)은 묻지 않는다.
- 한 질문에 한 항목만 묻는다.
- 욕구 질문에서 사용자가 '건너뛰기'·'모르겠어요'를 고르면 질문을 끝내고 explore_only(9개 영역 안내만, 검색 없음).
- 자유 입력(욕구 직접 쓰기, 확인 화면 메모)은 먼저 위기 확인을 거친다. 선택형 답은 생략한다.
- 욕구 자유 입력은 후처리(postprocess)를 끈 추출로 영역 후보만 받아 선택지로 다시 확인한다.
  "취업이요" 같은 짧은 답에서 후처리 게이트가 영역을 지우기 때문이다.
- 같은 항목의 추가 화면은 최대 1회(MAX_EXTRA_PER_SLOT). 후보 재확인과 형식 오류 뒤 선택지 재표시가
  이 1회를 같이 쓴다. 추가 화면은 max_total에 세지 않지만 항목당 1회로 묶여 제한을 우회하지 못한다.
  추가 화면에서도 형식이 틀리면 explore_only(end_reason=need_retry_exhausted)로 끝낸다.
- 확인 화면 메모는 위기 확인에만 쓰고 검색에는 반영하지 않는다.
- 활동범위·제약은 선택형만 받는다. postprocess 규칙 함수를 재사용하지 않는다.

외부 의존은 함수 인자로 받는다.
- crisis_fn(text) -> check_crisis 반환값
- extract_fn(text, cfg_extract) -> extract-v1 기록
"""
from __future__ import annotations

import copy

from confirm.session import (CONFIRM_ACTIONS, UNKNOWN_ACTIVITY, add_action, apply_confirm_action,
                             valid_activity, valid_age, valid_region)
from extract.schema import ACTIVITY_RANGES, SIDO, is_failed

SLOTS = ("need", "region_sido", "activity_range", "age")
ANSWER_TYPES = ("선택", "자유", "건너뜀", "모름")
ASK_ACTIVITY = [a for a in ACTIVITY_RANGES if a != UNKNOWN_ACTIVITY]
NEED_EXTRACT_OVERRIDES = {"postprocess": False}
DEFAULT_QUESTIONS = {"order": list(SLOTS), "enabled": {k: True for k in SLOTS}, "max_total": 2}

PROMPTS = {
    "need": "어떤 도움이 가장 필요하세요?",
    "region_sido": "어느 지역에 살고 계세요?",
    "activity_range": "요즘 평소에 어디까지 다니세요?",
    "age": "나이를 알려주실 수 있을까요?",
}
SKIP_LABELS = {"need": "모르겠어요", "region_sido": "건너뛰기",
               "activity_range": "말하고 싶지 않아요", "age": "건너뛰기"}
EXTRACT_FAILED = "extract_failed"
EXTRACT_FAILED_REASK = "입력하신 내용을 제대로 이해하지 못했어요. 다시 한번 적어 줄래요?"
EXTRACT_FAILED_CHOOSE = "입력하신 내용을 제대로 이해하지 못했어요. 아래에서 필요한 도움을 골라 줄래요?"


def question_cfg(cfg: dict) -> dict:
    q = {**DEFAULT_QUESTIONS, **((cfg.get("stage2") or {}).get("questions") or {})}
    q["enabled"] = {**DEFAULT_QUESTIONS["enabled"], **(q.get("enabled") or {})}
    return q


# ── 질문 선택 ─────────────────────────────────────────────────────
def slot_needed(session: dict, slot: str) -> bool:
    p, prov = session["confirmed_profile"], session["provenance"]
    if slot == "need":
        return not p["needs"]
    if slot == "region_sido":
        return p["region_sido"] is None and prov.get("region_sido") in (None, "unknown")
    if slot == "activity_range":
        return p["activity_range"] == UNKNOWN_ACTIVITY and prov.get("activity_range") in (None, "unknown")
    if slot == "age":
        return p["age"] is None and prov.get("age") in (None, "unknown")
    raise ValueError(slot)


def asked_count(session: dict) -> int:
    """다시 확인용 후속 질문(followup)은 같은 항목이라 세지 않는다."""
    return sum(1 for q in session["questions"] if not q.get("followup"))


def next_slot(session: dict, cfg: dict) -> str | None:
    qc = question_cfg(cfg)
    if asked_count(session) >= qc["max_total"]:
        return None
    asked = {q["slot"] for q in session["questions"]}
    if slot_needed(session, "need") and (not qc["enabled"].get("need", False) or "need" in asked):
        return None  # 영역을 받을 수 없으면 다른 항목을 물어도 검색할 수 없음 → explore_only
    for slot in qc["order"]:
        if qc["enabled"].get(slot, False) and slot not in asked and slot_needed(session, slot):
            return slot
    return None


def pending(session: dict) -> dict | None:
    q = session["questions"][-1] if session["questions"] else None
    return q if q and q.get("answer_type") is None else None


def advance(session: dict, cfg: dict) -> dict:
    """다음 질문을 띄우거나 ready / explore_only로 끝낸다."""
    s = copy.deepcopy(session)
    slot = next_slot(s, cfg)
    if slot:
        q = {"slot": slot, "order": asked_count(s) + 1, "answer": None, "answer_type": None}
        if slot == "need" and is_failed(s.get("extracted")):
            q["reason"] = EXTRACT_FAILED
        s["questions"].append(q)
        s["status"] = "awaiting_answer"
    else:
        s["status"] = "ready" if s["confirmed_profile"]["needs"] else "explore_only"
    return s


def render(question: dict, labels: dict[str, str], domains: list[str]) -> dict:
    """화면 표시용. labels는 query_terms.yaml domains.<영역>.label_plain."""
    slot = question["slot"]
    if slot == "need":
        opts = question.get("options") or domains
        options = [{"value": d, "label": labels.get(d, d)} for d in opts]
        # 추가 화면(후보 재확인·재표시)에서는 직접 쓰기를 받지 않는다
        allow_free = not question.get("followup")
        prompt = PROMPTS[slot]
        if question.get("reason") == EXTRACT_FAILED:
            prompt = EXTRACT_FAILED_REASK if allow_free else EXTRACT_FAILED_CHOOSE
        return {"slot": slot, "prompt": prompt, "options": options,
                "allow_free": allow_free, "skip_label": SKIP_LABELS[slot]}
    if slot == "region_sido":
        options = [{"value": r, "label": r} for r in SIDO]
    elif slot == "activity_range":
        options = [{"value": a, "label": a} for a in ASK_ACTIVITY]
    else:
        options = []  # 숫자 입력
    return {"slot": slot, "prompt": PROMPTS[slot], "options": options, "allow_free": False,
            "number_input": slot == "age", "skip_label": SKIP_LABELS[slot]}


# ── 답 처리 ──────────────────────────────────────────────────────
def _crisis_stop(s: dict, target: str, result: dict, at) -> dict:
    s = add_action(s, "crisis_stop", target, {"by": result.get("by"), "evidence": result.get("evidence", "")}, at)
    s["status"] = "crisis_stop"
    return s


MAX_EXTRA_PER_SLOT = 1  # 같은 항목의 추가 화면(후보 재확인 또는 선택지 재표시) 최대 횟수


def extra_used(s: dict, slot: str) -> bool:
    """이 항목에서 추가 화면을 이미 썼는지. 후보 재확인과 형식 오류 재표시가 1회를 같이 쓴다."""
    return sum(1 for q in s["questions"] if q["slot"] == slot and q.get("followup")) >= MAX_EXTRA_PER_SLOT


def _append_followup(s: dict, cur: dict, options: list[str], reason: str) -> dict:
    """같은 항목의 추가 화면. 질문 수(max_total)에는 세지 않지만 항목당 1회로 묶여 있어 제한을 우회하지 못함."""
    s["questions"].append({"slot": cur["slot"], "order": cur["order"], "followup": True, "reason": reason,
                           "options": options, "answer": None, "answer_type": None})
    s["status"] = "awaiting_answer"
    return s


def need_candidates(text: str, extract_fn, cfg_extract: dict) -> tuple[list[str], bool]:
    """후처리 없이 영역 후보와 추출 실패 여부를 받는다. 명시·확인 필요 모두 후보로 둔다."""
    rec = extract_fn(text, {**cfg_extract, **NEED_EXTRACT_OVERRIDES}) or {}
    return [n["domain"] for n in rec.get("needs", [])], is_failed(rec)


def answer(session: dict, value, answer_type: str, *, cfg: dict, domains: list[str],
           crisis_fn=None, extract_fn=None, at: str | None = None) -> dict:
    q = pending(session)
    if session["status"] != "awaiting_answer" or q is None:
        raise ValueError("대기 중인 질문이 없음")
    if answer_type not in ANSWER_TYPES:
        raise ValueError(answer_type)
    slot = q["slot"]
    s = copy.deepcopy(session)
    cur = s["questions"][-1]

    free_on_followup = False
    if answer_type == "자유":
        if slot != "need":
            raise ValueError("자유 입력은 욕구 질문에서만 받음")
        result = crisis_fn(value)  # 자유 입력은 화면과 무관하게 먼저 위기 확인
        if result.get("crisis"):
            cur.update(answer=value, answer_type="자유", crisis_by=result.get("by"))
            return _crisis_stop(s, slot, result, at)
        if cur.get("followup") or extra_used(s, slot):
            # 재확인 화면은 선택형만 받음. 추가 화면을 이미 썼으므로 아래 형식 오류 처리로 보냄
            free_on_followup = True
        else:
            candidates, failed = need_candidates(value, extract_fn, cfg.get("extract") or {})
            cands = [d for d in candidates if d in domains]
            cur.update(answer=value, answer_type="자유", candidates=cands)
            s = add_action(s, "answer", slot, value, at, answer_type="자유")
            # 후보를 선택지로 다시 확인 (후보가 없으면 9개 영역 전체). 항목당 추가 화면 1회
            reason = EXTRACT_FAILED if failed else "confirm_candidates"
            return _append_followup(s, cur, cands or list(domains), reason)

    if answer_type in ("건너뜀", "모름"):
        cur.update(answer=None, answer_type=answer_type)
        s = add_action(s, "answer", slot, None, at, answer_type=answer_type)
        if slot == "need":
            s["status"] = "explore_only"
            return s
        return advance(s, cfg)

    # 선택형: 위기 확인 생략, 형식 검증
    if slot == "need" and free_on_followup:
        ok = None
    elif slot == "need":
        vals = value if isinstance(value, list) else [value]
        allowed = cur.get("options") or domains
        vals = [v for v in vals if v in allowed]
        ok = vals or None
    elif slot == "region_sido":
        ok = valid_region(value)
    elif slot == "activity_range":
        ok = value if value in ASK_ACTIVITY and valid_activity(value) else None
    else:
        ok = valid_age(value)
    if ok is None:
        # 형식 오류: 값은 저장하지 않고 answer_type '모름', rejected 표시 (사용자가 고른 '모름'과 구분)
        cur.update(answer=None, answer_type="모름", rejected=True)
        s = add_action(s, "answer", slot, None, at, answer_type="모름", rejected=True)
        if slot == "need":
            if not extra_used(s, slot):
                return _append_followup(s, cur, list(domains), "retry_invalid")  # 선택지를 한 번 다시 보여줌
            s["status"] = "explore_only"
            s["end_reason"] = "need_retry_exhausted"
            return s
        return advance(s, cfg)
    cur.update(answer=ok, answer_type="선택")
    s = add_action(s, "answer", slot, ok, at, answer_type="선택")
    return advance(s, cfg)


def memo(session: dict, text: str, *, crisis_fn, at: str | None = None) -> dict:
    """확인 화면 자유 메모. 위기 확인에만 쓴다.

    - 검색에는 반영하지 않는다: confirmed_profile·access에 넣지 않고(derive가 memo를 무시),
      query.build도 memo 행동을 읽지 않는다. 쿼리 재료는 original_text, 확정 영역 evidence, 질문 답뿐이다.
    - 위기가 아니면 actions에 기록만 남는다.
    """
    result = crisis_fn(text)
    if result.get("crisis"):
        return _crisis_stop(session, "memo", result, at)
    return add_action(session, "memo", None, text, at)


def step(session: dict, action: dict, *, cfg: dict, domains: list[str], crisis_fn=None, extract_fn=None) -> dict:
    """pipeline.resume 진입점. action = {"type", "target", "value", "answer_type"?, "at"?}."""
    if session["status"] in ("crisis_stop", "explore_only", "ready"):
        raise ValueError(f"끝난 세션: {session['status']}")
    t = action.get("type")
    if t == "memo":
        return memo(session, action.get("value"), crisis_fn=crisis_fn, at=action.get("at"))
    if t == "answer":
        return answer(session, action.get("value"), action.get("answer_type", "선택"), cfg=cfg, domains=domains,
                      crisis_fn=crisis_fn, extract_fn=extract_fn, at=action.get("at"))
    if t in CONFIRM_ACTIONS:
        s = apply_confirm_action(session, action, domains)
        return advance(s, cfg) if t == "finish_confirm" else s
    raise ValueError(f"알 수 없는 action type: {t}")


def auto_finish(session: dict) -> dict:
    """사용자 입력 없이 진행(pipeline.run). confirmation은 'none' 그대로, '확인 필요'는 쓰지 않는다."""
    s = copy.deepcopy(session)
    if s["status"] == "crisis_stop":
        return s
    s["status"] = "ready" if s["confirmed_profile"]["needs"] else "explore_only"
    return s
