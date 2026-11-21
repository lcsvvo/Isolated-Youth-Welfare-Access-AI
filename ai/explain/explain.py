"""5단계 설명 생성 (Kanana). 서비스 원문만 근거로 쓰고, 근거가 원문에 없으면 템플릿으로 대체.

템플릿으로 바뀔 때마다 이유를 로그 한 줄로 남긴다(로거 이름 "explain").
남기는 것: 서비스 ID, 시도 번호, 이유 종류, 걸린 시간(ms), 예외 종류 이름.
남기지 않는 것: 사용자 입력, 서비스 원문, 모델 출력, 예외 메시지(원문이 섞일 수 있음).
"""
import logging
import time

from core.utils import evidence_in

log = logging.getLogger("explain")

SOURCE_COLS = ["target_text", "criteria_text", "benefit_text", "apply_text"]
SYSTEM = ("너는 청년에게 복지 서비스를 부담 없이 설명한다. reason과 how_to_apply는 일관된 해요체로 작성한다. 명사형 종결(지원, 함)과 합니다체를 섞지 않는다.  [서비스 원문]에 있는 내용만 사용한다. "
          "'신청할 수 있습니다'처럼 자격을 단정하지 않는다. 확인이 필요한 조건은 check_needed에 적는다.\n"
          '다음 JSON만 출력: {"reason": "1~2문장", "how_to_apply": "...", '
          '"check_needed": ["..."], "evidence": "<서비스 원문을 그대로 자른 문장>"}')


def _fail(sid, attempt, total, reason, started):
    log.warning("[explain] servId=%s attempt=%d/%d reason=%s elapsed_ms=%d",
                sid, attempt, total, reason, round((time.perf_counter() - started) * 1000))


def explain(svc, extracted, checks, llm, retries=1):
    sid = str(svc.get("servId") or "unknown")
    source = "\n".join(str(svc.get(c, "")) for c in SOURCE_COLS if svc.get(c))
    user = (f"[사용자 상황] {extracted.get('needs')} / 활동 범위: {extracted.get('activity_range')}\n"
            f"[규칙 판정] {checks}\n[서비스 원문]\n{source}")
    total = retries + 1
    last = "unknown"
    for n in range(1, total + 1):
        started = time.perf_counter()
        try:
            card = llm.chat_json(SYSTEM, user, task="explain")
        except Exception as e:      # 모델 호출·생성·JSON 파싱 실패. 메시지는 남기지 않고 종류만
            last = f"exception:{type(e).__name__}"
            _fail(sid, n, total, last, started)
            continue
        if not isinstance(card, dict):
            last = "bad_format: not_an_object"
            _fail(sid, n, total, last, started)
            continue
        evidence = card.get("evidence", "")
        if not isinstance(evidence, str):
            last = "bad_format: evidence_not_text"
            _fail(sid, n, total, last, started)
            continue
        if evidence_in(evidence, source):
            card["fallback"] = False
            return card
        last = "validation: evidence_empty" if not evidence.strip() else "validation: evidence_not_in_source"
        _fail(sid, n, total, last, started)
    # 두 번 다 실패 → 원문을 그대로 보여주는 템플릿
    log.warning("[explain] servId=%s reason=fallback_used last=%s", sid, last)
    return {"reason": "아래 원문 내용을 확인해 주세요.", "how_to_apply": svc.get("apply_text", ""),
            "check_needed": [k for k, v in checks.items() if v == "확인 필요"],
            "evidence": svc.get("target_text", ""), "fallback": True}
