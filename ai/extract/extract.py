"""1단계 욕구·조건 추출 (Kanana). 라벨링 규칙 파일을 그대로 프롬프트에 넣는다."""
import json

import numpy as np
from extract.postprocess import postprocess
from extract.schema import failed_record, normalize

SCHEMA_HINT = """출력은 아래 JSON 하나만 (형식 extract-v1, ai/extract/SCHEMA.md):
{"needs": [{"domain": "<영역 중 하나>", "level": "main 또는 sub", "evidence": "<원문을 그대로 자른 표현>"}],
 "activity_range": "방 안 | 집 안 | 집 근처 | 필요하면 외출 | 판단 불가",
 "activity_evidence": "<활동 범위 근거 원문, 없으면 null>",
 "constraints": [{"channel": "전화 또는 방문", "level": "부담 또는 불가", "evidence": "<원문>"}],
 "age": <글에 나온 나이 숫자, 없으면 null>,
 "region_sido": "<글에 나온 시도 약칭(서울, 경기 등), 없으면 null>"}
드러나지 않은 항목은 빈 목록, '판단 불가', null로 둔다."""


def build_system_prompt(rules_text, domains):
    return ("너는 청년 복지 상담 보조원이다. 사용자 문장에서 욕구와 조건을 추출한다.\n"
            f"영역은 반드시 다음 중에서만 고른다: {', '.join(domains)}\n"
            "문장에 직접 드러난 것만 판단하고, 드러나지 않으면 '판단 불가'로 둔다.\n\n"
            f"[라벨링 규칙]\n{rules_text}\n\n{SCHEMA_HINT}")


def pick_examples(text, embedder, examples, example_emb, n=3):
    """입력과 가장 비슷한 개발 세트 문장 n개를 예시로 고른다 (임베딩 재사용)."""
    if not examples:
        return []
    q = embedder.encode([text])[0]
    idx = np.argsort(-(example_emb @ q))
    return [examples[i] for i in idx if examples[i]["text"] != text][:n]  # 자기 자신 제외


def extract(text, llm, embedder, rules_text, domains, examples, example_emb, cfg):
    shots = pick_examples(text, embedder, examples, example_emb, cfg["n_examples"])
    shot_txt = "\n".join(f"예시 입력: {e['text']}\n예시 출력: {json.dumps(e['label'], ensure_ascii=False)}"
                         for e in shots)
    user = f"{shot_txt}\n\n입력: {text}"
    try:
        out = llm.chat_json(build_system_prompt(rules_text, domains), user,
                            task="extract", retries=cfg["max_retries"])
    except Exception:
        out = None  # 호출 오류 또는 JSON이 끝내 깨짐

    # JSON 객체를 못 받으면 실패 기록(extract_status=failed, 되묻기). '정보 없음'과 구분하기 위함
    # 코드가 LLM 결과를 검사해 extract-v1 형식으로 맞춘다 (목록 밖 값 제거, 근거 확인)
    rec = normalize(out, text, domains) if isinstance(out, dict) else failed_record()
    # 원문 표현 규칙으로 제약·활동 범위·상태 서술 영역을 보정 (kanana_v1 오류 분석 반영)
    return postprocess(rec, text) if cfg.get("postprocess", True) else rec
