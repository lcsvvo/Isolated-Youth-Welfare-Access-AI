import json
import re


def normalize(s: str) -> str:
    """공백·문장부호를 지워 근거 문장 비교를 너그럽게 한다."""
    return re.sub(r"[\s\.,!?~·'\"“”‘’()\[\]-]", "", s or "")


def evidence_in(evidence: str, source: str) -> bool:
    """근거 문장이 원문에 들어 있는지 검사 (LLM이 지어냈는지 확인)."""
    e = normalize(evidence)
    return bool(e) and e in normalize(source)


def parse_json(text: str) -> dict:
    """LLM 출력에서 첫 { 부터 마지막 } 까지 잘라 JSON으로 읽는다."""
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("JSON을 찾지 못함")
    return json.loads(text[start:end + 1])
