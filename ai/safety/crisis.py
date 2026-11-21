"""0단계 위기 확인: 키워드 OR LLM. 둘 중 하나라도 감지하면 위기로 본다 (재현율 우선).

키워드 목록은 configs/crisis_terms.yaml. 입력과 키워드 모두 공백을 지우고 포함 여부를 본다.
LLM은 키워드에 없는 위기 표현을 잡는 백업이다. 고립·무기력 서술만으로는 위기로 보지 않게 기준을 좁혔다
(kanana_v1: "애매하면 true" 프롬프트에서 비위기 문장 158/250 오경보).
LLM 호출이 실패하면 안전한 쪽(위기)으로 둔다.
"""
import os
import re

import yaml

DEFAULT_TERMS = ["죽고싶", "사라지고싶", "자살", "자해", "살기싫", "끝내고싶"]
TERMS_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "configs", "crisis_terms.yaml")
SYSTEM = """너는 위기 신호 분류기다. 사용자 문장에 자살·자해 위험 신호가 있는지 판단한다.

위기(true): 아래 중 하나가 문장에 드러남
- 죽고 싶다, 자해하고 싶다는 말
- 사라지고 싶다, 없어지고 싶다, 내가 없는 게 낫다는 말
- 잠들어서 깨어나지 않았으면, 내일이 오지 않았으면 같은 말
- 살아갈 이유·의미가 없다, 살아 있는 것 자체가 버겁다는 말

위기 아님(false): 아래만 있고 위 표현이 없음
- 취업 실패, 시험 낙방, 돈 걱정, 집에만 있음, 외로움, 사람이 무서움
- 무기력, 우울감, 잠을 못 잠, 지침, 자신감이 떨어짐
- 상담이나 도움을 원한다는 말

힘든 상황이나 우울한 감정만으로는 위기가 아니다. 위 위기 표현이 실제로 있을 때만 true로 한다.

예시
입력: 몇 년째 취업이 안 돼서 요즘은 방에만 있어요. 사람 만나는 것도 무서워요.
출력: {"crisis": false, "evidence": ""}
입력: 너무 지치고 아무것도 하기 싫어요. 상담을 받아 보고 싶어요.
출력: {"crisis": false, "evidence": ""}
입력: 요즘은 그냥 내가 없어지면 다들 편할 것 같다는 생각이 들어요.
출력: {"crisis": true, "evidence": "내가 없어지면 다들 편할 것 같다"}

다음 JSON만 출력: {"crisis": true 또는 false, "evidence": "위기 표현이 있는 원문, 없으면 빈 문자열"}"""


def _squash(s):
    return re.sub(r"\s+", "", s or "")


def _load(path):
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_terms(path=TERMS_FILE):
    terms = [_squash(t) for t in _load(path).get("terms") or [] if _squash(t)]
    return terms or DEFAULT_TERMS


def load_excludes(path=TERMS_FILE):
    return [_squash(t) for t in _load(path).get("exclude") or [] if _squash(t)]


def keyword_hit(text, terms=None, excludes=None):
    """처음 걸린 키워드, 없으면 None. exclude 표현을 먼저 지우고 본다."""
    s = _squash(text)
    for x in (load_excludes() if excludes is None else excludes):
        s = s.replace(x, " ")
    return next((t for t in (terms or load_terms()) if t in s), None)


def llm_check(text, llm):
    """LLM 판정. 실패하면 위기로 둔다."""
    try:
        r = llm.chat_json(SYSTEM, f"입력: {text}", task="crisis")
    except Exception:
        return {"crisis": True, "by": "llm_error", "evidence": ""}
    if r.get("crisis") is True or str(r.get("crisis")).lower() == "true":
        return {"crisis": True, "by": "llm", "evidence": r.get("evidence", "")}
    return {"crisis": False}


def check_crisis(text, llm, terms=None):
    hit = keyword_hit(text, terms)
    if hit:
        return {"crisis": True, "by": "keyword", "evidence": hit}
    return llm_check(text, llm)
