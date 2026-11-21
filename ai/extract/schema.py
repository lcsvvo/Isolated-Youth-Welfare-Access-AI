"""1단계(욕구·조건 추출) 출력 JSON 형식 extract-v1.

파이프라인 뒤 단계(검색 쿼리, 규칙 검증, 설명)와 평가 코드가 모두 이 형식을 읽는다.
LLM 출력은 `normalize()`를 거쳐야 이 형식이 된다. 형식 설명은 ai/extract/SCHEMA.md.

{
  "schema_version": "extract-v1",
  "needs": [{"domain": "<앱 9개 영역>", "level": "main|sub", "evidence": "<원문>", "status": "명시|확인 필요"}],
  "activity_range": "방 안|집 안|집 근처|필요하면 외출|판단 불가",
  "activity_evidence": "<원문>" 또는 null,
  "constraints": [{"channel": "전화|방문", "level": "부담|불가", "evidence": "<원문>"}],
  "age": 정수 또는 null,
  "region_sido": "<시도 약칭>" 또는 null,
  "next_action": "추천|되묻기",
  "extract_status": "ok|failed"
}
extract_status: LLM 추출이 끝내 실패(호출 오류·JSON 깨짐·JSON 객체 아님)하면 "failed".
  이때 needs는 비어 있고 next_action은 "되묻기"라서, '글에 정보가 없음'(ok + 빈 값)과 구분하려면 이 값을 본다.
  이 키가 없는 예전 기록은 "ok"로 읽는다.
위기 판정은 0단계(safety/crisis.py) 결과로 따로 두고 이 JSON에는 넣지 않는다.
"""
from __future__ import annotations

import re

from core.utils import evidence_in

SCHEMA_VERSION = "extract-v1"
ACTIVITY_RANGES = ["방 안", "집 안", "집 근처", "필요하면 외출", "판단 불가"]
NEED_LEVELS = ["main", "sub"]
CONSTRAINT_CHANNELS = ["전화", "방문"]
CONSTRAINT_LEVELS = ["부담", "불가"]
SIDO = ["서울", "부산", "대구", "인천", "광주", "대전", "울산", "세종", "경기",
        "강원", "충북", "충남", "전북", "전남", "경북", "경남", "제주"]
_SIDO_ALIAS = {
    "서울특별시": "서울", "부산광역시": "부산", "대구광역시": "대구", "인천광역시": "인천",
    "광주광역시": "광주", "대전광역시": "대전", "울산광역시": "울산", "세종특별자치시": "세종",
    "경기도": "경기", "강원도": "강원", "강원특별자치도": "강원", "충청북도": "충북",
    "충청남도": "충남", "전라북도": "전북", "전북특별자치도": "전북", "전라남도": "전남",
    "경상북도": "경북", "경상남도": "경남", "제주도": "제주", "제주특별자치도": "제주",
}
_LEVEL_ALIAS = {"불가능": "불가", "부담스러움": "부담"}
_CHANNEL_ALIAS = {"대면": "방문", "외출": "방문", "통화": "전화"}
AGE_RANGE = (10, 99)
EXTRACT_STATUSES = ["ok", "failed"]

# 평가용: 카드 정답(`정답_제약`)은 값 하나라서 가장 강한 제약 하나로 줄인다
GOLD_CONSTRAINT = {
    ("방문", "불가"): "방문 불가(필수 제한)",
    ("전화", "불가"): "전화 불가",
    ("방문", "부담"): "방문 부담(선호)",
    ("전화", "부담"): "전화 부담(선호)",
}
GOLD_CONSTRAINT_ORDER = list(GOLD_CONSTRAINT)


def empty_record() -> dict:
    return {"schema_version": SCHEMA_VERSION, "needs": [], "activity_range": "판단 불가",
            "activity_evidence": None, "constraints": [], "age": None, "region_sido": None,
            "next_action": "되묻기", "extract_status": "ok"}


def _str(v) -> str:
    return v.strip() if isinstance(v, str) else ""


def _age(v):
    if isinstance(v, bool):
        return None
    if isinstance(v, str):
        m = re.fullmatch(r"\s*(\d{1,2})\s*(세|살)?\s*", v)
        v = int(m.group(1)) if m else None
    if isinstance(v, float) and v.is_integer():
        v = int(v)
    return v if isinstance(v, int) and AGE_RANGE[0] <= v <= AGE_RANGE[1] else None


def _sido(v):
    v = _str(v)
    v = _SIDO_ALIAS.get(v, v)
    return v if v in SIDO else None


def normalize(raw: dict | None, text: str, domains: list[str]) -> dict:
    """LLM 출력을 extract-v1 형식으로 맞춘다.

    - 목록 밖 영역·값은 버리고, 같은 영역이 여러 번 나오면 main을 남긴다.
    - 근거가 원문에 없으면 영역은 status '확인 필요', 활동 범위는 '판단 불가', 제약은 버린다.
    - 같은 채널 제약이 겹치면 '불가'를 남긴다.
    - 남은 영역이 없으면 next_action은 '되묻기'.
    """
    raw = raw if isinstance(raw, dict) else {}
    rec = empty_record()

    needs: dict[str, dict] = {}
    for n in raw.get("needs") or []:
        if not isinstance(n, dict) or n.get("domain") not in domains:
            continue
        ev = _str(n.get("evidence"))
        item = {"domain": n["domain"], "level": n.get("level") if n.get("level") in NEED_LEVELS else "main",
                "evidence": ev, "status": "명시" if evidence_in(ev, text) else "확인 필요"}
        prev = needs.get(item["domain"])
        if prev is None or (prev["level"] == "sub" and item["level"] == "main"):
            needs[item["domain"]] = item
    rec["needs"] = list(needs.values())

    ar = _str(raw.get("activity_range") or raw.get("outing_range"))
    ar_ev = _str(raw.get("activity_evidence"))
    if ar in ACTIVITY_RANGES and ar != "판단 불가" and evidence_in(ar_ev, text):
        rec["activity_range"], rec["activity_evidence"] = ar, ar_ev

    cons: dict[str, dict] = {}
    for c in raw.get("constraints") or []:
        if not isinstance(c, dict):
            continue
        ch = _CHANNEL_ALIAS.get(_str(c.get("channel") or c.get("type")), _str(c.get("channel") or c.get("type")))
        lv = _LEVEL_ALIAS.get(_str(c.get("level")), _str(c.get("level")))
        ev = _str(c.get("evidence"))
        if ch not in CONSTRAINT_CHANNELS or lv not in CONSTRAINT_LEVELS or not evidence_in(ev, text):
            continue
        if ch not in cons or (cons[ch]["level"] == "부담" and lv == "불가"):
            cons[ch] = {"channel": ch, "level": lv, "evidence": ev}
    rec["constraints"] = [cons[ch] for ch in CONSTRAINT_CHANNELS if ch in cons]

    rec["age"] = _age(raw.get("age"))
    rec["region_sido"] = _sido(raw.get("region_sido"))
    rec["next_action"] = "추천" if rec["needs"] else "되묻기"
    return rec


def failed_record() -> dict:
    """LLM 추출 실패 기록. 값은 빈 기록과 같고 extract_status만 failed."""
    rec = empty_record()
    rec["extract_status"] = "failed"
    return rec


def is_failed(rec: dict | None) -> bool:
    """추출 실패 여부. 키가 없는 예전 기록은 성공으로 본다."""
    return bool(rec) and rec.get("extract_status", "ok") == "failed"


def to_gold_labels(rec: dict) -> dict:
    """페르소나 카드 정답 열(`정답_영역` 등)과 같은 모양으로 바꾼다. 평가 코드에서 쓴다."""
    cons = {(c["channel"], c["level"]) for c in rec.get("constraints", [])}
    strongest = next((GOLD_CONSTRAINT[k] for k in GOLD_CONSTRAINT_ORDER if k in cons), "없음")
    return {
        "영역": sorted({n["domain"] for n in rec.get("needs", [])}),
        "활동범위": rec.get("activity_range", "판단 불가"),
        "제약": strongest,
        "나이언급": "Y" if rec.get("age") is not None else "N",
        "지역언급": "Y" if rec.get("region_sido") else "N",
    }
