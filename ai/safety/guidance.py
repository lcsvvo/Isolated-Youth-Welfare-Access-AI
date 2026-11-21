"""위기 중지 시 보여줄 고정 안내. 연락처는 configs/crisis_guidance.yaml의 정적 값만 쓴다 (LLM 생성 금지)."""
import copy
import os

import yaml

GUIDANCE_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             "configs", "crisis_guidance.yaml")
CONTACT_KEYS = ("name", "number", "hours", "description", "source_url", "verified_at")


def _need(d, keys, where):
    if not isinstance(d, dict):
        raise ValueError(f"crisis_guidance: {where} 형식 오류")
    for k in keys:
        if not str(d.get(k) or "").strip():
            raise ValueError(f"crisis_guidance: {where}.{k} 누락 또는 빈 값")


def _validate(cfg):
    _need(cfg, ("message", "disclaimer", "source_version"), "root")
    emergency = cfg.get("emergency")
    _need(emergency, ("message",), "emergency")
    for where, items in (("emergency.numbers", emergency.get("numbers")), ("contacts", cfg.get("contacts"))):
        if not isinstance(items, list) or not items:
            raise ValueError(f"crisis_guidance: {where} 비어 있음")
        for i, c in enumerate(items):
            _need(c, CONTACT_KEYS, f"{where}[{i}]")


def crisis_guidance(path=GUIDANCE_FILE) -> dict:
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    _validate(cfg)
    cfg = copy.deepcopy(cfg)
    cfg["message"] = " ".join(str(cfg["message"]).split())
    return {
        "type": "crisis_guidance",
        "message": cfg["message"],
        "emergency": cfg["emergency"],
        "contacts": cfg["contacts"],
        "disclaimer": cfg["disclaimer"],
        "source_version": cfg["source_version"],
    }
