"""Read version-bound research addenda without modifying SSIS source records."""

import hashlib
import json
from copy import deepcopy
from pathlib import Path

from core.data import load_services

Q2_CARD_LABELS = {
    "supplement_possible": "보완 가능 · 원문 확인 필요",
    "source_conflict": "출처 충돌 · 확인 필요",
    "source_check_required": "출처 확인 필요",
    "freshness_check_required": "최신성 확인 필요",
    "ssis_information_present": "SSIS 원문에 기재",
    "insufficient_evidence": "정보 부족 · 확인 필요",
}
Q2_ITEMS = {"age", "income", "channel", "period", "documents", "contact"}


def configured_addendum_path(configured, explicit=None, *, custom_master=False, optional=False):
    if explicit is not None:
        return Path(explicit).resolve()
    if custom_master or configured is None:
        return None
    path = Path(configured)
    return None if optional and not path.exists() else path


def load_addendum(path, kind, master_path, service_ids):
    if path is None:
        return {}
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    master_hash = hashlib.sha256(Path(master_path).read_bytes()).hexdigest()
    if data["schema"] != kind or data["master_sha256"] != master_hash:
        raise ValueError(f"{kind}: schema or service master version mismatch")
    rows = data["services"]
    if not isinstance(rows, dict) or not set(rows) <= set(service_ids):
        raise ValueError(f"{kind}: invalid service keys")
    if kind == "q1-review-v1" and set(rows) != set(service_ids):
        raise ValueError(f"{kind}: incomplete service coverage")
    if kind == "q2-info-v1":
        for sid, row in rows.items():
            if (row.get("same_program_confirmed") is not True or set(row.get("items", {})) != Q2_ITEMS
                    or any(item.get("auto_fill") is not False or item.get("status") not in Q2_CARD_LABELS
                           for item in row["items"].values())):
                raise ValueError(f"{kind}: invalid confirmed pair or field status for {sid}")
    return rows


def attach_q1(services, overlay):
    """Ranking metadata only; raw and search fields stay as loaded."""
    if not overlay:
        return services
    result = services.copy()
    for col in ("q1_core_domains", "q1_unjudged_domains"):
        result[col] = [tuple(overlay[sid][col]) for sid in result.index]
    return result


def card_review_data(serv_id, q1, q2):
    """Separate sourced metadata from generated explanations and eligibility."""
    data = {}
    access = q1.get(serv_id, {}).get("access")
    if access:
        data["q1_access_review"] = deepcopy(access)
    if serv_id in q2:
        comparison = deepcopy(q2[serv_id])
        for item in comparison["items"].values():
            item["status_label"] = Q2_CARD_LABELS[item["status"]]
            if item.get("freshness") == "latest_not_verified":
                item["freshness_label"] = "현재 안내 확인 필요"
        data["q2_source_comparison"] = comparison
    return data


class RecommendationReview:
    """Load ranking metadata and card evidence for the same SSIS snapshot."""

    def __init__(self, master_path, *, q1_path=None, q2_path=None):
        master = load_services(master_path).set_index("servId", drop=False)
        self.q1 = load_addendum(q1_path, "q1-review-v1", master_path, master.index)
        self.q2 = load_addendum(q2_path, "q2-info-v1", master_path, master.index)
        self.services = attach_q1(master, self.q1)

    def card_fields(self, serv_id):
        return card_review_data(serv_id, self.q1, self.q2)
