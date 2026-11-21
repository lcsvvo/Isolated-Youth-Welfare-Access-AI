"""활동범위(사용자 외출 단계)와 서비스 최소 외출필요도의 순위 조정. 제외 기준이 아님.

사용자 외출 단계는 서비스 외출필요도와 같은 척도가 아님(src/recommend/eligibility.py 원칙).
따라서 어떤 경우에도 제외하지 않고 감점만 함. 값은 단계 차용일 뿐 대응이 증명되지 않았으므로 잠정임.

활동범위 -> 사용자 최대 외출 단계(잠정 대응표)
| activity_range | 단계 |
| 방 안 | 0 |
| 집 안 | 0 |
| 집 근처 | 1 |
| 필요하면 외출 | 2 |
| 판단 불가, 없음 | None (판단 보류) |
서비스 outing_need_min은 0~3 또는 N. N이나 누락이면 판단 보류.
감점 = weight x max(0, outing_need_min - 사용자 단계)
"""
import pandas as pd

from rules import overlay

USER_LEVEL = {"방 안": 0, "집 안": 0, "집 근처": 1, "필요하면 외출": 2, "판단 불가": None}

# 잠정 가중치(근거 데이터 없음). config로 덮어쓸 수 있도록 모듈 변수로 둠.
DEFAULT_WEIGHTS = {"per_level": 0.5}


def outing_penalty(svc, profile):
    """(penalty, reason). 항상 0 이상이며 제외하지 않음."""
    ov = overlay.get_access(svc.get("servId"))
    if ov and (ov.get("activity_visit") or ov["use_visit"]) in ("불명", "서비스별 상이", "담당자 방문", "불필요"):
        return 0.0, "판단 보류: 검수된 이용 조건 불명·혼재 또는 사용자 방문 불필요"
    user = USER_LEVEL.get(profile.get("activity_range"))
    v = svc.get("outing_need_min")
    try:
        need = None if v is None or (isinstance(v, float) and pd.isna(v)) else int(float(v))
    except (TypeError, ValueError):
        need = None
    if user is None:
        return 0.0, "판단 보류: 활동 범위 미확인"
    if need is None:
        return 0.0, "판단 보류: 서비스 외출 조건 원문 없음"
    gap = max(0, need - user)
    if gap == 0:
        return 0.0, f"서비스 최소 외출 {need}단계, 사용자 {user}단계 이내"
    return DEFAULT_WEIGHTS["per_level"] * gap, f"서비스 최소 외출 {need}단계, 사용자 {user}단계. 순위만 조정"
