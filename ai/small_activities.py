"""Optional everyday activities from the team's HTML draft; no model inference."""
import json
import re
from pathlib import Path

RANGES = {"방 안": 0, "집 안": 1, "집 근처": 2, "필요하면 외출": 3, "외출": 3}


def should_offer(text, slot=None):
    text = re.sub(r"\s+", "", text or "")
    urgent = re.search(r"급하|급해|당장|오늘까지|내일까지|밀렸|쫓겨|퇴거|잘곳이?없", text)
    essential = re.search(r"월세|집세|보증금|주거|퇴거|잘곳|생활비|긴급생계|공과금", text)
    if urgent and essential:
        return False
    if re.search(r"의욕이?없|의욕이?안나|기운이?없|무기력|아무것도(?:하기)?싫|할(?:수가|수도)?없|하고싶은(?:게|것이?)없|뭘해야|무엇을해야|뭐부터해야|뭐부터할지|어디서부터시작", text):
        return True
    return slot in (None, "need") and bool(re.fullmatch(r"(?:잘|아직|정말|그냥)?(?:모르겠어요|모르겠어|몰라요|잘몰라요)[.!?]*", text))


def load_catalog():
    return json.loads((Path(__file__).parent / "data/small_activities.json").read_text(encoding="utf-8"))


def select_activities(scope="방 안", energy=1, contact=0, visit_blocked=False):
    if scope not in RANGES or energy not in (1, 2, 3) or contact not in (0, 1):
        raise ValueError("활동 조건을 확인해 주세요")
    catalog = load_catalog()
    ceiling = min(RANGES[scope], 1 if visit_blocked else 3)
    rows = [a for a in catalog["activities"] if RANGES[a["range"]] <= ceiling
            and a["effort"] <= energy and a["contact_max"] <= contact]
    return sorted(rows, key=lambda a: (-RANGES[a["range"]], a["effort"], a["contact_max"], a["id"]))[:3]


PAIR_FEW = 3      # 표시 서비스가 이보다 적으면 작은 활동을 함께 보여 줌
PAIR_DOMAIN = "일상·여가"   # 주 추천이 작은 활동인 영역


def pair_activities(needs, activity_range=None, n_services=0, visit_blocked=False, declined=False, limit=3):
    """추천 결과 옆에 붙일 작은 활동. 서비스를 대신하지 않고 보완만 함.

    - 서비스가 PAIR_FEW개 미만이거나 욕구에 일상·여가가 있을 때만 보여 줌
    - 앞에서 작은 활동을 사양했으면 서비스가 0개일 때만 보여 줌
    - 사용자가 직접 고르지 않았으므로 부담은 보수적으로: effort 2 이하, 상호작용 없음
    - 활동 범위를 모르면 방 안 기준, 방문 불가면 집 안까지
    - 욕구 영역(main 먼저)과 겹치는 활동을 앞에 두고, 부족하면 일상·여가로 채움
    """
    domains = [n["domain"] for n in sorted(needs or [], key=lambda n: n.get("level") != "main")]
    if declined and n_services > 0:
        return []
    if n_services >= PAIR_FEW and PAIR_DOMAIN not in domains:
        return []
    ceiling = min(RANGES.get(activity_range, 0), 1 if visit_blocked else 3)
    rows = [dict(a) for a in load_catalog()["activities"]
            if RANGES[a["range"]] <= ceiling and a["effort"] <= 2 and a["contact_max"] == 0]

    def rank(a):
        hit = [i for i, d in enumerate(domains) if d in a["domains"]]
        a["matched"] = domains[hit[0]] if hit else None
        tier = hit[0] if hit else (len(domains) if PAIR_DOMAIN in a["domains"] else len(domains) + 1)
        return (tier, -RANGES[a["range"]], a["effort"], a["id"])

    rows.sort(key=rank)
    return rows[:limit]
