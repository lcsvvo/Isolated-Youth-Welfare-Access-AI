"""검수 반영 전후 비교 (재검증 단계). 사람 검수 결과(overlay)를 켜고 끌 때 규칙 판정이 어떻게 달라지는지 봄.

실행 (저장소 루트에서)
  python -m ai.eval.rules_overlay_compare [--review-dir DIR]

출력: outputs/07_rules/tables/
- overlay_compare.csv  고정 프로필 x 판정(연령, 지역, 접근) x 상태별 서비스 수 (반영 전 off, 반영 후 on)
- overlay_changes.csv  판정이 바뀐 행만 (servId, profile, check, before, after, reason)

주의: services_master는 읽기만 함. 검수 파일이 없으면 변경 0건이 정상임.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

AI = Path(__file__).resolve().parents[1]
ROOT = AI.parent
sys.path.insert(0, str(AI))

from rules import overlay  # noqa: E402
from rules.access import check_access  # noqa: E402
from rules.filter import check_age  # noqa: E402
from rules.region import check_region  # noqa: E402

MASTER = ROOT / "data" / "processed" / "services_master.csv"
OUT = ROOT / "outputs" / "07_rules" / "tables"
CHECKS = (("연령", check_age), ("지역", check_region), ("접근", check_access))
STATUSES = ("통과", "확인 필요", "제외")

PROFILES = {
    "25세 서울 노원구 방문 불가": {"age": 25, "region_sido": "서울특별시", "region_sgg": "노원구",
                              "access": [{"polarity": "avoid", "channel": "방문", "level": "불가"}]},
    "25세 서울 시군구 없음 방문 부담": {"age": 25, "region_sido": "서울특별시",
                                 "access": [{"polarity": "avoid", "channel": "방문", "level": "부담"}]},
    "30세 부산 접근 없음": {"age": 30, "region_sido": "부산광역시"},
}


def judge_all(services: list[dict], profile: dict) -> dict:
    """{(servId, check): (status, reason)}"""
    return {(s["servId"], name): fn(s, profile) for s in services for name, fn in CHECKS}


def compare(services: list[dict], profiles: dict = PROFILES):
    """(집계표, 변경표) DataFrame. 검수 반영 off 후 on 순서로 계산함."""
    count_rows, change_rows = [], []
    for pname, prof in profiles.items():
        overlay.set_enabled(False)
        try:
            before = judge_all(services, prof)
        finally:
            overlay.set_enabled(True)
        after = judge_all(services, prof)
        for name, _ in CHECKS:
            for st in STATUSES:
                count_rows.append({
                    "profile": pname, "check": name, "status": st,
                    "off": sum(v[0] == st for k, v in before.items() if k[1] == name),
                    "on": sum(v[0] == st for k, v in after.items() if k[1] == name)})
        for key, (st, _) in before.items():
            a_st, a_why = after[key]
            if a_st != st:
                change_rows.append({"servId": key[0], "profile": pname, "check": key[1],
                                    "before": st, "after": a_st, "reason": a_why})
    counts = pd.DataFrame(count_rows, columns=["profile", "check", "status", "off", "on"])
    changes = pd.DataFrame(change_rows, columns=["servId", "profile", "check", "before", "after", "reason"])
    return counts, changes


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--review-dir", default=None)
    ap.add_argument("--master", default=str(MASTER))
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args(argv)
    saved = os.environ.get(overlay.ENV_DIR)
    if args.review_dir:
        os.environ[overlay.ENV_DIR] = args.review_dir
    overlay.reset_cache()
    try:
        return _run(args)
    finally:   # 호출 뒤 환경변수와 캐시를 되돌림
        if saved is None:
            os.environ.pop(overlay.ENV_DIR, None)
        else:
            os.environ[overlay.ENV_DIR] = saved
        overlay.reset_cache()


def _run(args):
    df = pd.read_csv(args.master, dtype=str, keep_default_na=False)
    services = df.to_dict("records")
    counts, changes = compare(services)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    counts.to_csv(out / "overlay_compare.csv", index=False, encoding="utf-8-sig")
    changes.to_csv(out / "overlay_changes.csv", index=False, encoding="utf-8-sig")
    print(f"검수 폴더: {overlay.review_dir()}")
    print(f"서비스 {len(services)}건, 프로필 {len(PROFILES)}개")
    if len(changes):
        print(changes.groupby(["profile", "check", "before", "after"]).size().rename("건수").to_string())
    print(f"판정 변경 {len(changes)}건")
    for w in overlay.warnings:
        print("경고:", w)
    return counts, changes


if __name__ == "__main__":
    main()
