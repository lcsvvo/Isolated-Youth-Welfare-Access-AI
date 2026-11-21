"""보조금24 전체 수집 (행정안전부_대한민국 공공서비스(혜택), 공공데이터포털 15113968).

엔드포인트: https://api.odcloud.kr/api/gov24/v3/{serviceList, serviceDetail, supportConditions}
인증: serviceKey (공공데이터포털 '인코딩' 키를 .env/.env.txt의 GOV24_API_KEY에 그대로)
결과: data/raw/gov24/gov24_<endpoint>.csv, data/raw/gov24/_collection.json

실행: python -m src.collect.gov24
"""
from __future__ import annotations

import pandas as pd

from ._common import RAW, get_json, require_key, write_log

BASE = "https://api.odcloud.kr/api/gov24/v3/"
ENDPOINTS = ("serviceList", "serviceDetail", "supportConditions")
PER_PAGE = 1000


def fetch(endpoint: str, key: str) -> tuple[pd.DataFrame, int]:
    rows: list[dict] = []
    page, total = 1, 0
    while True:
        d = get_json(BASE + endpoint, {"page": page, "perPage": PER_PAGE}, raw_key=("serviceKey", key))
        total = int(d.get("totalCount", 0))
        batch = d.get("data") or []
        rows += batch
        if not batch or len(rows) >= total:
            break
        page += 1
    if len(rows) != total:
        raise SystemExit(f"[중단] {endpoint}: 수집 {len(rows)}건 != totalCount {total}")
    df = pd.DataFrame(rows)
    if df["서비스ID"].duplicated().any():
        raise SystemExit(f"[중단] {endpoint}: 서비스ID 중복")
    return df, total


def main() -> None:
    key = require_key("GOV24_API_KEY")
    out = RAW / "gov24"
    out.mkdir(parents=True, exist_ok=True)
    files, meta = [], {}
    for ep in ENDPOINTS:
        df, total = fetch(ep, key)
        path = out / f"gov24_{ep}.csv"
        df.to_csv(path, index=False, encoding="utf-8-sig")
        files.append(path)
        meta[ep] = {"url": BASE + ep, "perPage": PER_PAGE, "totalCount": total, "rows": len(df)}
        print(f"{ep}: {len(df):,}건")
    write_log(out, "행정안전부 보조금24 (data.go.kr 15113968)", meta, files)


if __name__ == "__main__":
    main()
