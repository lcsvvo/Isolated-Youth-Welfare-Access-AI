"""STEP 02 앱 영역 대응표 및 SSIS 마스터 기준 서비스 건수.

마스터 재생성 후 실행: python -m src.labeling.app_domain_report
"""
from collections import Counter

import pandas as pd

from ..collect._common import PROCESSED, ROOT
from ..preprocess import keywords as K

OUT = ROOT / "outputs" / "02_labeling" / "tables"
V2_NAMES = K.APP_DOMAIN_V3_TO_V2      # 영역 이름은 keywords.py에서만 정의한다


def compare() -> pd.DataFrame:
    rows = []
    for area in K.Q21_LABELS:
        old = [(domain, status, condition) for q, domain, status, condition, _ in K.APP_DOMAIN_MAP_V2 if q == area]
        new = [(domain, level, condition) for q, domain, level, condition, _ in K.APP_DOMAIN_MAP_V3 if q == area]
        core = next(domain for domain, level, _ in new if level == "핵심")
        partial = [domain for domain, level, _ in new if level == "부분"]
        old_domains = {domain for domain, _, _ in old}
        old_conditions = {(domain, condition) for domain, _, condition in old if condition}
        if old_domains == {None}:
            diff = "대응 없음 해소"
        elif core not in V2_NAMES:
            diff = "신설 영역"
        elif any((V2_NAMES.get(domain), condition) in old_conditions for domain, _, condition in new):
            diff = "조건 유지"
        elif any(V2_NAMES.get(domain) not in old_domains for domain in partial):
            diff = "부분 대응 추가"
        elif core not in old_domains:
            diff = "영역 이름 변경"
        else:
            diff = "동일"
        rows.append({
            "q21_area": area,
            "v2_domain": "|".join(domain or "(대응 없음)" for domain, _, _ in old),
            "v2_status": "|".join(status for _, status, _ in old),
            "v3_core": core,
            "v3_partial": "|".join(partial),
            "diff_type": diff,
        })
    return pd.DataFrame(rows)


def service_counts(master: pd.DataFrame) -> pd.DataFrame:
    def count(col: str) -> Counter:
        return Counter(domain for labels in master[col].dropna().str.split("|") for domain in set(labels))

    v2 = count("app_domain_v2")
    core = count("app_domain")
    partial = count("app_domain_partial")
    return pd.DataFrame([
        {"app_domain": domain, "n_services_v2": v2[V2_NAMES.get(domain)],
         "n_services_v3_core": core[domain], "n_services_v3_partial": partial[domain]}
        for domain in K.APP_DOMAINS_V3
    ])


def main() -> None:
    master = pd.read_csv(PROCESSED / "services_master.csv", dtype=str, encoding="utf-8-sig", keep_default_na=False)
    tables = {
        "app_domain_map_v3": pd.DataFrame(
            K.APP_DOMAIN_MAP_V3, columns=["q21_area", "app_domain", "level", "condition", "note"]
        ),
        "app_domain_map_compare": compare(),
        "app_domain_v2_v3_counts": service_counts(master),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(OUT / f"{name}.csv", index=False, encoding="utf-8-sig")
        print(f"{name}.csv {len(table)}행")


if __name__ == "__main__":
    main()
