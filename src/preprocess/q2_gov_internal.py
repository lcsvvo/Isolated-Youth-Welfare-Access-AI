"""보조금24 내부 불일치(JA 구조화 조건 대 본문) 재사용 표. 두 플랫폼 간 차이와 섞지 않는다.

기존 gov24_eligibility.csv 의 열(CB.GOV24_INTERNAL["columns"])을 그대로 가져온다(재계산·수정 없음).
A·B·C 쌍의 보조금24 서비스마다 쌍 정보와 함께 붙이고, 열 값별 건수만 기술 집계한다.
출력: data/processed/q2_gov24_internal.csv, logs/validation/q2_item_rules/gov24_internal_counts.csv
실행: python -m src.preprocess.q2_gov_internal
"""
from __future__ import annotations

import pandas as pd

from ..collect._common import PROCESSED, ROOT
from . import q2_codebook as CB
from . import q2_items_v3 as V

ELIG = PROCESSED / "gov24_eligibility.csv"
OUT = PROCESSED / "q2_gov24_internal.csv"
COUNTS = V.RULES_DIR / "gov24_internal_counts.csv"
LABEL = "보조금24 내부(JA 구조화 조건 대 본문) 불일치: 플랫폼 간 차이와 별개"


def build(pairs: pd.DataFrame | None = None, elig: pd.DataFrame | None = None) -> pd.DataFrame:
    pairs = V.pairs_table() if pairs is None else pairs
    elig = V.read(ELIG) if elig is None else elig
    cols = CB.GOV24_INTERNAL["columns"]
    e = elig[["서비스ID"] + cols].rename(columns={"서비스ID": "gov_id"})
    out = pairs[["servId", "gov_id", "tier", "gov_shared", "match_confirmed"]].merge(e, on="gov_id", how="left", validate="m:1")
    out[cols] = out[cols].fillna("")
    out["source_note"] = CB.GOV24_INTERNAL["caveat"]
    return out


def counts(t: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for c in CB.GOV24_INTERNAL["columns"]:
        g = t.groupby(c).agg(n_pairs=("servId", "size"), n_gov_services=("gov_id", "nunique")).reset_index().rename(columns={c: "value"})
        rows.append(g.assign(column=c))
    r = pd.concat(rows)[["column", "value", "n_pairs", "n_gov_services"]]
    r["scope"] = LABEL
    return r.reset_index(drop=True)


def main() -> None:
    t = build()
    t.to_csv(OUT, index=False, encoding="utf-8-sig")
    c = counts(t)
    c.to_csv(COUNTS, index=False, encoding="utf-8-sig")
    print(f"{LABEL}\n{len(t)}쌍 → {OUT.relative_to(ROOT)}")
    print(c.drop(columns="scope").to_string(index=False))


if __name__ == "__main__":
    main()
