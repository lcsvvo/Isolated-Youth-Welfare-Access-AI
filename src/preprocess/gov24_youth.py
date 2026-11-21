"""보조금24 3개 엔드포인트를 서비스ID로 병합하고 청년 대상 플래그를 붙인다.

입력: data/raw/gov24/gov24_*.csv
출력: data/processed/gov24_merged.csv (전체), data/processed/gov24_youth.csv (youth_overlap=True)

청년 대상 정의 (잠정, 팀 합의 필요)
  - youth_overlap : 사용자구분에 '개인' 포함 + 대상연령(JA0110~JA0111)이 19~34세와 겹침
                    전 연령 대상(예: 0~120세) 서비스도 포함되므로 범위가 넓다.
  - youth_focused : youth_overlap 중 대상연령 시작 15세 이상, 종료 45세 이하 (청년 집중 서비스)
  - youth_age_unknown : 개인 대상인데 대상연령 미기재

실행: python -m src.preprocess.gov24_youth
"""
from __future__ import annotations

import pandas as pd

from ..collect._common import PROCESSED, RAW
from ..collect.gov24_codes import JA_CODES

YOUTH_MIN, YOUTH_MAX = 19, 34
FOCUS_START_MIN, FOCUS_END_MAX = 15, 45


def main() -> None:
    src = RAW / "gov24"
    lst = pd.read_csv(src / "gov24_serviceList.csv", dtype=str)
    det = pd.read_csv(src / "gov24_serviceDetail.csv", dtype=str)
    cond = pd.read_csv(src / "gov24_supportConditions.csv", dtype=str)

    det_cols = ["서비스ID", "서비스목적", "구비서류", "문의처", "온라인신청사이트URL", "법령", "자치법규", "행정규칙"]
    df = lst.merge(det[det_cols], on="서비스ID", how="left", validate="one_to_one")
    cond = cond.drop(columns=["서비스명"]).rename(columns={k: f"{k}_{v}" for k, v in JA_CODES.items()})
    df = df.merge(cond, on="서비스ID", how="left", validate="one_to_one")

    a0 = pd.to_numeric(df["JA0110_대상연령(시작)"], errors="coerce")
    a1 = pd.to_numeric(df["JA0111_대상연령(종료)"], errors="coerce")
    personal = df["사용자구분"].fillna("").str.contains("개인")
    df["youth_age_unknown"] = personal & (a0.isna() | a1.isna())
    df["youth_overlap"] = personal & (a0 <= YOUTH_MAX) & (a1 >= YOUTH_MIN)
    df["youth_focused"] = df["youth_overlap"] & (a0 >= FOCUS_START_MIN) & (a1 <= FOCUS_END_MAX)
    df["has_online_url"] = df["온라인신청사이트URL"].notna()

    PROCESSED.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED / "gov24_merged.csv", index=False, encoding="utf-8-sig")
    df[df["youth_overlap"]].to_csv(PROCESSED / "gov24_youth.csv", index=False, encoding="utf-8-sig")
    print(f"전체 {len(df):,} / 청년 연령 겹침 {int(df['youth_overlap'].sum()):,} "
          f"/ 청년 집중 {int(df['youth_focused'].sum()):,} / 연령 미기재(개인) {int(df['youth_age_unknown'].sum()):,}")


if __name__ == "__main__":
    main()
