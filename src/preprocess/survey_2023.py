"""2023 고립·은둔 청년 실태조사에 2024 청년삶 F7·F9 은둔 정의를 적용해 수요 테이블을 만든다.

입력: data/raw/isolated_youth_survey_2023/*.xlsx (읽기 전용)
출력: data/processed/demand_2023_hikikomori.csv  (은둔 정의 충족 + 은둔 본조사 응답자, Q21)
      data/processed/demand_2023_isolated_aux.csv (고립 본조사 시트 중 19~34세 Q21, 보조 비교용)

은둔 정의 (2024 청년삶 파일설계서 358 외출 빈도 ⑤~⑧, 360 은둔 생활 이유, 19~34세)
  2023 1차조사 sq3 ∈ {1,2,3,4}  : 2024 358번 ⑤~⑧과 같은 보기, 코딩 방향이 반대
  sq3a1 ∉ {5,6}                 : 장애·몸이 불편 / 임신·출산 제외
  sq1a1 ∈ {1,2,3}               : 19~34세
  pid로 은둔 본조사 시트와 내부 결합 → 4,448명

고립 보조집단: 고립 본조사 시트 응답자에 연령(sq1a1 ∈ {1,2,3})만 적용 → 3,032명. 은둔 판별 기준은 적용하지 않는다.

가중치가 없는 조사이므로 응답자 비율을 모집단 비율로 해석하지 않는다.
실행: python -m src.preprocess.survey_2023
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pandas as pd

from ..collect._common import PROCESSED
from .keywords import Q21_LABELS
from .raw_manifest import SURVEY_DIR, survey_input_files
from .xlsx_reader import read_sheet

EXPECTED_HIKI = 4448
EXPECTED_ISO_19_34 = 3032   # 2026-11-02 STEP 00 보완에서 확인한 값
OUTING_CODES = {"1", "2", "3", "4"}
EXCLUDED_REASONS = {"5", "6"}
AGE_CODES = {"1", "2", "3"}
Q21_COLS = {f"q21{i}": f"q21_{i:02d}" for i in range(1, 11)}
MISSING = {".", ""}


def survey_path() -> Path:
    files = survey_input_files()
    if len(files) != 1:
        raise FileNotFoundError(f"{SURVEY_DIR}에 xlsx가 1개여야 합니다: {files}")
    return files[0]


@lru_cache(maxsize=None)
def load_sheet(name: str) -> pd.DataFrame:
    return read_sheet(survey_path(), name)


def hikikomori_mask(screen: pd.DataFrame) -> pd.Series:
    return (screen["sq3"].isin(OUTING_CODES)
            & ~screen["sq3a1"].isin(EXCLUDED_REASONS)
            & screen["sq1a1"].isin(AGE_CODES))


def _q21(df: pd.DataFrame) -> pd.DataFrame:
    out = df[["pid", *Q21_COLS]].rename(columns=Q21_COLS)
    for c in Q21_COLS.values():
        out[c] = out[c].where(~out[c].isin(MISSING))
    return out


def build_demand() -> tuple[pd.DataFrame, dict]:
    screen = load_sheet("(데이터)1차조사")
    hiki = load_sheet("(데이터)은둔")
    m = hikikomori_mask(screen)
    keep = screen.loc[m, ["pid", "sq1a1", "sq3", "sq3a1", "gubun"]]
    demand = keep.merge(_q21(hiki), on="pid", how="inner", validate="one_to_one")
    counts = {
        "screen_rows": len(screen),
        "definition_met_in_screen": int(m.sum()),
        "hiki_sheet_rows": len(hiki),
        "linked_definition_and_hiki_sheet": len(demand),
        "hiki_sheet_not_meeting_definition": int((~hiki["pid"].isin(keep["pid"])).sum()),
    }
    if len(demand) != EXPECTED_HIKI:
        raise ValueError(f"은둔 정의 적용 결과 {len(demand)}명 (기대 {EXPECTED_HIKI}명). 중단하고 보고할 것. {counts}")
    return demand, counts


def build_isolated_aux() -> tuple[pd.DataFrame, dict]:
    """고립 본조사 응답자 중 19~34세만. 은둔 판별 기준(sq3·sq3a1)은 적용하지 않는다."""
    iso = load_sheet("(데이터)고립")
    screen = load_sheet("(데이터)1차조사")[["pid", "sq1a1", "sq3", "sq3a1", "gubun"]]
    linked = screen.merge(_q21(iso), on="pid", how="inner", validate="one_to_one")
    aux = linked[linked["sq1a1"].isin(AGE_CODES)].reset_index(drop=True)
    counts = {"iso_sheet_rows": len(iso), "iso_linked_to_screen": len(linked), "iso_age_19_34": len(aux),
              "iso_age_35_39_excluded": int((linked["sq1a1"] == "4").sum()),
              "overlap_with_hiki_sheet": int(iso["pid"].isin(load_sheet("(데이터)은둔")["pid"]).sum())}
    if len(aux) != EXPECTED_ISO_19_34:
        raise ValueError(f"고립 19~34세 {len(aux)}명 (기대 {EXPECTED_ISO_19_34}명). {counts}")
    return aux, counts


def q21_label_table() -> pd.DataFrame:
    return pd.DataFrame({"need_area": list(Q21_LABELS), "label": list(Q21_LABELS.values())})


def main() -> None:
    demand, counts = build_demand()
    aux, iso_counts = build_isolated_aux()
    PROCESSED.mkdir(parents=True, exist_ok=True)
    demand.to_csv(PROCESSED / "demand_2023_hikikomori.csv", index=False, encoding="utf-8-sig")
    aux.to_csv(PROCESSED / "demand_2023_isolated_aux.csv", index=False, encoding="utf-8-sig")
    print(counts, iso_counts)


if __name__ == "__main__":
    main()
