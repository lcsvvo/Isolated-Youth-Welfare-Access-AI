"""서비스 표 읽기. 실제 마스터(data/processed/services_master.csv)와 가짜 sample을 같은 방식으로 읽는다.

읽기 규칙 (docs/services_master_spec.md 기준)
- encoding utf-8-sig, dtype=str: 읽을 때는 형 추론을 하지 않는다(수치 열에 N·NA 토큰이 섞여 있어 추론하면 의미가 바뀜).
  대신 아래 열 스키마(NUMERIC_TOKEN_COLS·CATEGORY_COLS)로 값 어휘를 검증한다.
- keep_default_na=False, na_values=[""]: 빈 칸만 NaN. 마스터의 `N`(원문 근거 없음), `NA`(열린 경계, 예: age_min/age_max,
  cost), `U`(youth_scope 미확정)는 pandas 기본 NA 목록에 걸려 의미가 바뀌므로 그대로 문자열로 보존한다.
- age_min/age_max도 문자열 그대로 둔다("19", "NA", "N"). rules/normalize.parse_bound가 문자열을 정수·열린 경계·정보 없음으로 해석한다.
- 빈 칸이 NaN인 열: region_sido(중앙 사업 461건), online_bokjiro 등. 규칙 코드는 NaN을 None으로 취급한다.
"""
import json
import logging
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

READ_KWARGS = dict(dtype=str, encoding="utf-8-sig", keep_default_na=False, na_values=[""])
ID_COL = "servId"
REQUIRED_COLS = ("servId", "servNm", "summary_text", "target_text", "criteria_text", "benefit_text",
                 "apply_text", "app_domain", "region_sido", "age_min", "age_max")
ALWAYS_REQUIRED = ("servId", "servNm")

# 열 스키마. 형 변환은 하지 않고(의미 보존) 값 어휘만 검증한다. 실제 해석은 규칙 단계(rules/normalize 등)가 한다.
# - 수치형(정수 문자열 + 허용 토큰): N = 원문 근거 없음, NA = 열린 경계(나이 상·하한 없음)
NUMERIC_TOKEN_COLS = {
    "age_min": {"N", "NA"}, "age_max": {"N", "NA"},
    "apply_barrier": {"N"}, "use_barrier": {"N"}, "apply_outing": {"N"}, "use_outing": {"N"},
    "outing_need_min": {"N"}, "outing_need_min_v2": {"N"},
}
# - 범주형(정해진 코드만). U = youth_scope 미확정
CATEGORY_COLS = {"source": {"central", "local"}, "youth_scope": {"A", "B", "C", "U"}}
# - 그 밖의 열은 원문·자유 텍스트 또는 '|' 구분 라벨(app_domain 등)로 문자열 그대로 둔다. 빈 칸만 결측(NaN).
_INT = r"-?\d+"


def schema_errors(df):
    """스키마에 맞지 않는 값 {열: [값...]}. 빈 칸(NaN)은 결측으로 허용한다. 없는 열은 건너뛴다."""
    errors = {}
    for col, tokens in NUMERIC_TOKEN_COLS.items():
        if col in df:
            v = df[col].dropna()
            bad = v[~v.str.fullmatch(_INT) & ~v.isin(tokens)]
            if len(bad):
                errors[col] = sorted(bad.unique().tolist())[:10]
    for col, codes in CATEGORY_COLS.items():
        if col in df:
            v = df[col].dropna()
            bad = v[~v.isin(codes)]
            if len(bad):
                errors[col] = sorted(bad.unique().tolist())[:10]
    return errors


def meta_path(path):
    path = Path(path)
    return path.with_name(path.stem + ".meta.json")


def load_services(path, strict=None):
    """서비스 표를 읽어 검증한다. servId는 비어 있지 않고 유일해야 한다.

    strict=None이면 옆에 `<이름>.meta.json`이 있을 때(실제 마스터) strict. strict면 REQUIRED_COLS 누락과
    meta의 rows 불일치를 오류로 처리하고, 아니면(sample) servId·servNm만 필수이고 나머지 누락은 경고한다.
    """
    path = Path(path)
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path, **READ_KWARGS)
    meta = meta_path(path)
    if strict is None:
        strict = meta.exists()

    absent = [c for c in ALWAYS_REQUIRED if c not in df]
    if absent:
        raise ValueError(f"{path.name}: 필수 칼럼 없음 {absent}")
    missing = [c for c in REQUIRED_COLS if c not in df]
    if missing:
        if strict:
            raise ValueError(f"{path.name}: 검색·규칙 필수 칼럼 없음 {missing}")
        log.warning("%s: 선택 칼럼 없음(검색·규칙에서 건너뜀): %s", path.name, ", ".join(missing))

    ids = df[ID_COL]
    if ids.isna().any() or (ids.str.strip() == "").any():
        raise ValueError(f"{path.name}: servId가 빈 행이 있음")
    dup = ids[ids.duplicated()].unique().tolist()
    if dup:
        raise ValueError(f"{path.name}: servId 중복 {len(dup)}건 (예: {dup[:3]})")

    bad = schema_errors(df)
    if bad:
        msg = f"{path.name}: 스키마 밖 값 {bad}"
        if strict:
            raise ValueError(msg)
        log.warning(msg)

    if meta.exists():
        rows = json.loads(meta.read_text(encoding="utf-8")).get("rows")
        if rows is not None and rows != len(df):
            msg = f"{path.name}: 행 수 {len(df)} ≠ {meta.name}의 rows {rows}"
            if strict:
                raise ValueError(msg)
            log.warning(msg)
    return df
