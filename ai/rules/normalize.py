"""규칙 모듈용 값 정규화. 사용자 입력(나이·지역)과 서비스 원자료(연령 경계·시도)를 같은 형식으로 맞춤.

서비스 쪽 표기 규칙은 docs/services_master_spec.md 기준임.
- 연령 경계: 정수 / `NA`(열린 경계) / `N`(원문 근거 없음)
- 시도: 행정구역 전체 이름. 중앙 사업은 빈칸. 통합 지자체 '전남광주통합특별시'는 광주·전남 두 곳으로 봄
"""
from __future__ import annotations

import logging
import math
import re
from contextlib import contextmanager
from contextvars import ContextVar
from functools import lru_cache
from pathlib import Path

import pandas as pd
import yaml

from extract.schema import _SIDO_ALIAS, AGE_RANGE, SIDO

NATIONWIDE = "전국"
# 행정구역 통합 표기 (master에 555건). 광주·전남 어느 쪽 거주자에게도 해당
MERGED_SIDO = {"전남광주통합특별시": {"광주", "전남"}}


def _is_nan(v) -> bool:
    return v is None or (isinstance(v, float) and math.isnan(v))


def norm_age(v) -> int | None:
    """사용자 나이. 정수 또는 숫자 문자열만 받고 AGE_RANGE(10~99) 밖이면 None. bool·실수·기타는 None."""
    if isinstance(v, bool):
        return None
    if isinstance(v, str) and re.fullmatch(r"\d{1,3}", v.strip()):
        v = int(v.strip())
    return v if isinstance(v, int) and AGE_RANGE[0] <= v <= AGE_RANGE[1] else None


def parse_bound(v) -> tuple[str, int | None]:
    """서비스 연령 경계 → ("int", n) | ("open", None) | ("none", None).

    "NA" 문자열과 NaN/None은 열린 경계임(pandas가 "NA"를 NaN으로 읽음). "N"·빈 문자열·해석 불가는 정보 없음.
    """
    if isinstance(v, bool):
        return ("none", None)
    if _is_nan(v):
        return ("open", None)
    if isinstance(v, int):
        return ("int", v)
    if isinstance(v, float):
        return ("int", int(v)) if v.is_integer() else ("none", None)
    if isinstance(v, str):
        s = v.strip()
        if s == "NA":
            return ("open", None)
        if re.fullmatch(r"\d+", s):
            return ("int", int(s))
    return ("none", None)


def norm_sido(v) -> set[str]:
    """사용자·서비스 시도 → SIDO 약칭 집합. 약칭, 정식 명칭, 통합 표기, '전국'을 받음. 비었거나 모르는 값은 빈 집합."""
    if not isinstance(v, str):
        return set()
    s = v.strip()
    if not s:
        return set()
    if s == NATIONWIDE:
        return {NATIONWIDE}
    if s in SIDO:
        return {s}
    if s in MERGED_SIDO:
        return set(MERGED_SIDO[s])
    a = _SIDO_ALIAS.get(s)
    return {a} if a in SIDO else set()


# ── 시군구 ────────────────────────────────────────────────────────
_AI_DIR = Path(__file__).resolve().parents[1]
_SGG_MAP = ContextVar("services_sgg_map", default=None)
_log = logging.getLogger(__name__)


def norm_sgg(v) -> str | None:
    """시군구 이름 → 앞뒤 공백 제거한 문자열. 비었거나 N·NA·nan이면 None."""
    if not isinstance(v, str):
        return None
    s = v.strip()
    return None if s in ("", "N", "NA", "nan") else s


def build_sgg_map(services: pd.DataFrame, services_path: str | Path) -> dict[str, frozenset[str]]:
    """검색에 사용한 서비스 표의 시도·시군구 쌍. 통합 표기는 광주·전남에 모두 반영함."""
    cols = ("region_sido", "region_sgg")
    missing = [col for col in cols if col not in services]
    if missing:
        _log.warning("%s: %s 열이 없어 시군구를 확인할 수 없습니다", services_path, ", ".join(missing))
        return {}
    districts: dict[str, set[str]] = {}
    for raw_sido, raw_sgg in zip(*(services[col] for col in cols)):
        sgg = norm_sgg(raw_sgg)
        if sgg:
            for sido in norm_sido(raw_sido) - {NATIONWIDE}:
                districts.setdefault(sido, set()).add(sgg)
    return {sido: frozenset(names) for sido, names in districts.items()}


@contextmanager
def use_sgg_map(districts):
    """세션 행동을 재적용하는 동안 해당 Pipeline의 시군구만 검증함."""
    token = _SGG_MAP.set(districts)
    try:
        yield
    finally:
        _SGG_MAP.reset(token)


@lru_cache(maxsize=8)
def _read_sgg_map(path: Path, mtime_ns: int, size: int) -> dict[str, frozenset[str]]:
    services = (pd.read_parquet(path) if path.suffix == ".parquet"
                else pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig"))
    return build_sgg_map(services, path)


def sgg_map(services_path=None) -> dict[str, frozenset[str]]:
    """Pipeline 밖에서는 기본 config의 마스터를 읽고 절대경로와 파일 상태별로 캐시함."""
    if services_path is None:
        active = _SGG_MAP.get()
        if active is not None:
            return active
        cfg = yaml.safe_load((_AI_DIR / "configs/config.yaml").read_text(encoding="utf-8"))
        services_path = _AI_DIR / cfg["paths"]["services"]
    path = Path(services_path).resolve()
    try:
        stat = path.stat()
        return _read_sgg_map(path, stat.st_mtime_ns, stat.st_size)
    except (OSError, ValueError, ImportError) as exc:
        raise RuntimeError(f"시군구 확인용 서비스 마스터를 읽을 수 없습니다: {path} ({exc})") from exc


def valid_sgg_for(sido, sgg) -> str | None:
    """입력한 시·도 또는 통합 시·도의 실제 시군구 이름을 돌려준다."""
    if not isinstance(sido, str) or (sido not in SIDO and sido not in MERGED_SIDO):
        return None
    s = norm_sgg(sgg)
    areas = norm_sido(sido) - {NATIONWIDE}
    if s is None or not areas:
        return None
    districts = sgg_map()
    return s if any(s in districts.get(area, ()) for area in areas) else None
