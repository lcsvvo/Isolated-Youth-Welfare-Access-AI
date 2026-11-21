"""STEP 00 입력 원자료의 SHA-256 목록. 원본 불변 테스트의 기준이 된다.

실행: python -m src.preprocess.raw_manifest  (logs/validation/raw_hashes_step00.json 갱신)
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ..collect._common import RAW, ROOT

MANIFEST = ROOT / "logs" / "validation" / "raw_hashes_step00.json"
SURVEY_DIR = RAW / "isolated_youth_survey_2023"
INPUTS = [
    "ssis/central_list.csv",
    "ssis/central_detail_full.csv",
    "ssis/local_list.csv",
    "ssis/local_detail_full.csv",
    "gov24/gov24_serviceList.csv",
    "gov24/gov24_serviceDetail.csv",
    "gov24/gov24_supportConditions.csv",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def survey_input_files() -> list[Path]:
    # Excel에서 원본을 열어 둔 동안 생성되는 잠금 파일은 조사 원자료가 아니다.
    return sorted(p for p in SURVEY_DIR.glob("*.xlsx") if not p.name.startswith("~$"))


def input_files() -> list[Path]:
    return [RAW / p for p in INPUTS] + survey_input_files()


def current_hashes() -> dict[str, str]:
    return {p.relative_to(ROOT).as_posix(): sha256(p) for p in input_files()}


def write_manifest() -> dict[str, str]:
    hashes = current_hashes()
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(hashes, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return hashes


if __name__ == "__main__":
    for k, v in write_manifest().items():
        print(v[:12], k)
