"""수집 공통 유틸: 키 로드, 재시도 GET, 수집 로그."""
from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"


def load_env() -> None:
    """저장소 루트의 .env, .env.txt 순서로 읽는다. 이미 설정된 환경변수는 덮어쓰지 않는다."""
    for name in (".env", ".env.txt"):
        path = ROOT / name
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def require_key(name: str) -> str:
    load_env()
    key = os.environ.get(name, "")
    if not key:
        raise SystemExit(f"[중단] .env 또는 .env.txt에 {name}를 넣어 주세요.")
    return key


def get_json(base: str, params: dict, raw_key: tuple[str, str] | None = None,
             retries: int = 3, timeout: int = 60) -> dict:
    """GET 후 JSON을 반환한다. raw_key=(이름, 값)은 이미 URL 인코딩된 키를 재인코딩 없이 붙인다."""
    query = urllib.parse.urlencode(params)
    if raw_key:
        query += f"&{raw_key[0]}={raw_key[1]}"
    url = f"{base}?{query}"
    last = ""
    for i in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", "ignore")[:300]
            if e.code in (401, 403):
                raise SystemExit(f"[중단] 인증 오류 {e.code}: {body}") from None
            last = f"HTTP {e.code}: {body}"
        except Exception as e:  # noqa: BLE001
            last = repr(e)
        time.sleep(2 * (i + 1))
    raise SystemExit(f"[중단] 요청 실패({retries}회): {last}")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_log(folder: Path, source: str, endpoints: dict, files: list[Path]) -> None:
    """data/raw/<출처>/_collection.json 에 조회조건·건수·해시를 남긴다 (키는 기록하지 않음)."""
    log = {
        "source": source,
        "collected_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "endpoints": endpoints,
        "files": [{"name": p.name, "bytes": p.stat().st_size, "sha256": sha256(p)} for p in files],
    }
    (folder / "_collection.json").write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")
