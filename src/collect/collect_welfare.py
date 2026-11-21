"""
반 걸음 — 공공데이터포털 API 3종 수집기
  1) 한국사회보장정보원_중앙부처복지서비스 (개발계정 일 100건)
  2) 한국사회보장정보원_지자체복지서비스   (개발계정 일 1,000건)
  # 3) 성평등가족부_청소년상담복지센터       (개발계정 일 10,000건)  -> 이건 청소년이라 일단 제외

특징
  - 받은 응답은 data/raw/ 아래에 원문 XML/JSON 그대로 저장 → 이미 받은 건 다시 호출하지 않음
    (일일 한도에 걸리면 멈추고, 다음 날 같은 명령을 다시 실행하면 이어서 받음)
  - 필드명을 하드코딩하지 않고 응답 XML을 통째로 파싱 → API 필드가 바뀌어도 누락 없음
  - 결과: data/out/*.csv (평탄화), data/out/*.jsonl (중첩 목록 보존)

사용법
  pip install requests pandas
  .env 파일(아래 .env.example 참고)에 키 입력
  python collect_welfare.py probe            # API별 1건만 호출해 응답 구조 확인 (먼저 실행!)
  python collect_welfare.py central          # 중앙부처 목록 + 상세
  python collect_welfare.py local            # 지자체 목록 + 상세 (청년 관련부터)
  python collect_welfare.py youth            # 청소년상담복지센터 전체
  python collect_welfare.py all              # 위 세 개 순서대로
  python collect_welfare.py export           # 저장된 원문만으로 CSV/JSONL 다시 생성 (호출 0건)

옵션
  --max-calls N   이번 실행에서 쓸 최대 호출 수 (기본: API별 일일 한도)
  --sleep S       호출 간 대기(초), 기본 0.3
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import requests

# ──────────────────────────────────────────────────────────────
# 설정
# ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "out"

CENTRAL_BASE = "https://apis.data.go.kr/B554287/NationalWelfareInformationsV001"
LOCAL_BASE = "https://apis.data.go.kr/B554287/LocalGovernmentWelfareInformations"


DAILY_LIMIT = {"central": 100, "local": 1000, "youth": 10000}
PAGE_SIZE = {"central": 500, "local": 500, "youth": 1000}

# 생애주기 코드 004 = 청년 (지자체 상세 수집 우선순위용)
YOUTH_LIFE_CODE = "004"
YOUTH_WORDS = ("청년", "청소년", "고립", "은둔", "마음", "심리", "1인가구")

# 일일 한도 초과 등 '오늘은 더 못 받는' 에러 코드
QUOTA_CODES = {"22", "LIMITED_NUMBER_OF_SERVICE_REQUESTS_EXCEEDS_ERROR"}
RETRY_CODES = {"01", "05", "23"}  # 내부오류, 타임아웃, 초당 한도


def load_env(path: Path = ROOT / ".env") -> None:
    """python-dotenv 없이 .env 읽기."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def key_for(api: str) -> str:
    name = {"central": "CENTRAL_KEY", "local": "LOCAL_KEY", "youth": "YOUTH_KEY"}[api]
    key = os.environ.get(name) or os.environ.get("DATA_GO_KR_KEY")
    if not key:
        sys.exit(f"[중단] .env에 {name} 또는 DATA_GO_KR_KEY를 넣어 주세요 (Decoding 키).")
    return key


# ──────────────────────────────────────────────────────────────
# XML/JSON → dict
# ──────────────────────────────────────────────────────────────
def xml_to_obj(el: ET.Element):
    """자식이 없으면 텍스트, 있으면 dict. 같은 태그가 반복되면 list."""
    children = list(el)
    if not children:
        return (el.text or "").strip()
    out: dict = {}
    for c in children:
        v = xml_to_obj(c)
        if c.tag in out:
            if not isinstance(out[c.tag], list):
                out[c.tag] = [out[c.tag]]
            out[c.tag].append(v)
        else:
            out[c.tag] = v
    return out


def parse_body(text: str):
    text = text.strip()
    if text.startswith("{"):
        return json.loads(text)
    root = ET.fromstring(text)
    return {root.tag: xml_to_obj(root)}


def find_first(obj, keys: tuple[str, ...]):
    """중첩 구조 어디에 있든 첫 번째로 나오는 키 값을 찾음."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in keys:
                return v
        for v in obj.values():
            r = find_first(v, keys)
            if r is not None:
                return r
    elif isinstance(obj, list):
        for v in obj:
            r = find_first(v, keys)
            if r is not None:
                return r
    return None


def error_code(obj) -> str | None:
    """정상 응답이면 None, 아니면 에러 코드/메시지."""
    reason = find_first(obj, ("returnReasonCode",))  # 게이트웨이 에러 형식
    if reason and str(reason) not in ("00", "0"):
        msg = find_first(obj, ("returnAuthMsg", "errMsg")) or ""
        return f"{reason}:{msg}"
    code = find_first(obj, ("resultCode",))
    if code is not None and str(code) not in ("0", "00", "000", "INFO-000"):
        msg = find_first(obj, ("resultMessage", "resultMsg")) or ""
        return f"{code}:{msg}"
    return None


def as_list(v) -> list:
    if v is None or v == "":
        return []
    return v if isinstance(v, list) else [v]


# ──────────────────────────────────────────────────────────────
# 호출기 (한도·재시도·원문 저장)
# ──────────────────────────────────────────────────────────────
class QuotaExceeded(Exception):
    pass


class Caller:
    def __init__(self, api: str, max_calls: int | None, sleep: float):
        self.api = api
        self.key = key_for(api)
        self.budget = max_calls if max_calls is not None else DAILY_LIMIT[api]
        self.used = 0
        self.sleep = sleep
        self.s = requests.Session()

    def get(self, url: str, params: dict, cache: Path) -> dict:
        """cache 파일이 있으면 호출 없이 읽고, 없으면 호출 후 저장."""
        if cache.exists():
            return parse_body(cache.read_text(encoding="utf-8"))
        if self.used >= self.budget:
            raise QuotaExceeded(f"이번 실행 호출 한도 {self.budget}건 소진")

        q = {"serviceKey": self.key, **params}
        last_err = None
        for attempt in range(4):
            try:
                r = self.s.get(url, params=q, timeout=30)
                self.used += 1
                text = r.content.decode("utf-8", errors="replace")
                obj = parse_body(text)
            except (requests.RequestException, ET.ParseError, json.JSONDecodeError) as e:
                last_err = repr(e)
                time.sleep(2 ** attempt)
                continue

            err = error_code(obj)
            if err is None:
                cache.parent.mkdir(parents=True, exist_ok=True)
                cache.write_text(text, encoding="utf-8")
                time.sleep(self.sleep)
                return obj
            code = err.split(":")[0]
            if code in QUOTA_CODES or "LIMITED_NUMBER_OF_SERVICE_REQUESTS_EXCEEDS" in err:
                raise QuotaExceeded(f"일일 트래픽 초과 ({err})")
            if code in RETRY_CODES:
                last_err = err
                time.sleep(2 ** attempt)
                continue
            raise RuntimeError(f"[{self.api}] API 오류 {err}\n  url={url}\n  params={params}")
        raise RuntimeError(f"[{self.api}] 재시도 실패: {last_err}")


# ──────────────────────────────────────────────────────────────
# 1) 중앙부처
# ──────────────────────────────────────────────────────────────
def central_list(c: Caller) -> list[dict]:
    items, page = [], 1
    while True:
        obj = c.get(
            f"{CENTRAL_BASE}/NationalWelfarelistV001",
            {"callTp": "L", "pageNo": page, "numOfRows": PAGE_SIZE["central"], "srchKeyCode": "003"},
            RAW / "central" / "list" / f"page_{page:03d}.xml",
        )
        batch = as_list(find_first(obj, ("servList",)))
        items += batch
        total = int(find_first(obj, ("totalCount",)) or 0)
        print(f"  [central] 목록 {page}페이지: {len(batch)}건 (누적 {len(items)}/{total})")
        if not batch or len(items) >= total:
            break
        page += 1
    return items


def central_detail(c: Caller, items: list[dict]) -> None:
    ids = [i.get("servId") for i in items if i.get("servId")]
    done = sum((RAW / "central" / "detail" / f"{s}.xml").exists() for s in ids)
    print(f"  [central] 상세 {done}/{len(ids)} 이미 있음, {len(ids) - done}건 남음")
    for s in ids:
        c.get(
            f"{CENTRAL_BASE}/NationalWelfaredetailedV001",
            {"callTp": "D", "servId": s},
            RAW / "central" / "detail" / f"{s}.xml",
        )


# ──────────────────────────────────────────────────────────────
# 2) 지자체
# ──────────────────────────────────────────────────────────────
def local_list(c: Caller) -> list[dict]:
    items, page = [], 1
    while True:
        obj = c.get(
            f"{LOCAL_BASE}/LcgvWelfarelist",
            {"pageNo": page, "numOfRows": PAGE_SIZE["local"]},
            RAW / "local" / "list" / f"page_{page:03d}.xml",
        )
        batch = as_list(find_first(obj, ("servList",)))
        items += batch
        total = int(find_first(obj, ("totalCount",)) or 0)
        print(f"  [local] 목록 {page}페이지: {len(batch)}건 (누적 {len(items)}/{total})")
        if not batch or len(items) >= total:
            break
        page += 1
    return items


def youth_priority(item: dict) -> int:
    """0 = 청년 생애주기, 1 = 청년 관련 키워드, 2 = 생애주기 미지정, 3 = 나머지."""
    blob = json.dumps(item, ensure_ascii=False)
    life = str(item.get("lifeArray", "")) + str(item.get("lifeNmArray", ""))
    if YOUTH_LIFE_CODE in str(item.get("lifeArray", "")) or "청년" in life:
        return 0
    if any(w in blob for w in YOUTH_WORDS):
        return 1
    if not life.strip():
        return 2
    return 3


def local_detail(c: Caller, items: list[dict]) -> None:
    items = sorted(items, key=youth_priority)
    ids = [i.get("servId") for i in items if i.get("servId")]
    done = sum((RAW / "local" / "detail" / f"{s}.xml").exists() for s in ids)
    n_youth = sum(youth_priority(i) <= 1 for i in items)
    print(f"  [local] 상세 {done}/{len(ids)} 이미 있음 (청년 관련 {n_youth}건을 먼저 받음)")
    for s in ids:
        c.get(
            f"{LOCAL_BASE}/LcgvWelfaredetailed",
            {"servId": s},
            RAW / "local" / "detail" / f"{s}.xml",
        )


# ──────────────────────────────────────────────────────────────
# 3) 청소년상담복지센터
# ──────────────────────────────────────────────────────────────
def youth_all(c: Caller) -> list[dict]:
    url = os.environ.get("YOUTH_URL")
    if not url:
        sys.exit("[중단] .env에 YOUTH_URL(청소년상담복지센터 요청주소)을 넣어 주세요.")
    items, page = [], 1
    while True:
        obj = c.get(
            url,
            {"pageNo": page, "numOfRows": PAGE_SIZE["youth"], "type": "xml"},
            RAW / "youth" / f"page_{page:03d}.xml",
        )
        batch = as_list(find_first(obj, ("item",)))
        items += batch
        total = int(find_first(obj, ("totalCount",)) or 0)
        print(f"  [youth] {page}페이지: {len(batch)}건 (누적 {len(items)}/{total})")
        if not batch or (total and len(items) >= total):
            break
        page += 1
    return items


# ──────────────────────────────────────────────────────────────
# 내보내기 (저장된 원문 → CSV / JSONL)
# ──────────────────────────────────────────────────────────────
def flatten(d: dict) -> dict:
    """한 단계 dict는 '상위_하위'로 펼치고, 목록·중첩은 JSON 문자열로."""
    row = {}
    for k, v in d.items():
        if isinstance(v, dict) and all(not isinstance(x, (dict, list)) for x in v.values()):
            for k2, v2 in v.items():
                row[f"{k}_{k2}"] = v2
        elif isinstance(v, (dict, list)):
            row[k] = json.dumps(v, ensure_ascii=False)
        else:
            row[k] = v
    return row


def read_cached(folder: Path, list_key: str) -> list[dict]:
    rows = []
    for f in sorted(folder.glob("*.xml")):
        obj = parse_body(f.read_text(encoding="utf-8"))
        rows += as_list(find_first(obj, (list_key,)))
    return rows


def read_details(folder: Path) -> list[dict]:
    rows = []
    for f in sorted(folder.glob("*.xml")):
        obj = parse_body(f.read_text(encoding="utf-8"))
        root = next(iter(obj.values()))
        # 응답 루트 바로 아래 내용(헤더 제외)을 한 행으로
        rec = {k: v for k, v in root.items() if k not in ("resultCode", "resultMessage")} \
            if isinstance(root, dict) else {"raw": root}
        rec.setdefault("servId", f.stem)
        rows.append(rec)
    return rows


def write(name: str, rows: list[dict]) -> None:
    import pandas as pd

    if not rows:
        print(f"  [export] {name}: 데이터 없음, 건너뜀")
        return
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / f"{name}.jsonl", "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    df = pd.DataFrame([flatten(r) for r in rows])
    df.to_csv(OUT / f"{name}.csv", index=False, encoding="utf-8-sig")  # 엑셀 한글 깨짐 방지
    print(f"  [export] {name}: {len(df)}행 × {df.shape[1]}열 → data/out/{name}.csv")


def export_all() -> None:
    write("central_list", read_cached(RAW / "central" / "list", "servList"))
    write("central_detail", read_details(RAW / "central" / "detail"))
    write("local_list", read_cached(RAW / "local" / "list", "servList"))
    write("local_detail", read_details(RAW / "local" / "detail"))
    write("youth_centers", read_cached(RAW / "youth", "item"))


# ──────────────────────────────────────────────────────────────
# probe: API별 1건만 호출해서 구조 확인
# ──────────────────────────────────────────────────────────────
def probe(sleep: float) -> None:
    def show(title, obj):
        print(f"\n=== {title} ===")
        err = error_code(obj)
        print("  상태:", "정상" if err is None else f"오류 {err}")
        print("  totalCount:", find_first(obj, ("totalCount",)))
        first = find_first(obj, ("servList", "item"))
        first = as_list(first)[0] if first else None
        if isinstance(first, dict):
            print("  첫 항목 필드:", list(first.keys()))
            print("  예시:", json.dumps(first, ensure_ascii=False)[:400])
        else:
            print("  원문 일부:", json.dumps(obj, ensure_ascii=False)[:600])

    tmp = RAW / "_probe"
    for api, url, params, f in [
        ("central", f"{CENTRAL_BASE}/NationalWelfarelistV001",
         {"callTp": "L", "pageNo": 1, "numOfRows": 3, "srchKeyCode": "003"}, "central.xml"),
        ("local", f"{LOCAL_BASE}/LcgvWelfarelist", {"pageNo": 1, "numOfRows": 3}, "local.xml"),
        ("youth", os.environ.get("YOUTH_URL", ""), {"pageNo": 1, "numOfRows": 3, "type": "xml"}, "youth.xml"),
    ]:
        if not url:
            print(f"\n=== {api} === YOUTH_URL 없음, 건너뜀")
            continue
        try:
            obj = Caller(api, 1, sleep).get(url, params, tmp / f)
            show(api, obj)
        except Exception as e:  # noqa: BLE001
            print(f"\n=== {api} === 실패: {e}")
        (tmp / f).unlink(missing_ok=True)  # probe 결과는 캐시로 남기지 않음


# ──────────────────────────────────────────────────────────────
def main() -> None:
    load_env()
    ap = argparse.ArgumentParser()
    ap.add_argument("target", choices=["probe", "central", "local", "youth", "all", "export"])
    ap.add_argument("--max-calls", type=int, default=None)
    ap.add_argument("--sleep", type=float, default=0.3)
    a = ap.parse_args()

    if a.target == "probe":
        probe(a.sleep)
        return
    if a.target == "export":
        export_all()
        return

    targets = ["central", "local", "youth"] if a.target == "all" else [a.target]
    for t in targets:
        c = Caller(t, a.max_calls, a.sleep)
        print(f"\n▶ {t} 수집 시작 (이번 실행 최대 {c.budget}건 호출)")
        try:
            if t == "central":
                central_detail(c, central_list(c))
            elif t == "local":
                local_detail(c, local_list(c))
            else:
                youth_all(c)
            print(f"  ✔ {t} 완료 (호출 {c.used}건)")
        except QuotaExceeded as e:
            print(f"  ⏸ {t} 중단: {e}. 내일 같은 명령으로 이어서 받으면 됩니다. (호출 {c.used}건)")
    export_all()


if __name__ == "__main__":
    main()
