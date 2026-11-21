"""규칙 검수 240건을 팀원 4명(R1~R4)에게 나누고, 입력 결과를 원본 검수 시트에 합치는 도구.

사용
- python -m ai.eval.rules_team_review split [--refresh]
- python -m ai.eval.rules_team_review merge [--apply]

원칙
- split: 선정 목록(selection.csv)을 정해진 할당량으로 배정함. 가능하면 같은 서비스는 같은 검수자에게 보냄
- 다른 유형의 입력 칸은 '-'(해당 없음)로 채움. 빈 칸은 미검수를 뜻함
- merge: 기본은 점검만 하고 원본 시트는 바꾸지 않음. --apply일 때만 원본의 빈 사람 입력 칸을 채움
- 원본에 이미 값이 있으면 덮어쓰지 않고 동일 또는 충돌로 기록함
- consensus_log.csv가 있으면 원검수와 정확히 대조한 뒤 합의 적용값을 비교함
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from ai.rules import overlay as OV

ROOT = Path(__file__).resolve().parents[2]
REVIEW_DIR = ROOT / "outputs" / "07_rules" / "review"
TEAM_DIR = REVIEW_DIR / "team"

REVIEWERS = ("R1", "R2", "R3", "R4")
TYPES = ("연령", "지역", "접근")
QUOTAS = {  # 검수자 순서 R1~R4
    "연령": (15,) * len(REVIEWERS),
    "지역": (20,) * len(REVIEWERS),
    "접근": (25,) * len(REVIEWERS),
}

SHEET_FILES = {"연령": OV.AGE_FILE, "지역": OV.REGION_FILE, "접근": OV.ACCESS_FILE}

COLUMNS = [
    "no", "review_type", "servId", "servNm", "입력할_칸", "선정_이유", "검수_대상_원문", "AI_제안",
    "decision", "human_min", "human_max",
    "region_rule", "region_area",
    "apply_visit", "apply_alt", "use_visit", "use_alt",
    "reason", "reviewer", "reviewed_at",
]
INPUT_COLS = {
    "연령": ["decision", "human_min", "human_max"],
    "지역": ["region_rule", "region_area"],
    "접근": ["apply_visit", "apply_alt", "use_visit", "use_alt"],
}
REQUIRED = {"연령": ["decision"], "지역": ["region_rule"], "접근": ["apply_visit", "apply_alt", "use_visit", "use_alt"]}
COMMON = ["reason", "reviewer", "reviewed_at"]

# overlay.py의 허용값을 그대로 연결함
ALLOWED = {
    "decision": OV.AGE_DECISIONS,
    "region_rule": OV.REGION_RULES,
    "apply_visit": OV.APPLY_VISIT,
    "apply_alt": OV.APPLY_ALT,
    "use_visit": OV.USE_VISIT,
    "use_alt": OV.USE_ALT,
}

GUIDE = {
    "연령": "decision(유지/수정/제외/보류), human_min, human_max, reason",
    "지역": "region_rule(거주 필수/거주 또는 재직·재학/신청자 거주 조건 아님/지역 제한 없음/거주 조건 불명), region_area, reason",
    "접근": "apply_visit(필수/선택/불필요/불명/서비스별 상이), apply_alt(있음/없음 명시/불명/해당 없음/서비스별 상이), "
            "use_visit(사용자 방문 필수/담당자 방문/선택/불필요/불명/서비스별 상이), use_alt(있음/없음 명시/불명/해당 없음/서비스별 상이), reason",
}

REPORT_COLS = ["file", "no", "review_type", "servId", "status", "action", "detail"]
ACCESS_CAP = 90  # 접근 원문 부분별 최대 글자 수 (전체 300자 이내)


# ---------- 입출력 ----------
def read_csv(path: Path) -> list[dict]:
    with open(path, encoding="utf-8-sig", newline="") as f:
        return [dict(r) for r in csv.DictReader(f)]


def read_header(path: Path) -> list[str]:
    with open(path, encoding="utf-8-sig", newline="") as f:
        return next(csv.reader(f))


def write_csv(path: Path, header: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header, lineterminator="\r\n", extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _cap(text: str, n: int) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= n else text[: n - 1] + "…"


# ---------- 배정 ----------
def assign(selection: list[dict]) -> dict[tuple[str, str], str]:
    """(review_type, servId) -> 검수자. 유형별 할당량을 맞추고 우선순위와 도메인을 분산함."""
    by_type: dict[str, list[dict]] = {t: [] for t in TYPES}
    for r in selection:
        by_type[r["review_type"]].append(r)
    for t in TYPES:
        if len(by_type[t]) != sum(QUOTAS[t]):
            raise ValueError(f"{t} 선정 {len(by_type[t])}건이 할당량 합계 {sum(QUOTAS[t])}와 다름")

    remain = {t: list(QUOTAS[t]) for t in TYPES}
    dom_count = {t: [Counter() for _ in REVIEWERS] for t in TYPES}
    reviewer_domains = [Counter() for _ in REVIEWERS]
    prio_count = {t: [Counter() for _ in REVIEWERS] for t in TYPES}
    total = [0] * len(REVIEWERS)
    out: dict[tuple[str, str], str] = {}

    def take(t: str, row: dict, i: int) -> None:
        out[(t, row["servId"])] = REVIEWERS[i]
        remain[t][i] -= 1
        dom_count[t][i][row["domain"]] += 1
        reviewer_domains[i][row["domain"]] += 1
        prio_count[t][i][row["prio"]] += 1
        total[i] += 1

    def balance(rows: list[dict], i: int) -> tuple[int, int, int, int, int]:
        return (
            sum(prio_count[r["review_type"]][i][r["prio"]] for r in rows),
            sum(dom_count[r["review_type"]][i][r["domain"]] for r in rows),
            sum(reviewer_domains[i][r["domain"]] for r in rows),
            total[i],
            i,
        )

    # 1) 여러 유형에 걸친 서비스는 같은 검수자에게 먼저 배정
    types_of: dict[str, list[dict]] = {}
    for t in TYPES:
        for r in by_type[t]:
            types_of.setdefault(r["servId"], []).append(r)
    for sid in sorted(types_of):
        rows = types_of[sid]
        if len(rows) < 2:
            continue
        cands = [i for i in range(len(REVIEWERS)) if all(remain[r["review_type"]][i] > 0 for r in rows)]
        if not cands:
            continue  # 공통 잔여 자리가 없으면 유형별 배정에서 처리
        i = min(cands, key=lambda k: balance(rows, k))
        for r in rows:
            take(r["review_type"], r, i)

    # 2) 나머지는 prio 내림차순으로 도메인 분산 라운드로빈
    for t in TYPES:
        rest = [r for r in by_type[t] if (t, r["servId"]) not in out]
        rest.sort(key=lambda r: (-int(r["prio"] or 0), r["servId"]))
        for r in rest:
            cands = [i for i in range(len(REVIEWERS)) if remain[t][i] > 0]
            i = min(cands, key=lambda k: balance([r], k))
            take(t, r, i)
    return out


# ---------- 시트 조회 ----------
def load_sheets(review_dir: Path) -> dict[str, dict[str, dict]]:
    return {t: {r["servId"]: r for r in read_csv(review_dir / SHEET_FILES[t])} for t in TYPES}


def source_text(t: str, s: dict) -> str:
    if t == "연령":
        return f"[근거] {s.get('evidence_snippet', '')} (출처: {s.get('source_field', '')})"
    if t == "지역":
        where = " ".join(x for x in (s.get("region_sido", ""), s.get("region_sgg", "")) if x)
        return f"[근거] {s.get('evidence', '')}" + (f" ({where})" if where else "")
    parts = [("신청", s.get("apply_evidence", "")), ("이용", s.get("use_evidence", "")), ("현재", s.get("current_reason", ""))]
    return " / ".join(f"[{k}] {_cap(v, ACCESS_CAP) or '-'}" for k, v in parts)


def ai_text(t: str, s: dict) -> str:
    g = lambda k: s.get(k, "") or "-"
    if t == "연령":
        return f"후보: {g('candidate_type')}, 추출 {s.get('extracted_min', '')}~{s.get('extracted_max', '')}세"
    if t == "지역":
        return f"{g('ai_region_rule')} ({g('ai_confidence')}), 지역 {g('ai_region_area')}"
    return (f"신청 방문 {g('ai_apply_visit')}, 신청 대체 {g('ai_apply_alt')} / "
            f"이용 방문 {g('ai_use_visit')}, 이용 대체 {g('ai_use_alt')}")


def has_filled_input(path: Path, t_cols=INPUT_COLS) -> bool:
    """기존 R 파일에 본인 유형 입력이 하나라도 있는지."""
    if not path.exists():
        return False
    for r in read_csv(path):
        t = r.get("review_type", "")
        cols = t_cols.get(t, []) + COMMON
        if any((r.get(c) or "").strip() not in ("", "-") for c in cols):
            return True
    return False


# ---------- split ----------
def split(selection_path: Path, review_dir: Path, team_dir: Path, refresh: bool = False) -> dict:
    selection = read_csv(selection_path)
    keys = [(r["review_type"], r["servId"]) for r in selection]
    if len(set(keys)) != len(keys):
        raise ValueError("selection에 (review_type, servId) 중복이 있음")
    sheets = load_sheets(review_dir)
    for t, sid in keys:
        if sid not in sheets[t]:
            raise ValueError(f"원본 시트에 없음: {t} {sid}")
    who = assign(selection)

    per: dict[str, list[dict]] = {r: [] for r in REVIEWERS}
    for sel in selection:
        t, sid = sel["review_type"], sel["servId"]
        sheet = sheets[t][sid]
        row = {c: "" for c in COLUMNS}
        for ot in TYPES:
            if ot != t:
                for c in INPUT_COLS[ot]:
                    row[c] = "-"
        row.update(
            review_type=t, servId=sid, servNm=sel["servNm"] or sheet.get("servNm", ""),
            입력할_칸=GUIDE[t], 선정_이유=sel["selection_reason"],
            검수_대상_원문=source_text(t, sheet), AI_제안=ai_text(t, sheet),
        )
        per[who[(t, sid)]].append(row)

    written = {}
    for rv in REVIEWERS:
        rows = sorted(per[rv], key=lambda r: (r["servId"], TYPES.index(r["review_type"])))
        for n, r in enumerate(rows, 1):
            r["no"] = n
        path = team_dir / f"{rv}.csv"
        if has_filled_input(path):
            if not refresh:
                written[rv] = ("건너뜀(입력 있음, --refresh로 별도 파일 생성)", path)
                continue
            path = team_dir / f"{rv}_refresh.csv"
        write_csv(path, COLUMNS, rows)
        written[rv] = ("작성", path)
    return {"assignment": who, "written": written, "rows": per}


# ---------- merge ----------
def _blank(v) -> bool:
    return (v or "").strip() == ""


def check_row(r: dict) -> tuple[str, str]:
    """(status, detail). status: 검수 완료 / 미검수 / 일부 입력 / 오류."""
    t = r.get("review_type", "")
    if t not in TYPES:
        return "오류", f"review_type 값 오류: {t!r}"
    own = INPUT_COLS[t] + ["reason"]
    vals = {c: (r.get(c) or "").strip() for c in own}
    errs = []
    for c in INPUT_COLS[t]:
        if vals[c] == "-":
            errs.append(f"{c}: 본인 유형 칸에 '-' 입력")
        elif c in ALLOWED and vals[c] and vals[c] not in ALLOWED[c]:
            errs.append(f"{c}: 허용값 아님({vals[c]})")
    if t == "연령":
        for c in ("human_min", "human_max"):
            v = vals[c]
            if v and v != "-":
                if not v.lstrip("-").isdigit() or not 0 <= int(v) <= 120:
                    errs.append(f"{c}: 0~120 정수 아님({v})")
    if errs:
        return "오류", "; ".join(errs)
    if all(v == "" for v in vals.values()):
        return "미검수", ""
    missing = [c for c in REQUIRED[t] if not vals[c]]
    if missing:
        return "일부 입력", "필수 칸 비어 있음: " + ", ".join(missing)
    return "검수 완료", ""


def merge(team_dir: Path, review_dir: Path, apply: bool = False) -> list[dict]:
    sheets = load_sheets(review_dir)
    # 합의 로그는 원검수와 적용값의 차이를 명시한다. 원본 변경 시 자동 덮어쓰기 금지.
    ledger_path = review_dir / "consensus_log.csv"
    ledger = {}
    if ledger_path.exists():
        for entry in read_csv(ledger_path):
            key = (entry["review_type"], entry["servId"])
            if key in ledger:
                raise ValueError(f"합의 로그 중복 키: {key}")
            ledger[key] = entry
    headers = {t: read_header(review_dir / SHEET_FILES[t]) for t in TYPES}
    report: list[dict] = []
    seen: dict[tuple[str, str], str] = {}
    updates: dict[str, dict[str, dict]] = {t: {} for t in TYPES}  # 유형 -> servId -> 새 값

    def add(file, r, status, action, detail=""):
        report.append({"file": file, "no": r.get("no", ""), "review_type": r.get("review_type", ""),
                       "servId": r.get("servId", ""), "status": status, "action": action, "detail": detail})

    for rv in REVIEWERS:
        path = team_dir / f"{rv}.csv"
        if not path.exists():
            continue
        for r in read_csv(path):
            t, sid = r.get("review_type", ""), r.get("servId", "")
            entry = ledger.get((t, sid))
            if entry:
                cols = INPUT_COLS[t]
                original = json.loads(entry["original"])
                effective = json.loads(entry["effective"])
                actual = {c: (r.get(c) or "").strip() for c in cols}
                if (entry["file"] != rv or entry["no"] != r.get("no") or actual != original
                        or set(effective) != set(cols)):
                    add(path.name, r, "오류", "충돌", "합의 로그와 원검수 불일치: 재검토 필요")
                    continue
                # 원본 팀 CSV는 보존하고 합의된 값만 비교한다.
                r = {**r, **effective}
                if rv == "R1" and r.get("reviewed_at") == "20261115":
                    r["reviewed_at"] = "2026-11-15"
            status, detail = check_row(r)
            if status == "오류":
                add(path.name, r, status, "오류", detail)
                continue
            key = (t, sid)
            if key in seen:
                add(path.name, r, status, "오류", f"중복 키(이미 {seen[key]}에 있음)")
                continue
            seen[key] = path.name
            if status == "미검수":
                add(path.name, r, status, "미검수")
                continue
            if status == "일부 입력":
                add(path.name, r, status, "일부 입력", detail)
                continue
            orig = sheets[t].get(sid)
            if orig is None:
                add(path.name, r, status, "오류", "원본 시트에 servId 없음")
                continue
            cols = INPUT_COLS[t]
            if all(_blank(orig.get(c)) for c in cols):
                new = {c: (r.get(c) or "").strip() for c in cols}
                new["reason"] = (r.get("reason") or "").strip()
                new["reviewer"] = (r.get("reviewer") or "").strip() or rv
                new["reviewed_at"] = (r.get("reviewed_at") or "").strip()
                updates[t][sid] = new
                add(path.name, r, status, "반영" if apply else "반영 예정")
            else:
                diff = [c for c in cols if (orig.get(c) or "").strip() != (r.get(c) or "").strip()]
                if diff:
                    add(path.name, r, status, "충돌", "원본과 다른 칸: " + ", ".join(diff))
                else:
                    add(path.name, r, status, "동일")

    if apply:
        for t in TYPES:
            if not updates[t]:
                continue
            path = review_dir / SHEET_FILES[t]
            rows = read_csv(path)
            for row in rows:
                new = updates[t].get(row["servId"])
                if new:
                    for c, v in new.items():
                        if _blank(row.get(c)):
                            row[c] = v
            write_csv(path, headers[t], rows)

    write_csv(team_dir / "merge_report.csv", REPORT_COLS, report)
    return report


# ---------- CLI ----------
def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("split", "merge"):
        s = sub.add_parser(name)
        s.add_argument("--review-dir", type=Path, default=REVIEW_DIR)
        s.add_argument("--team-dir", type=Path, default=TEAM_DIR)
        if name == "split":
            s.add_argument("--selection", type=Path, default=None)
            s.add_argument("--refresh", action="store_true")
        else:
            s.add_argument("--apply", action="store_true")
    a = p.parse_args(argv)

    if a.cmd == "split":
        sel = a.selection or a.team_dir / "selection.csv"
        res = split(sel, a.review_dir, a.team_dir, a.refresh)
        for rv in REVIEWERS:
            c = Counter(r["review_type"] for r in res["rows"][rv])
            msg, path = res["written"][rv]
            print(f"{rv}: " + ", ".join(f"{t} {c[t]}" for t in TYPES) + f" / {msg} {path.name}")
    else:
        report = merge(a.team_dir, a.review_dir, a.apply)
        c = Counter(r["action"] for r in report)
        print("dry-run" if not a.apply else "apply")
        print(", ".join(f"{k} {v}" for k, v in c.items()) or "읽은 행 없음")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
