"""방문 불가 사용자의 '대체 경로 근거 못 찾음' 확인 필요 서비스에 대한 접근 조건 구조화와 검수 시트 생성.

실행 (저장소 루트에서)
  python -m ai.eval.rules_access_extract            # 요약표 갱신, 검수 CSV는 없거나 사람 열이 비어 있을 때만 생성
  python -m ai.eval.rules_access_extract --refresh  # 검수 CSV에 사람 입력이 있으면 별도 access_condition_review_refresh.csv에 씀

출력: outputs/07_rules/
- review/access_condition_review.csv   대상 서비스 1행씩. 사람 열(apply_visit 등)은 비워 둠
- tables/access_condition_summary.csv  ai_* 열별 값 분포

주의
- LLM이 아닌 규칙(정규식) 기반 추출임. 값은 사람 검수용 제안일 뿐이며 services_master와 규칙에 되쓰지 않음.
- 신청(apply)과 이용(use)을 나눠 봄. 방문 신청만 안내된 경우는 필수로 보지 않고 불명으로 둠.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

AI = Path(__file__).resolve().parents[1]
ROOT = AI.parent
sys.path.insert(0, str(AI))

from rules.access import (CHECK, ONLINE_TOKENS, OTHER_REMOTE_TOKENS, check_access,  # noqa: E402
                          route_info, _val)

MASTER = ROOT / "data" / "processed" / "services_master.csv"
OUT = ROOT / "outputs" / "07_rules"
PROFILE = {"access": [{"polarity": "avoid", "channel": "방문", "level": "불가"}]}
TARGET_MARK = "대체 경로 근거 못 찾음"
SNIP = 150

APPLY_VISIT_VALUES = ("필수", "선택", "불필요", "불명")
USE_VISIT_VALUES = ("사용자 방문 필수", "담당자 방문", "선택", "불필요", "불명")
ALT_VALUES = ("있음", "없음 명시", "불명")
HUMAN_COLS = ["apply_visit", "apply_alt", "use_visit", "use_alt", "reason", "reviewer", "reviewed_at"]
AI_COLS = ["ai_apply_visit", "ai_apply_alt", "ai_apply_alt_evidence",
           "ai_use_visit", "ai_use_alt", "ai_use_alt_evidence"]
EVID_COLS = ["apply_evidence", "use_evidence"]
REVIEW_COLS = (["servId", "servNm", "apply_cat", "apply_outing_basis", "use_outing"] + AI_COLS + EVID_COLS
               + ["current_reason"] + HUMAN_COLS)

# ── 정규식 (잠정 사전, 사람 검수 전제) ─────────────────────────────
VISIT = r"(?:방문|내방|출석|내소)"
ONL = r"(?:온라인|인터넷|복지로|전화|우편|모바일|e-?mail|이메일)"
# 방문 필수를 말하는 강한 표현
APPLY_MUST_RE = re.compile(
    rf"반드시\s*(?:본인이?\s*)?(?:직접\s*)?{VISIT}|본인이?\s*직접\s*{VISIT}|직접\s*{VISIT}하여야|"
    rf"{VISIT}\s*필수|내방\s*필수")
# 대리·타 경로 불가 명시
NO_ALT_RE = re.compile(
    r"대리\s*(?:신청|접수|수령|이용|인)?\s*(?:은|는|도)?\s*(?:불가|안\s*됨|할\s*수\s*없)|"
    r"온라인\s*(?:신청\s*)?(?:은|는)?\s*불가|비대면\s*(?:신청\s*|이용\s*)?(?:은|는)?\s*불가|"
    rf"{VISIT}\s*(?:신청|접수|이용)?\s*(?:으로)?\s*만\s*가능|{VISIT}\s*외\s*(?:신청|접수|이용)?\s*불가")
# 대체 경로 있음 (부정 표현은 먼저 제거한 뒤 적용)
APPLY_ALT_RE = re.compile(
    r"대리\s*(?:신청|접수|인)|대리인|가족.{0,8}신청|직권|찾아가는|온라인|인터넷|복지로|정부\s*24|우편|팩스|"
    r"이메일|e-?mail|전화\s*(?:신청|접수)|비대면|모바일", re.I)
USE_ALT_RE = re.compile(
    r"비대면|온라인|화상|원격|재택|인터넷|택배|배송|우편|대리\s*(?:수령|이용)|전화\s*상담|이동\s*(?:식|형|버스)|출장|순회")
# 온라인 등으로 신청한 뒤 방문하는 순차 절차
SEQ_RE = re.compile(rf"{ONL}[^.;\n]{{0,25}}(?:신청|접수)[^.;\n]{{0,12}}(?:후|뒤|하고)[^.;\n]{{0,25}}{VISIT}", re.I)
SEP = r"(?:또는|혹은|,|/|·|이나|나)"
CHOICE_RE = re.compile(rf"{ONL}[^.;\n]{{0,12}}{SEP}\s*[^.;\n]{{0,12}}{VISIT}|{VISIT}[^.;\n]{{0,12}}{SEP}\s*[^.;\n]{{0,12}}{ONL}", re.I)
APPLY_VISIT_ANY_RE = re.compile(VISIT)
# 담당자(제공자)가 찾아오는 이용 방식
PROVIDER_RE = re.compile(
    r"찾아가는|찾아\s*(?:가|와|뵙|옵)|가정\s*방문|(?:자택|가정|집|거주지|현장)\s*(?:을|를)?\s*(?:직접\s*)?방문|"
    r"(?:집|자택|가정|거주지)\s*(?:으로|에)\s*(?:직접\s*)?(?:방문|찾아|와)|상담원[^.;\n]{0,10}방문|"
    r"방문\s*(?:상담|간호|진료|요양|목욕|돌봄|재활|교육|지도|검진|급식|서비스|케어)|출장\s*(?:상담|서비스)|"
    r"(?:담당자|직원|인력|요원|도우미|전문가|사회복지사|상담사)[^.;\n]{0,10}(?:가정|집|자택)[^.;\n]{0,6}방문")
USER_MUST_RE = re.compile(
    rf"센터[^.;\n]{{0,8}}{VISIT}|{VISIT}\s*필수|직접\s*{VISIT}|(?:기관|센터|시설|병원|의원|교육장|학교)[^.;\n]{{0,6}}(?:에\s*)?"
    rf"(?:방문|출석|참여)\s*(?:하여|해서|후|필수)|반드시\s*{VISIT}|정기\s*(?:적\s*)?(?:출석|방문)")
USE_CHOICE_RE = re.compile(r"\[선택|온라인\s*(?:또는|혹은|,|/)\s*(?:방문|대면)|(?:방문|대면)\s*(?:또는|혹은|,|/)\s*(?:온라인|비대면)")
REVIEW_MARKERS = ("[혼재", "[의료", "확인 필요")


def _snip(text: str, m: re.Match, width: int = SNIP) -> str:
    """매치 주변을 width자 이내로 잘라 근거 문구로 사용."""
    s, e = m.start(), m.end()
    half = max((width - (e - s)) // 2, 0)
    a, b = max(0, s - half), min(len(text), e + half)
    return re.sub(r"\s+", " ", text[a:b]).strip()[:width]


def _find(rx, sources):
    """(출처, 근거 문구) 첫 건. 없으면 None."""
    for name, text in sources:
        m = rx.search(text)
        if m:
            return name, _snip(text, m)
    return None


def _ev(hit) -> str:
    return f"{hit[0]}: {hit[1]}"[:SNIP] if hit else ""


def _sources(svc, cols):
    out = []
    for c in cols:
        t = str(_val(svc, c) or "")
        if t.strip():
            out.append((c, re.sub(r"\s+", " ", t)))
    return out


def _strip_neg(sources):
    return [(n, NO_ALT_RE.sub(" ", t)) for n, t in sources]


def extract_apply(svc) -> dict:
    """신청 범위: 방문 필요 정도와 대체 경로."""
    ri = route_info(svc)
    srcs = _sources(svc, ("apply_text", "apply_text_raw"))
    toks = ri["tokens"]
    has_visit_tok = "방문" in toks
    remote_toks = toks & (ONLINE_TOKENS | OTHER_REMOTE_TOKENS | {"전화"})
    text_all = " ".join(t for _, t in srcs)
    must = _find(APPLY_MUST_RE, srcs)
    noalt = _find(NO_ALT_RE, srcs)
    seq = _find(SEQ_RE, srcs)
    choice = _find(CHOICE_RE, srcs)
    alt_hit = _find(APPLY_ALT_RE, _strip_neg(srcs))
    basis = _val(svc, "apply_outing_basis")
    remote_flag = bool(remote_toks) or _val(svc, "online_bokjiro") == "Y" or basis == "remote_route"

    # 대체 경로
    if noalt:
        alt, alt_ev = "없음 명시", _ev(noalt)
    elif alt_hit:
        alt, alt_ev = "있음", _ev(alt_hit)
    elif remote_flag:
        alt = "있음"
        alt_ev = f"apply_cat: {','.join(sorted(toks))}" if remote_toks else f"online_bokjiro={_val(svc, 'online_bokjiro')}, basis={basis}"
    else:
        alt, alt_ev = "불명", ""

    # 방문 필요 정도
    if seq:
        visit, ev = "불필요", _ev(seq)       # 온라인 신청 뒤 방문은 이용·제출 단계로 보고 use에서 확인
    elif must:
        visit, ev = "필수", _ev(must)
    elif choice or (has_visit_tok and remote_toks):
        visit, ev = "선택", (_ev(choice) if choice else f"apply_cat: {','.join(sorted(toks))}")
    elif remote_flag and not has_visit_tok and not APPLY_VISIT_ANY_RE.search(text_all):
        visit, ev = "불필요", (f"apply_cat: {','.join(sorted(toks))}" if toks else f"basis={basis}")
    else:
        # 방문 신청만 안내된 경우 등은 방문 필수로 보지 않음(다른 경로 확인 필요)
        visit, ev = "불명", ""
        if alt != "없음 명시":
            alt, alt_ev = "불명", ""
    return {"ai_apply_visit": visit, "ai_apply_alt": alt, "ai_apply_alt_evidence": alt_ev, "apply_evidence": ev}


def extract_use(svc) -> dict:
    """이용 범위: 사용자 방문 필요 정도와 대체 경로."""
    srcs = _sources(svc, ("use_outing_evidence", "benefit_text", "summary_text", "target_text"))
    ev_text = str(_val(svc, "use_outing_evidence") or "")
    prov = _find(PROVIDER_RE, srcs)
    must = _find(USER_MUST_RE, srcs)
    choice = _find(USE_CHOICE_RE, srcs)
    noalt = _find(NO_ALT_RE, srcs)
    alt_hit = _find(USE_ALT_RE, _strip_neg(srcs))
    try:
        uo = int(float(_val(svc, "use_outing")))
    except (TypeError, ValueError):
        uo = None
    rc2 = str(_val(svc, "remote_complete")) == "2"

    if prov and (uo is None or uo <= 1 or not must):
        visit, ev = "담당자 방문", _ev(prov)
    elif choice or "[선택" in ev_text:
        visit, ev = "선택", (_ev(choice) if choice else f"use_outing_evidence: {ev_text}"[:SNIP])
    elif uo is not None and uo >= 2 and any(m in ev_text for m in REVIEW_MARKERS):
        visit, ev = "불명", f"use_outing_evidence: {ev_text}"[:SNIP]
    elif must or (uo is not None and uo >= 2 and ev_text.strip()):
        visit = "사용자 방문 필수"
        ev = _ev(must) if must else f"use_outing={uo}, use_outing_evidence: {ev_text}"[:SNIP]
    elif uo is not None and uo <= 1:
        visit, ev = "불필요", f"use_outing={uo}"
    else:
        visit, ev = "불명", ""

    if noalt:
        alt, alt_ev = "없음 명시", _ev(noalt)
    elif visit in ("담당자 방문", "선택"):
        alt, alt_ev = "있음", ev                      # 사용자가 방문하지 않는 방식 자체가 대체 경로
    elif alt_hit or rc2:
        alt, alt_ev = "있음", (_ev(alt_hit) if alt_hit else "remote_complete=2")
    else:
        alt, alt_ev = "불명", ""
    return {"ai_use_visit": visit, "ai_use_alt": alt, "ai_use_alt_evidence": alt_ev, "use_evidence": ev}


def extract_service(svc) -> dict:
    return {**extract_apply(svc), **extract_use(svc)}


def target_rows(df: pd.DataFrame) -> list[tuple[int, str]]:
    """방문 불가 + 확인 필요 + 대체 경로 근거 못 찾음 서비스의 (행 위치, 사유)."""
    out = []
    for i, rec in enumerate(df.to_dict("records")):
        st, why = check_access(rec, PROFILE)
        if st == CHECK and TARGET_MARK in why:
            out.append((i, why))
    return out


def _priority(r) -> int:
    if r["ai_use_visit"] == "담당자 방문" or "없음 명시" in (r["ai_apply_alt"], r["ai_use_alt"]):
        return 0
    if "불명" in (r["ai_apply_visit"], r["ai_use_visit"]):
        return 1
    return 2


def build_review(df: pd.DataFrame) -> pd.DataFrame:
    recs = df.to_dict("records")
    rows = []
    for i, why in target_rows(df):
        rec = recs[i]
        rows.append({"servId": rec.get("servId", ""), "servNm": rec.get("servNm", ""),
                     "apply_cat": rec.get("apply_cat", ""), "apply_outing_basis": rec.get("apply_outing_basis", ""),
                     "use_outing": rec.get("use_outing", ""), **extract_service(rec), "current_reason": why,
                     **{c: "" for c in HUMAN_COLS}})
    rev = pd.DataFrame(rows, columns=REVIEW_COLS)
    if len(rev):
        rev["_p"] = rev.apply(_priority, axis=1)
        rev = rev.sort_values(["_p", "servId"], kind="stable").drop(columns="_p")
    return rev.reset_index(drop=True)


def summarize(rev: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for col in ("ai_apply_visit", "ai_apply_alt", "ai_use_visit", "ai_use_alt"):
        for v, n in rev[col].value_counts().items():
            parts.append({"column": col, "value": v, "n_services": int(n),
                          "share": round(n / len(rev), 3) if len(rev) else 0})
    return pd.DataFrame(parts, columns=["column", "value", "n_services", "share"])


def has_human_input(path: Path) -> bool:
    """기존 검수 파일에 채워진 사람 열이 하나라도 있는지."""
    try:
        old = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    except Exception:
        return True  # 읽을 수 없으면 보존 쪽으로
    cols = [c for c in HUMAN_COLS if c in old.columns]
    return bool(cols) and bool(old[cols].apply(lambda s: s.str.strip() != "").any().any())


def write_review(rev: pd.DataFrame, path: Path, refresh: bool = False) -> Path | None:
    """사람 입력이 있는 기존 파일은 덮어쓰지 않음. 그 경우 --refresh일 때만 *_refresh.csv로 씀."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists() or not has_human_input(path):
        rev.to_csv(path, index=False, encoding="utf-8-sig")
        return path
    if refresh:
        alt = path.with_name(path.stem + "_refresh" + path.suffix)
        rev.to_csv(alt, index=False, encoding="utf-8-sig")
        return alt
    return None


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="방문 불가 대체 경로 미확인 서비스의 접근 조건 구조화")
    ap.add_argument("--refresh", action="store_true", help="사람 입력이 있는 검수 CSV는 건드리지 않고 *_refresh.csv에 씀")
    args = ap.parse_args(argv)
    df = pd.read_csv(MASTER, dtype=str, keep_default_na=False)
    rev = build_review(df)
    summ = summarize(rev)
    (OUT / "tables").mkdir(parents=True, exist_ok=True)
    summ.to_csv(OUT / "tables" / "access_condition_summary.csv", index=False, encoding="utf-8-sig")
    written = write_review(rev, OUT / "review" / "access_condition_review.csv", args.refresh)
    print("대상 서비스:", len(rev))
    print(summ.to_string(index=False))
    print("검수 CSV:", written if written else "사람 입력이 있어 기존 파일 유지(--refresh로 별도 파일 생성)")


if __name__ == "__main__":
    main()
