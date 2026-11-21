"""2단계 사용자 확인·수정 및 검색 쿼리 생성 검증 (2번 담당 범위).

검색 순위 평가, 서비스 임베딩 계산, RRF 평가, 관련성 후보 풀 생성은 하지 않는다(검색 담당·팀 통합 평가).

하위 명령 (저장소 루트에서, kiwipiepy 필요)
  python -m ai.eval.stage2_query_review sheets   # 사람 검수용 CSV 2종 생성 (이미 있으면 덮어쓰지 않음)
  python -m ai.eval.stage2_query_review check    # 검수 결과를 읽어 쿼리 생성 결과 비교표 생성

출력: outputs/06_stage2_query/
- review/q3_need_review.xlsx           Q3 일상어 질의 30개 확정 욕구 검수 (AI 제안값과 사람 확정값 분리)
                                       시트: 안내, 검수, 기술정보. git 제외(*.xlsx), 로컬 검수용
- review/query_terms_review.xlsx       쿼리 사전 v0.1 검수 (유지/수정/제외/보류)
                                       시트: 안내, 1_우선검수(자동 검토 표시 행), 2_일괄확인(나머지), 기술정보
- review/*_reviewed.csv                check 실행 시 내보내는 검수 결과 사본(공유·커밋용)
- tables/query_check_dev.csv           개발 말뭉치 label 기준 쿼리 생성 결과와 점검 항목
- tables/query_check_q3.csv            Q3 검수 완료 행만. 검수 완료 행이 없으면 만들지 않음
- tables/query_check_summary.csv       점검 항목 집계
- tables/input_validation.csv          입력 파일·행 수

자료 사용 원칙
- 사전 수정 근거는 개발 말뭉치(dev_corpus.jsonl)와 서비스마스터만. Q3는 평가용으로 결과만 보고함
- 사람 확정값(human_*)은 review_status가 '검수 완료'인 행만 사용. AI 제안값(ai_*)은 확정값으로 쓰지 않음
- 이 스크립트는 query_terms.yaml을 바꾸지 않음
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
import yaml

AI = Path(__file__).resolve().parents[1]
ROOT = AI.parent
for p in (str(AI),):
    if p not in sys.path:
        sys.path.insert(0, p)

from confirm.questions import auto_finish  # noqa: E402
from confirm.session import add_action, new_session  # noqa: E402
from extract.postprocess import rule_activity, rule_constraints  # noqa: E402
from extract.schema import empty_record  # noqa: E402
from query.build import ACCESS_MODES, CHANNEL_TOKENS, build_queries, load_terms  # noqa: E402

OUT = ROOT / "outputs" / "06_stage2_query"
PATHS = {
    "q3_pairs": ROOT / "data" / "raw" / "q3_query_pairs" / "q3_query_pairs_v2.csv",
    "q3_intent": ROOT / "outputs" / "05_q3" / "review" / "q3_intent_review_final.csv",
    "corpus": AI / "data" / "dev_corpus.jsonl",
    "services": ROOT / "data" / "processed" / "services_master.csv",
    "config": AI / "configs" / "config.yaml",
    "terms": AI / "configs" / "query_terms.yaml",
}
# 사람 검수 파일은 엑셀(판단 열 중심 + 기술정보 시트 분리). *.xlsx는 git 제외라 로컬에서만 검수하고,
# check 실행 시 검수 결과를 *_reviewed.csv로 내보내 공유한다.
NEED_BOOK = "q3_need_review.xlsx"
TERMS_BOOK = "query_terms_review.xlsx"
NEED_REVIEWED = "q3_need_review_reviewed.csv"
TERMS_REVIEWED = "query_terms_review_reviewed.csv"
NEED_SHEET = NEED_BOOK      # 이전 이름 호환
TERMS_SHEET = TERMS_BOOK
DECISIONS = ["유지", "수정", "제외", "보류"]
STATUSES = ["미검수", "검수 완료", "합의 필요"]
CSV = {"index": False, "encoding": "utf-8-sig"}
DOMAINS = ["주거", "건강", "일상·여가", "관계참여", "교육훈련", "미취업(구직)", "노동(일하는 중)", "미래설계", "경제"]
# AI 제안 main: Q3 query_group → 앱 영역 (2번 작성 대응표, 미검수). 사람 확정 전에는 쓰지 않음
GROUP_TO_DOMAIN = {"마음건강": "건강", "일상회복": "일상·여가", "관계": "관계참여", "여가·체육": "일상·여가",
                   "교육": "교육훈련", "취업": "미취업(구직)", "경제": "경제", "주거": "주거", "건강": "건강",
                   "공간": "일상·여가"}
ANALYSIS_SET = {"동일 의도": "주 분석", "일부 동일": "별도", "불일치": "제외"}
DONE = "검수 완료"
BODY_COLS = ["servNm", "summary_text", "benefit_text"]
ELDER = r"(?:노인|어르신|장애)"
# 사용자 지정 과도 확장 검토 대상
OVER_EXPANSION_REVIEW = {"난방비", "연료비"}


# ── 공통 ─────────────────────────────────────────────────────────
def read_csv(path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, encoding="utf-8-sig", keep_default_na=False)


def read_jsonl(path) -> list[dict]:
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12] if path.exists() else ""


def write_once(df: pd.DataFrame, path: Path, refresh: bool) -> str:
    """검수 시트는 사람이 채우므로 이미 있으면 덮어쓰지 않는다."""
    if path.exists() and not refresh:
        return "기존 유지"
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, **CSV)
    return "생성"


def split_domains(v: str) -> list[str]:
    return [d.strip() for d in re.split(r"[|,]", v or "") if d.strip()]


# ── 1) Q3 확정 욕구 검수 시트 ─────────────────────────────────────
def ai_need_suggestion(row, terms) -> tuple[str, str, str]:
    """AI 제안(규칙): main = query_group 대응표, sub = 일상어에 다른 영역의 핵심어·확장 패턴이 있으면 후보."""
    main = GROUP_TO_DOMAIN.get(row["query_group"], "")
    hits = []
    for d, v in (terms.get("domains") or {}).items():
        if d == main:
            continue
        words = list((v or {}).get("core") or [])
        words += [p for e in terms.get("expansions") or [] if e.get("domain") == d for p in e.get("patterns") or []]
        found = [w for w in words if w in row["daily_query"]]
        if found:
            hits.append((d, found))
    sub = "|".join(d for d, _ in hits)
    basis = f"main: query_group '{row['query_group']}' 대응표"
    if hits:
        basis += "; sub: " + ", ".join(f"{d}({'/'.join(f)})" for d, f in hits)
    return main, sub, basis


def need_sheet(terms) -> pd.DataFrame:
    """의도 검수 결과(지홍 1인 검수본)는 시트에 넣지 않는다. 같은 검수자가 의도 검수를 블라인드로 할 수 있게
    분석 구분(주 분석/별도/제외)은 check 단계에서 의도 검수 파일을 읽어 붙인다."""
    pairs = read_csv(PATHS["q3_pairs"])
    rows = []
    for r in pairs.to_dict("records"):
        main, sub, basis = ai_need_suggestion(r, terms)
        rows.append({
            "pair_id": r["pair_id"], "query_group": r["query_group"],
            "daily_query": r["daily_query"], "admin_query": r["admin_query"],
            "intent": r["intent"], "shared_elements": r["shared_elements"],
            "ai_main": main, "ai_sub": sub, "ai_basis": basis,
            "human_main": "", "human_sub": "", "human_reason": "",
            "reviewer": "", "reviewed_at": "", "review_status": "미검수",
        })
    return pd.DataFrame(rows)


# ── 2) 사전 검수 시트 ─────────────────────────────────────────────
def _snippets(texts: pd.Series, term: str, k=2) -> str:
    out = []
    for t in texts[texts.str.contains(re.escape(term))].head(k):
        i = t.find(term)
        out.append(t[max(0, i - 15): i + len(term) + 15].replace("\n", " "))
    return " / ".join(out)


def term_evidence(master: pd.DataFrame, body: pd.Series, apply_: pd.Series, term: str, domain: str | None) -> dict:
    hit = body.str.contains(re.escape(term))
    n = int(hit.sum())
    doms = Counter(d for v in master.loc[hit, "app_domain"] for d in str(v).split("|"))
    labeled = domain in DOMAINS and master["app_domain"].str.contains(re.escape(domain)).any()
    share = (doms.get(domain, 0) / n) if (n and labeled) else None  # 라벨 0건 영역(노동·미래설계)은 계산 불가
    elder = (int((hit & body.str.contains(ELDER)).sum()) / n) if n else None
    return {"master_count": n, "apply_text_count": int(apply_.str.contains(re.escape(term)).sum()),
            "domain_share": None if share is None else round(share, 3),
            "top_app_domains": ", ".join(f"{d} {c}" for d, c in doms.most_common(3)),
            "elder_disabled_share": None if elder is None else round(elder, 3),
            "evidence_servIds": "|".join(master.loc[hit, "servId"].head(3)),
            "evidence_snippets": _snippets(body, term)}


def auto_flags(ev: dict, term: str, tokens: list[str], domain: str | None, item_type: str) -> str:
    f = []
    if item_type == "candidate":
        f.append("보류·제외 후보")
    if term in OVER_EXPANSION_REVIEW:
        f.append("과도 확장 검토 대상")
    if ev["master_count"] >= 100:
        f.append("출현 100건 이상(범위 넓음)")
    if ev["domain_share"] is not None and ev["domain_share"] < 0.5 and item_type in ("core", "expansion"):
        f.append("해당 영역 비중 50% 미만")
    if domain in ("노동(일하는 중)", "미래설계") and item_type in ("core", "expansion"):
        f.append("영역 라벨 검증 불가")
    if ev["elder_disabled_share"] is not None and ev["elder_disabled_share"] >= 0.5:
        f.append("노인·장애 동시 출현 50% 이상")
    if len(tokens) > 1:
        f.append("Kiwi 분해: " + " ".join(tokens))
    if ev["master_count"] == 0:
        f.append("마스터 출현 0")
    return "; ".join(f)


def terms_sheet(terms, tokenize) -> pd.DataFrame:
    master = read_csv(PATHS["services"])
    body = master[BODY_COLS].agg(" ".join, axis=1)
    apply_ = master["apply_text"]
    rows = []

    def add(item_type, item_id, domain, patterns, term, condition, current_use, note=""):
        toks = tokenize(term)
        ev = term_evidence(master, body, apply_, term, domain)
        rows.append({"row_id": f"T{len(rows) + 1:03d}", "item_type": item_type, "item_id": item_id,
                     "domain": domain or "", "daily_patterns": "|".join(patterns or []), "admin_term": term,
                     "kiwi_tokens": " ".join(toks), "apply_condition": condition, **ev,
                     "current_use": current_use, "review_flags": auto_flags(ev, term, toks, domain, item_type),
                     "prior_note": note, "decision": "", "revised_term": "", "decision_reason": "",
                     "reviewer": "", "reviewed_at": ""})

    for d, v in (terms.get("domains") or {}).items():
        v = v or {}
        note = "; ".join(x for x in (v.get("source", ""), v.get("limitation", "")) if x)
        # v0.2: core가 비어 있거나 없는 영역은 core 행을 만들지 않음
        for i, w in enumerate(v.get("core") or []):
            add("core", f"core:{d}", d, [], w, f"영역 확정 시, 순서 {i + 1}(main 상한·sub 상한 적용)", "사용", note)
    for e in terms.get("expansions") or []:
        cond = f"{e.get('apply_when')}; block_if 뒤 12자: {'|'.join(e.get('block_if') or []) or '없음'}"
        if e.get("bm25_allowed", True) is False and e.get("dense_phrase"):
            # v0.2 Dense 전용 확장: BM25에는 넣지 않고 dense_phrase만 사용함
            ph = e["dense_phrase"]
            for w in ([ph] if isinstance(ph, str) else list(ph)):
                add("expansion", e["id"], e.get("domain"), e.get("patterns"), w, cond + "; BM25 미사용",
                    "사용(Dense 전용)", e.get("note", ""))
        else:
            for w in e.get("add_terms") or []:
                add("expansion", e["id"], e.get("domain"), e.get("patterns"), w, cond, "사용", e.get("note", ""))
        for c in e.get("candidates") or []:
            add("candidate", e["id"], e["domain"], e.get("patterns"), c["term"], cond, c.get("decision", "보류"),
                c.get("reason", ""))
    for a in terms.get("access_terms") or []:
        cond = (f"want 문장(want_markers) + 패턴; scope {a.get('scope')}; bm25_allowed {a.get('bm25_allowed')}; "
                f"main_path {a.get('main_path')}; block_if: {'|'.join(a.get('block_if') or [])}")
        for w in a.get("positive_terms", []):
            add("access", a["id"], None, a.get("patterns"), w, cond, "사용", a.get("dense_phrase", ""))
    for w in terms.get("banned_terms") or []:
        add("banned", "banned_terms", None, [], w, "어떤 경로로도 쿼리에 넣지 않음", "금지")
    for w in terms.get("hold_terms") or []:
        add("hold", "hold_terms", None, [], w, "사전에서 넣지 않음(원문 단어는 유지)", "보류")
    return pd.DataFrame(rows)


# ── 검수용 엑셀 ─────────────────────────────────────────────────
NEED_VIEW = [("pair_id", "질의 ID"), ("daily_query", "일상어 질의"), ("admin_query", "행정어 질의(참고)"),
             ("intent", "질의 의도(참고)"), ("ai_main", "AI 제안 main"), ("ai_sub", "AI 제안 sub"),
             ("ai_basis", "AI 제안 근거"), ("human_main", "확정 main"), ("human_sub", "확정 sub(| 구분)"),
             ("human_reason", "수정 사유"), ("reviewer", "검수자"), ("reviewed_at", "검수일(YYYY-MM-DD)"),
             ("review_status", "검수 상태")]
NEED_TECH = ["pair_id", "query_group", "shared_elements"]
TYPE_KO = {"core": "핵심어", "expansion": "확장어", "candidate": "보류·제외 후보", "access": "접근어",
           "banned": "금지어", "hold": "보류어"}
TERMS_VIEW = [("row_id", "행 ID"), ("group", "묶음"), ("item_type_ko", "유형"), ("domain", "영역"),
              ("daily_patterns", "일상어 표현"), ("admin_term", "행정어"), ("condition_short", "적용 조건"),
              ("review_flags", "검토 표시"), ("master_count", "서비스 출현 수"), ("domain_share", "해당 영역 비중"),
              ("top_app_domains", "주요 영역(건수)"), ("evidence_snippets", "서비스 근거 문맥"),
              ("current_use", "현재 상태"), ("prior_note", "기존 메모"), ("decision", "판정"),
              ("revised_term", "수정 행정어"), ("decision_reason", "판정 사유"), ("reviewer", "검수자"),
              ("reviewed_at", "검수일")]
TERMS_TECH = ["row_id", "item_type", "item_id", "kiwi_tokens", "apply_condition", "apply_text_count",
              "elder_disabled_share", "evidence_servIds"]
INPUT_COLS = {"human_main", "human_sub", "human_reason", "reviewer", "reviewed_at", "review_status",
              "decision", "revised_term", "decision_reason"}
CONDITION_SHORT = {"core": "영역 확정 시", "expansion": "영역 확정 + 일상어 표현 일치",
                   "candidate": "쿼리 미사용(후보)", "access": "원한다는 문장 + 일상어 표현 일치",
                   "banned": "쿼리 미사용", "hold": "사전에서 미사용"}


def _guide(rows: list[str]) -> pd.DataFrame:
    return pd.DataFrame({"안내": rows})


def need_book_frames(need: pd.DataFrame) -> dict[str, pd.DataFrame]:
    view = need[[c for c, _ in NEED_VIEW]].rename(columns=dict(NEED_VIEW))
    guide = _guide([
        "- 대상: Q3 일상어 질의 30개의 확정 욕구 영역",
        "- AI 제안(main·sub)은 참고값. 확정값으로 쓰지 않음",
        "- 확정 main: 목록에서 선택. 확정 sub: 영역명을 | 로 구분(없으면 빈칸)",
        "- AI 제안을 그대로 쓰면 같은 값을 확정 칸에 입력",
        "- 검수 상태가 '검수 완료'인 행만 검증에 사용. 빈칸·'미검수'·'합의 필요'는 사용 안 함",
        "- 영역 9개: " + ", ".join(DOMAINS),
        "- 기술정보 시트는 참고용. 수정하지 않음",
    ])
    return {"안내": guide, "검수": view, "기술정보": need[NEED_TECH]}


def terms_book_frames(terms_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    t = terms_df.copy()
    t["item_type_ko"] = t["item_type"].map(TYPE_KO)
    t["condition_short"] = t["item_type"].map(CONDITION_SHORT)
    t["group"] = t["item_type_ko"] + ":" + t["domain"].where(t["domain"] != "", t["item_id"])
    cols = [c for c, _ in TERMS_VIEW]
    flagged = t[t["review_flags"] != ""]
    rest = t[t["review_flags"] == ""].sort_values(["item_type", "domain", "row_id"])
    guide = _guide([
        f"- 1_우선검수: 자동 검토 표시가 붙은 {len(flagged)}행. 한 행씩 판정",
        f"- 2_일괄확인: 나머지 {len(rest)}행. 묶음별로 훑고 같은 판정은 채우기 핸들로 일괄 입력",
        "- 판정: 유지 / 수정 / 제외 / 보류. 수정이면 '수정 행정어'에 새 단어",
        "- 빈칸은 미검수로 처리. 자동 유지 아님",
        "- 판정 근거는 개발 말뭉치와 서비스마스터만 사용. Q3 질의는 근거로 쓰지 않음",
        "- 검토 표시: 과도 확장 검토 대상(난방비·연료비), 출현 100건 이상, 해당 영역 비중 50% 미만,",
        "  영역 라벨 검증 불가(노동·미래설계), 노인·장애 동시 출현 50% 이상, Kiwi 분해",
        "- 해당 영역 비중 = 그 단어가 나오는 서비스 중 해당 영역(app_domain) 서비스 비율",
        "- 기술정보 시트는 참고용. 수정하지 않음",
    ])
    return {"안내": guide,
            "1_우선검수": flagged[cols].rename(columns=dict(TERMS_VIEW)),
            "2_일괄확인": rest[cols].rename(columns=dict(TERMS_VIEW)),
            "기술정보": t[TERMS_TECH]}


def write_book(frames: dict[str, pd.DataFrame], path: Path, refresh: bool, domain_cols=(), sub_hint=()) -> str:
    """판단 열을 앞에 두고, 입력 칸에 색·드롭다운을 붙인다. 이미 있으면 덮어쓰지 않는다."""
    if path.exists() and not refresh:
        return "기존 유지"
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.worksheet.datavalidation import DataValidation

    path.parent.mkdir(parents=True, exist_ok=True)
    head = PatternFill("solid", fgColor="0F172A")
    inp = PatternFill("solid", fgColor="FEF9C3")
    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        for name, df in frames.items():
            df.to_excel(xw, sheet_name=name, index=False)
            ws = xw.sheets[name]
            ws.freeze_panes = "B2" if name != "안내" else None
            labels = list(df.columns)
            for j, col in enumerate(labels, start=1):
                c = ws.cell(row=1, column=j)
                c.font, c.fill = Font(bold=True, color="FFFFFF"), head
                c.alignment = Alignment(wrap_text=True, vertical="center")
                long = any(k in col for k in ("근거", "문맥", "질의", "의도", "메모", "표시", "사유", "안내", "표현"))
                ws.column_dimensions[c.column_letter].width = 48 if long else 14
                key = next((k for k, v in NEED_VIEW + TERMS_VIEW if v == col), col)
                for i in range(2, len(df) + 2):
                    cell = ws.cell(row=i, column=j)
                    cell.alignment = Alignment(wrap_text=long, vertical="top")
                    if key in INPUT_COLS:
                        cell.fill = inp
                n = len(df) + 1
                if n < 2:
                    continue
                rng = f"{c.column_letter}2:{c.column_letter}{n}"
                options = (DECISIONS if key == "decision" else STATUSES if key == "review_status"
                           else DOMAINS if key in domain_cols else None)
                if options:
                    dv = DataValidation(type="list", formula1='"' + ",".join(options) + '"', allow_blank=True)
                    ws.add_data_validation(dv)
                    dv.add(rng)
    return "생성"


def read_book(path: Path, sheets: list[str], view: list[tuple[str, str]]) -> pd.DataFrame:
    back = {v: k for k, v in view}
    parts = [pd.read_excel(path, sheet_name=s, dtype=str).fillna("").rename(columns=back) for s in sheets]
    return pd.concat(parts, ignore_index=True)


def read_need_review(review_dir: Path) -> pd.DataFrame | None:
    p = review_dir / NEED_BOOK
    return read_book(p, ["검수"], NEED_VIEW) if p.exists() else None


def read_terms_review(review_dir: Path) -> pd.DataFrame | None:
    p = review_dir / TERMS_BOOK
    if not p.exists():
        return None
    df = read_book(p, ["1_우선검수", "2_일괄확인"], TERMS_VIEW)
    tech = pd.read_excel(p, sheet_name="기술정보", dtype=str).fillna("")
    return df.merge(tech[["row_id", "item_type"]], on="row_id", how="left")


# ── 3) 쿼리 생성 비교 ─────────────────────────────────────────────
def session_from(text: str, needs: list[dict], activity=None, constraints=None, source="label") -> dict:
    rec = {**empty_record(), "needs": needs, "next_action": "추천" if needs else "되묻기"}
    if activity:
        rec["activity_range"], rec["activity_evidence"] = activity
    rec["constraints"] = constraints or []
    s = new_session(text, {"crisis": False}, rec)
    s["case_source"] = source
    return auto_finish(s)


def dev_cases(corpus) -> list[dict]:
    """개발 말뭉치 사람 label(영역 1개 이상)을 그대로 확정값으로 씀. status는 명시로 둠(label은 사람 판정)."""
    out = []
    for r in corpus:
        lab = r.get("label") or {}
        if not lab.get("needs"):
            continue
        needs = [{**n, "status": "명시"} for n in lab["needs"]]
        act = (lab["activity_range"], lab.get("activity_evidence")) if lab.get("activity_range") not in (
            None, "판단 불가") else None
        out.append({"dataset": "dev", "case_id": r["id"], "text": r["text"],
                    "session": session_from(r["text"], needs, act, lab.get("constraints"), "dev_label")})
    return out


def q3_cases(sheet: pd.DataFrame) -> tuple[list[dict], list[str]]:
    """검수 완료 행만. 활동범위·제약은 Q3에 사람 라벨이 없어 1단계 규칙(rule_activity·rule_constraints)으로만 채움.
    분석 구분은 의도 검수 파일(PATHS['q3_intent'])에서 붙임. 파일이 없으면 '확인 필요'."""
    out, issues = [], []
    intent = read_csv(PATHS["q3_intent"]) if PATHS["q3_intent"].exists() else pd.DataFrame(columns=["pair_id"])
    eq = dict(zip(intent.get("pair_id", []), intent.get("intent_equivalence", [])))
    for r in sheet.to_dict("records"):
        r["analysis_set"] = ANALYSIS_SET.get(eq.get(r["pair_id"], ""), "확인 필요")
        if r.get("review_status") != DONE:
            continue
        main = r.get("human_main", "").strip()
        subs = [d for d in split_domains(r.get("human_sub", "")) if d != main]
        bad = [d for d in [main] + subs if d not in DOMAINS]
        if not main or bad:
            issues.append(f"{r['pair_id']}: 영역 값 확인 필요 {bad or '(main 없음)'}")
            continue
        text = r["daily_query"]
        needs = [{"domain": main, "level": "main", "evidence": text, "status": "명시"}]
        needs += [{"domain": d, "level": "sub", "evidence": text, "status": "명시"} for d in subs]
        ar, ev = rule_activity(text)
        out.append({"dataset": "q3", "case_id": r["pair_id"], "text": text, "admin_query": r["admin_query"],
                    "analysis_set": r.get("analysis_set", ""),
                    "session": session_from(text, needs, (ar, ev) if ar != "판단 불가" else None,
                                            rule_constraints(text), "q3_human_need+stage1_rules")})
    return out, issues


def deletion_variants(case) -> list[tuple[str, dict, str]]:
    """영역이 2개 이상이면 하나씩 지운 모의 시나리오(삭제된 욕구 잔존 점검용)."""
    s = case["session"]
    needs = s["confirmed_profile"]["needs"]
    out = [("base", s, "")]
    if len(needs) >= 2:
        for n in needs:
            d = add_action(s, "delete_need", n["domain"])
            out.append((f"delete:{n['domain']}", auto_finish(d), n["domain"]))
    return out


REVIEW_DECISION_COLS = ["row_id", "item_type", "domain", "admin_term", "decision", "revised_term",
                        "decision_reason", "reviewer", "reviewed_at"]


def review_decisions_table(review_dir: Path) -> pd.DataFrame:
    """사람이 검수한 사전 엑셀을 읽어 판정 기록표로 정리함. 빈 판정은 '미검수'로 표시. 파일이 없으면 빈 표."""
    review = read_terms_review(Path(review_dir))
    if review is None:
        return pd.DataFrame(columns=REVIEW_DECISION_COLS)
    out = review.reindex(columns=REVIEW_DECISION_COLS).fillna("").astype(str)
    out["decision"] = out["decision"].str.strip().replace("", "미검수")
    return out.reset_index(drop=True)


def term_flags(review: pd.DataFrame | None) -> dict[str, str]:
    """사전 검수 시트의 판정·자동 표시. 판정이 있으면 제외·보류 판정만 표시(유지·수정은 검수 통과),
    판정이 없으면 자동 표시를 씀."""
    if review is None or review.empty:
        return {}
    out = {}
    for r in review.to_dict("records"):
        if r["item_type"] not in ("core", "expansion", "access"):
            continue
        if r["decision"]:
            flag = f"판정:{r['decision']}" if r["decision"] in ("제외", "보류") else ""
        else:
            flag = r["review_flags"]
        if flag:
            out[r["admin_term"]] = flag
    return out


def check_row(case, scenario, session, removed, mode, terms, cfg, tokenize, flags) -> dict:
    q = build_queries(session, terms, cfg, tokenize=tokenize, access_mode=mode)
    x = q["query_expansion"]
    search = q["bm25_search_tokens"]
    text_tok = tokenize(case["text"])
    removed_reason_tok = {t for r in x["removed_sentences"] for t in tokenize(r["sentence"])}
    intended_drop = set(terms.get("banned_terms") or []) | {
        t for f in x["bm25_interface"]["sentence_token_fallback"] for t in f["excluded"]}
    # 의도 누락: 원문 토큰 중 규칙상 뺀 문장·금지어가 아닌데 검색 토큰에 없는 것 (template 전환, 재토큰화 차이 등)
    missing = [t for t in dict.fromkeys(text_tok)
               if t not in search and t not in removed_reason_tok and t not in intended_drop]
    # 과도 확장: 이번 쿼리에 들어간 확장어 중 사전 검수 판정(제외·보류·수정) 또는 자동 표시가 있는 것
    over = [f"{w}({flags[w]})" for w in x["expansion_words"] if w in flags]
    residue = []
    if removed:
        words = list((terms["domains"].get(removed) or {}).get("core") or [])
        words += [w for e in terms.get("expansions") or [] if e.get("domain") == removed
                  for w in e.get("add_terms") or []]
        if removed not in {n["domain"] for n in session["confirmed_profile"]["needs"]}:
            bad_tok = {t for w in words for t in tokenize(w)}
            residue = sorted(set(search) & bad_tok) + [w for w in words if w in q["dense"]]
    avoid = [(a["channel"], a["level"]) for a in q["access"] if a["polarity"] == "avoid"]
    chan_tok = {t for ch, _ in avoid for t in CHANNEL_TOKENS.get(ch, ())}
    wants = [a["pattern_id"] for a in q["access"] if a["polarity"] == "want"]
    want_in_query = bool(x["access_tokens_requested"]) or x["dense_phrase_source"] == "requested"
    row = {
        "dataset": case["dataset"], "case_id": case["case_id"], "scenario": scenario, "access_mode": mode,
        "original_text": case["text"],
        "confirmed_needs": "|".join(f"{n['domain']}({n['level']})" for n in session["confirmed_profile"]["needs"]),
        "activity_range": session["confirmed_profile"]["activity_range"],
        "avoid_constraints": "|".join(f"{c}:{l}" for c, l in avoid),
        "core_words": "|".join(x["core_words"]), "expansion_words": "|".join(x["expansion_words"]),
        "expansions_applied": "|".join(e["id"] for e in x["expansions_applied"]),
        "query_source_text": x["query_source_text"],
        "bm25_query": q["bm25"], "bm25_search_tokens": " ".join(search), "dense_query": q["dense"],
        "bm25_mode": q["query_mode"]["bm25"], "dense_mode": q["query_mode"]["dense"],
        "mode_reasons": "|".join(q["query_mode"]["bm25_reasons"] + q["query_mode"]["dense_reasons"]),
        "access_requested": "|".join(wants), "access_inferred": "|".join(x["access_inferred_ids"]),
        # 점검 항목
        "chk_intent_missing_tokens": " ".join(missing),
        "chk_over_expansion": "; ".join(over),
        "chk_deleted_need_residue": " ".join(residue),
        "chk_avoid_channel_in_bm25": " ".join(sorted(set(search) & chan_tok)),
        "chk_want_detected": bool(wants),
        "chk_want_in_query": want_in_query if mode in ("B1", "B2") else None,
        "bm25_interface_match": x["bm25_interface"]["match"],
        "bm25_search_duplicates": " ".join(x["bm25_interface"]["search_duplicates"]),
    }
    if case["dataset"] == "q3":
        admin = list(dict.fromkeys(tokenize(case["admin_query"])))
        row["analysis_set"] = case["analysis_set"]
        row["admin_query"] = case["admin_query"]
        # 행정어 근접도: 행정어 질의 토큰 중 BM25 토큰에 있는 비율. 검색 성능 지표 아님
        row["admin_token_overlap"] = round(sum(t in search for t in admin) / len(admin), 3) if admin else None
        row["admin_tokens_missing"] = " ".join(t for t in admin if t not in search)
    # 빌더 v0.2 신규 키는 없을 수 있어 .get 기본값 사용
    fb = x.get("bm25_fallback") or {}
    fb_tokens = list(fb.get("tokens") or [])
    need_tok = (set(x.get("core_tokens") or []) | set(x.get("expansion_tokens") or []) | set(fb_tokens)
                | set(tokenize(x["query_source_text"] or "")))
    row["bm25_fallback_used"] = bool(fb.get("used"))
    row["bm25_fallback_tokens"] = "|".join(fb_tokens)
    row["dense_only_expansions"] = "|".join(e["id"] for e in x.get("dense_only_expansions") or [])
    # 최종 BM25 검색 토큰에 욕구 내용 토큰이 하나도 없으면(접근어만 있거나 빔) 표시함
    row["chk_bm25_empty"] = "" if need_tok & set(search) else "need_tokens_empty"
    return row


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (ds, mode), g in df.groupby(["dataset", "access_mode"]):
        base = g[g["scenario"] == "base"]
        dele = g[g["scenario"] != "base"]
        rows.append({
            "dataset": ds, "access_mode": mode, "cases": base["case_id"].nunique(),
            "bm25_template_cases": int((base["bm25_mode"] == "template").sum()),
            # 집계 규칙: cases_* 열은 base 시나리오만 셈(삭제 시나리오 제외). 신규 열이 없는 입력은 0
            "cases_bm25_fallback": (int(base["bm25_fallback_used"].fillna(False).astype(bool).sum())
                                    if "bm25_fallback_used" in base else 0),
            "cases_bm25_empty": (int((base["chk_bm25_empty"].fillna("") != "").sum())
                                 if "chk_bm25_empty" in base else 0),
            "cases_with_missing_text_tokens": int((base["chk_intent_missing_tokens"] != "").sum()),
            "cases_with_over_expansion_flag": int((base["chk_over_expansion"] != "").sum()),
            "deletion_scenarios": len(dele),
            "deletion_with_residue": int((dele["chk_deleted_need_residue"] != "").sum()),
            "cases_with_avoid_channel_token": int((g["chk_avoid_channel_in_bm25"] != "").sum()),
            "cases_want_detected": int(base["chk_want_detected"].sum()),
            "cases_want_in_query": int(base["chk_want_in_query"].fillna(False).astype(bool).sum()),
            "bm25_interface_mismatch": int((~base["bm25_interface_match"].astype(bool)).sum()),
            "mean_admin_token_overlap": (round(pd.to_numeric(base["admin_token_overlap"]).mean(), 3)
                                         if "admin_token_overlap" in base and ds == "q3" else None),
        })
    return pd.DataFrame(rows)


def run_check(out: Path, tokenize, modes=ACCESS_MODES, terms_path=None) -> dict:
    terms = load_terms(str(terms_path or PATHS["terms"]))
    cfg = yaml.safe_load(open(PATHS["config"], encoding="utf-8"))
    review_dir = out / "review"
    tr_path, nr_path = review_dir / TERMS_BOOK, review_dir / NEED_BOOK
    term_review = read_terms_review(review_dir)
    need_review = read_need_review(review_dir)
    flags = term_flags(term_review)
    corpus = read_jsonl(PATHS["corpus"])
    cases = dev_cases(corpus)
    q3_issues, q3_n = [], 0
    if need_review is not None:
        qc, q3_issues = q3_cases(need_review)
        q3_n = len(qc)
        cases += qc
        need_review.to_csv(review_dir / NEED_REVIEWED, **CSV)          # 공유용 검수 결과 사본
    if term_review is not None:
        review_decisions_table(review_dir).to_csv(review_dir / TERMS_REVIEWED, **CSV)   # 판정 기록(미검수 표시)
    rows = [check_row(c, sc, s, rm, m, terms, cfg, tokenize, flags)
            for c in cases for sc, s, rm in deletion_variants(c) for m in modes]
    df = pd.DataFrame(rows)
    tables = out / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    df[df["dataset"] == "dev"].dropna(axis=1, how="all").to_csv(tables / "query_check_dev.csv", **CSV)
    written = ["query_check_dev.csv"]
    if q3_n:
        df[df["dataset"] == "q3"].to_csv(tables / "query_check_q3.csv", **CSV)
        written.append("query_check_q3.csv")
    summarize(df).to_csv(tables / "query_check_summary.csv", **CSV)
    written.append("query_check_summary.csv")
    reviewed_terms = int((term_review["decision"] != "").sum()) if term_review is not None else 0
    iv = pd.DataFrame([
        {"input": k, "path": rel(v), "exists": v.exists(), "sha256_12": sha(v)} for k, v in PATHS.items()
    ] + [
        {"input": "dev_cases", "path": "", "exists": True, "sha256_12": "", "rows": sum(c["dataset"] == "dev" for c in cases)},
        {"input": "q3_reviewed_cases", "path": rel(nr_path), "exists": nr_path.exists(),
         "sha256_12": sha(nr_path), "rows": q3_n, "note": "; ".join(q3_issues)},
        {"input": "terms_reviewed_rows", "path": rel(tr_path), "exists": tr_path.exists(),
         "sha256_12": sha(tr_path), "rows": reviewed_terms},
    ])
    iv.to_csv(tables / "input_validation.csv", **CSV)
    written.append("input_validation.csv")
    return {"written": written, "q3_reviewed": q3_n, "q3_issues": q3_issues, "rows": len(df),
            "terms_reviewed": reviewed_terms}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["sheets", "check"])
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--refresh-sheets", action="store_true", help="검수 시트를 다시 만듦(사람 입력이 지워짐)")
    ap.add_argument("--terms", type=Path, default=None, help="비교할 사전 파일(기본 configs/query_terms.yaml)")
    args = ap.parse_args(argv)
    from retrieve.bm25 import tokenize  # kiwipiepy 필요

    if args.command == "sheets":
        terms = load_terms(str(args.terms or PATHS["terms"]))
        review = args.out / "review"
        st1 = write_book(need_book_frames(need_sheet(terms)), review / NEED_BOOK, args.refresh_sheets,
                         domain_cols=("human_main",))
        st2 = write_book(terms_book_frames(terms_sheet(terms, tokenize)), review / TERMS_BOOK, args.refresh_sheets)
        print(f"{NEED_BOOK}: {st1}\n{TERMS_BOOK}: {st2}")
        return {"need": st1, "terms": st2}
    res = run_check(args.out, tokenize, terms_path=args.terms)
    print(json.dumps(res, ensure_ascii=False, indent=2))
    return res


if __name__ == "__main__":
    main()
