"""Q3 실행: 기준선 검색 → 정답 없이 가능한 기술 집계 → 의도 검수 → 관련성 검수 → (사람 정답이 있으면) 지표.

주장 범위: SSIS 5,288건에 단순 키워드 검색 기준선을 적용했을 때의 검색 한계(q3_eval.CLAIM_SCOPE).
검수 순서(관문): ① 질의쌍 의도 검수 합의본(review/q3_intent_review_final.csv)이 있어야 관련성 시트를 만든다.
'불일치' 쌍은 관련성 검수·대응 비교에서 뺀다. ② 관련성 합의본(review/q3_relevance_final.csv)이 있어야
지표와 자격 시트를 만든다.
query_group은 질의를 묶기 위한 분석용 분류이며 앱 9개 욕구 영역·Q21 영역과 대응하지 않는다. 앱 라벨 성능
평가로 연결하려면 별도 대응표가 필요하다(이 코드 범위 밖).

    python -m src.recommend.q3_run

산출 (outputs/05_q3/)
- tables/q3_search_results.csv        질의 × 조건별 상위 50 (servId로 원문 추적)
- tables/q3_query_stats.csv           질의 × 조건별 토큰 수, 검색 건수, 0건 여부, 10위 동점 규모
- tables/q3_pair_overlap.csv          같은 쌍의 일상어·행정어 상위 10 겹침 (정답과 무관한 기술값)
- tables/q3_elig_auto_top10.csv       주 분석 상위 10의 자동 자격 사전 분류 (잠정: 지역=필드, 연령=규칙)
- tables/q3_metric_status.csv         정답 확보 상태와 지표 산출 여부
- tables/q3_error_codebook.csv        오류 유형 코드북
- tables/q3_query_checks.csv          일상어 질의의 개발용 말뭉치 유사도(0.70 기준)·행정어 공통 토큰
- review/q3_intent_review_sheet.csv   질의 의도 검수 양식 (검색 결과 비노출). 2인이 _R1·_R2로 저장
- review/q3_intent_codebook.csv       의도 검수 코드북 (동일 의도/일부 동일/불일치, 직접 비교/욕구 추론 필요)
- tables/q3_intent_agreement.csv      _R1·_R2가 모두 있으면 kappa·일치율·불일치 쌍
- tables/q3_relevance_agreement.csv   q3_relevance_R1·_R2가 모두 있으면 3단계·주 기준 kappa
- review/q3_relevance_sheet.csv       관련성 블라인드 검수 (의도 합의본 확정 후 생성, 순위·조건·점수 비노출)
- review/q3_eligibility_sheet.csv     자격 적합성 검수 (관련 판정 확정 후 생성, 자동 분류 비노출)
- review/q3_manual_additions.csv      풀 밖에서 사람이 찾은 관련 서비스 추가 양식
- review/q3_pool_key.csv              풀 출처(조건·순위). 검수자에게 보내지 않는다
- q3_run_meta.json                    입력 해시·규칙 버전·코드 커밋

검수 시트가 이미 있으면 덮어쓰지 않는다(--refresh-sheets로만 재생성).
사람 판정 합의본을 review/q3_relevance_final.csv로 두면 지표 표(q3_metrics_*.csv)를 추가로 만든다.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.recommend import q3_eval as ev
from src.recommend import q3_query_check as qc
from src.recommend import q3_search as qs
from src.recommend import q3_eligibility_review as er

ROOT = Path(__file__).resolve().parents[2]
MASTER = ROOT / "data" / "processed" / "services_master.csv"
QUERIES = ROOT / "data" / "raw" / "q3_query_pairs" / "q3_query_pairs_v2.csv"  # v1은 이력 보존
OUT = ROOT / "outputs" / "05_q3"
SAVE_TOP = 50
POOL_DEPTH = 10
# 풀은 주 분석 조건만으로 만든다(검수 부담). 민감도 조건은 지표와 함께 judged_coverage10을 보고한다.
POOL_CONDITIONS = qs.PRIMARY_CONDITIONS
QTYPES = ("daily", "admin")
TEXT_COLS = ["servNm", "summary_text", "target_text", "criteria_text", "benefit_text", "overview_text"]
CSV = dict(index=False, encoding="utf-8-sig")


def _group(r) -> str:
    """v2는 query_group, v1은 need_group(이름만 다르고 같은 분석용 분류)."""
    return r["query_group"] if "query_group" in r else r["need_group"]


def load_queries(path: Path = QUERIES) -> pd.DataFrame:
    q = pd.read_csv(path, dtype=str, encoding="utf-8-sig", keep_default_na=False)
    if q["pair_id"].duplicated().any():
        raise ValueError("pair_id 중복")
    bad = set(q["profile_id"]) - set(ev.PROFILES)
    if bad:
        raise ValueError(f"정의되지 않은 프로필: {bad}")
    long = []
    for _, r in q.iterrows():
        for t in QTYPES:
            long.append({"query_id": f"{r.pair_id}_{t[0].upper()}", "pair_id": r.pair_id, "query_type": t,
                         "query_text": r[f"{t}_query"], "intent": r.intent, "query_group": _group(r),
                         "profile_id": r.profile_id})
    return pd.DataFrame(long)


def run_search(index: qs.KeywordIndex, queries: pd.DataFrame, conditions) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    res, stats, full = [], [], {}
    for _, q in queries.iterrows():
        for c in conditions:
            ranked = index.search(q.query_text, c)
            full[(q.query_id, c.cond_id)] = ranked
            top = ranked.head(SAVE_TOP).copy()
            top.insert(0, "cond_id", c.cond_id)
            top.insert(0, "query_id", q.query_id)
            res.append(top)
            t = qs.cutoff_ties(ranked, ev.K)
            kept, dropped = index.query_terms(q.query_text, c)
            stats.append({"query_id": q.query_id, "pair_id": q.pair_id, "query_type": q.query_type,
                          "cond_id": c.cond_id, "scope": c.scope, "tokenizer": c.tokenizer, "mode": c.mode,
                          "df_max": c.df_max, "n_terms": len(kept), "terms": "|".join(kept),
                          "dropped_df_terms": "|".join(dropped), "zero_result": t["n_hits"] == 0, **t})
    return pd.concat(res, ignore_index=True), pd.DataFrame(stats), full


def top_ids(full: dict, query_id: str, cond_id: str, k: int = ev.K) -> list[str]:
    return full[(query_id, cond_id)]["servId"].head(k).tolist()


def pair_overlap(queries: pd.DataFrame, full: dict, conditions) -> pd.DataFrame:
    rows = []
    for pid in queries["pair_id"].unique():
        for c in conditions:
            d = set(top_ids(full, f"{pid}_D", c.cond_id))
            a = set(top_ids(full, f"{pid}_A", c.cond_id))
            u = d | a
            rows.append({"pair_id": pid, "cond_id": c.cond_id, "n_daily": len(d), "n_admin": len(a),
                         "n_common": len(d & a), "jaccard": (len(d & a) / len(u)) if u else np.nan})
    return pd.DataFrame(rows)


def elig_auto_top10(queries: pd.DataFrame, full: dict, master: pd.DataFrame) -> pd.DataFrame:
    m = master.set_index("servId")
    rows = []
    for _, q in queries.iterrows():
        p = ev.PROFILES[q.profile_id]
        for c in qs.PRIMARY_CONDITIONS:
            for rank, sid in enumerate(top_ids(full, q.query_id, c.cond_id), 1):
                rows.append({"query_id": q.query_id, "pair_id": q.pair_id, "query_type": q.query_type,
                             "cond_id": c.cond_id, "rank": rank, "servId": sid, "profile_id": p.profile_id,
                             **ev.eligibility_auto(m.loc[sid], p)})
    return pd.DataFrame(rows)


def build_pool(queries: pd.DataFrame, full: dict) -> pd.DataFrame:
    rows = []
    for _, q in queries.iterrows():
        for c in POOL_CONDITIONS:
            for rank, sid in enumerate(top_ids(full, q.query_id, c.cond_id, POOL_DEPTH), 1):
                rows.append({"pair_id": q.pair_id, "servId": sid, "source_query": q.query_id,
                             "cond_id": c.cond_id, "rank": rank})
    return pd.DataFrame(rows)


ROUND1_PER_GROUP = 1  # 1차(일치도 확인) 라운드: query_group별 첫 쌍
INTENT_SHEET_COLS = ["intent_equivalence(동일 의도/일부 동일/불일치)", "comparison_class(직접 비교/욕구 추론 필요)",
                     "daily_natural(예/아니오)", "admin_adds_condition(예/아니오)", "copies_service_name(예/아니오)",
                     "reason", "reviewer", "reviewed_at", "judge_type"]


def review_rounds(queries_wide: pd.DataFrame) -> dict[str, int]:
    """1차 라운드는 query_group마다 pair_id가 가장 작은 쌍. 나머지는 2차. 결과를 보지 않고 정하는 규칙이다."""
    g = "query_group" if "query_group" in queries_wide else "need_group"
    first = queries_wide.sort_values("pair_id").groupby(g).head(ROUND1_PER_GROUP)["pair_id"]
    return {pid: (1 if pid in set(first) else 2) for pid in queries_wide["pair_id"]}


def eligibility_sheet(gold: pd.DataFrame, queries_wide: pd.DataFrame, master: pd.DataFrame,
                      seed: int = ev.SEED) -> pd.DataFrame:
    """관련성 1 이상으로 판정된 서비스만 자격 검수한다. 자동 분류는 넣지 않는다."""
    m = master.set_index("servId")
    prof = queries_wide.set_index("pair_id")["profile_id"]
    rng = np.random.default_rng(seed + 1)
    rows = []
    for pid, g in gold[gold["relevance"] >= 1].groupby("pair_id", sort=True):
        sids = sorted(g["servId"])
        rng.shuffle(sids)
        p = ev.PROFILES[prof[pid]]
        for i, sid in enumerate(sids, 1):
            r = m.loc[sid]
            rows.append({"item_id": f"{pid}-E{i:03d}", "pair_id": pid, "profile_id": p.profile_id,
                         "profile_age": p.age, "profile_region": f"{p.sido} {p.sgg}", "servId": sid,
                         "servNm": r.get("servNm", ""), "source": r.get("source", ""),
                         "region_sido": r.get("region_sido", ""), "region_sgg": r.get("region_sgg", ""),
                         "target_text": r.get("target_text", ""), "criteria_text": r.get("criteria_text", ""),
                         "age_judgment(일치/명백불일치/판단불가)": "", "region_judgment(일치/명백불일치/판단불가)": "",
                         "income_household": "확인 필요", "evidence": "", "reviewer": "", "reviewed_at": "",
                         "judge_type": ""})
    return pd.DataFrame(rows)


def intent_sheet(queries_wide: pd.DataFrame) -> pd.DataFrame:
    show = ["pair_id", "query_group", "need_group", "intent", "shared_elements", "daily_query", "daily_style",
            "daily_length", "admin_query"]
    intent = queries_wide[[c for c in show if c in queries_wide.columns]].copy()
    for col in INTENT_SHEET_COLS:
        intent[col] = ""
    return intent


def relevance_sheet(queries_wide: pd.DataFrame, pool: pd.DataFrame, master: pd.DataFrame,
                    pairs: set[str], seed: int = ev.SEED) -> pd.DataFrame:
    """의도 검수에서 불일치가 아닌 쌍만. 의도 판정 결과·순위·조건·점수는 넣지 않는다."""
    m = master.set_index("servId")
    rounds = review_rounds(queries_wide)
    rng = np.random.default_rng(seed)
    meta = queries_wide.set_index("pair_id")
    rel_rows = []
    for pid, g in pool[pool["pair_id"].isin(pairs)].groupby("pair_id", sort=True):
        sids = sorted(g["servId"].unique())
        rng.shuffle(sids)
        for i, sid in enumerate(sids, 1):
            r = m.loc[sid]
            rel_rows.append({"item_id": f"{pid}-{i:03d}", "review_round": rounds[pid], "pair_id": pid,
                             "intent": meta.loc[pid, "intent"],
                             "shared_elements": meta.loc[pid, "shared_elements"], "servId": sid,
                             **{c: r.get(c, "") for c in TEXT_COLS},
                             "relevance(2직접/1부분/0무관)": "", "evidence": "", "reviewer": "", "reviewed_at": "",
                             "judge_type": "", "note": ""})
    return pd.DataFrame(rel_rows)


REL_COL = "relevance(2직접/1부분/0무관)"


def relevance_agreement(r1: pd.DataFrame, r2: pd.DataFrame) -> pd.DataFrame:
    """관련성 2인 판정 일치도. 3단계(0/1/2)와 주 분석 기준(2 vs 0·1)을 따로 낸다. 쌍(pair) 단위 묶음은 보지 않는
    항목 단위 값이므로 참고용이다. 판정이 둘 다 있는 항목만 쓴다."""
    m = r1.merge(r2, on="item_id", suffixes=("_r1", "_r2"))
    a, b = m[f"{REL_COL}_r1"].fillna("").str.strip(), m[f"{REL_COL}_r2"].fillna("").str.strip()
    ok = a.isin(["0", "1", "2"]) & b.isin(["0", "1", "2"])
    aa, bb = a[ok].tolist(), b[ok].tolist()
    rows = []
    for name, f in [("relevance_3level", lambda v: v), ("relevance_primary(2 vs 0·1)", lambda v: "2" if v == "2" else "01")]:
        xa, xb = [f(v) for v in aa], [f(v) for v in bb]
        rows.append({"item": name, "n_items": len(xa),
                     "agreement": (sum(x == y for x, y in zip(xa, xb)) / len(xa)) if xa else np.nan,
                     "kappa": ev.cohen_kappa(xa, xb) if xa else np.nan,
                     "n_disagree": sum(x != y for x, y in zip(xa, xb))})
    return pd.DataFrame(rows)


def intent_agreement(r1: pd.DataFrame, r2: pd.DataFrame) -> pd.DataFrame:
    """두 검수자 파일(시트와 같은 열)의 일치도. 둘 다 값이 있는 쌍만 쓴다."""
    cols = {"intent_equivalence": INTENT_SHEET_COLS[0], "comparison_class": INTENT_SHEET_COLS[1]}
    m = r1.merge(r2, on="pair_id", suffixes=("_r1", "_r2"))
    rows = []
    for name, col in cols.items():
        a, b = m[f"{col}_r1"].fillna("").str.strip(), m[f"{col}_r2"].fillna("").str.strip()
        ok = (a != "") & (b != "")
        aa, bb = a[ok].tolist(), b[ok].tolist()
        rows.append({"item": name, "n_pairs": len(aa),
                     "agreement": (sum(x == y for x, y in zip(aa, bb)) / len(aa)) if aa else np.nan,
                     "kappa": ev.cohen_kappa(aa, bb) if aa else np.nan,
                     "disagree_pairs": "|".join(m.loc[ok & (a != b), "pair_id"])})
    return pd.DataFrame(rows)


PAIR_SUBSETS = {  # 주 분석은 S1. 불일치 쌍은 어느 묶음에도 넣지 않는다.
    "S1_동일의도_직접비교": lambda d: (d["intent_equivalence"] == "동일 의도") & (d["comparison_class"] == "직접 비교"),
    "S2_동일의도_전체": lambda d: d["intent_equivalence"] == "동일 의도",
    "S3_동일의도_욕구추론": lambda d: (d["intent_equivalence"] == "동일 의도") & (d["comparison_class"] == "욕구 추론 필요"),
    "S4_일부동일": lambda d: d["intent_equivalence"] == "일부 동일",
}


def compute_metrics(queries: pd.DataFrame, full: dict, gold: pd.DataFrame, conditions, intent: pd.DataFrame,
                    min_rel: int = ev.RELEVANCE_PRIMARY_MIN) -> tuple[pd.DataFrame, pd.DataFrame]:
    """질의 × 조건 지표와 의도 묶음별 대응쌍 비교. gold·intent는 사람 합의본."""
    judged = {pid: set(g["servId"]) for pid, g in gold.groupby("pair_id")}
    rel = {pid: set(g.loc[g["relevance"] >= min_rel, "servId"]) for pid, g in gold.groupby("pair_id")}
    rows = []
    for _, q in queries.iterrows():
        R, J = rel.get(q.pair_id, set()), judged.get(q.pair_id, set())
        for c in conditions:
            ranked = full[(q.query_id, c.cond_id)]
            ids = ranked["servId"].head(ev.K).tolist()
            rows.append({"query_id": q.query_id, "pair_id": q.pair_id, "query_type": q.query_type,
                         "cond_id": c.cond_id, "n_relevant_pool": len(R),
                         "recall10_pool": ev.recall_at_k(ids, R), "success10": ev.success_at_k(ids, R),
                         "exp_recall10_ties": (ev.expected_hits_random_ties(ranked, R) / len(R)) if R else np.nan,
                         "judged_coverage10": (len(set(ids) & J) / len(ids)) if ids else np.nan})
    per = pd.DataFrame(rows).merge(intent[["pair_id", "intent_equivalence", "comparison_class"]], on="pair_id", how="left")
    per["recall_note"] = ev.RECALL_NOTE
    comps = []
    piv_all = per.pivot_table(index="pair_id", columns=["query_type", "cond_id"], values="recall10_pool", dropna=False)
    cls = intent.set_index("pair_id")
    for sname, rule in PAIR_SUBSETS.items():
        keep = set(cls.index[rule(cls)]) & set(piv_all.index)
        piv = piv_all.loc[sorted(keep)]
        for c in conditions:  # H1: 같은 필드, 일상어 → 행정어
            if ("daily", c.cond_id) in piv and ("admin", c.cond_id) in piv:
                comps.append({"subset": sname, **ev.paired_summary(piv[("daily", c.cond_id)].values,
                              piv[("admin", c.cond_id)].values, f"H1 {c.cond_id}: admin - daily")})
        for t in QTYPES:  # H2: 같은 표현, 요약 → 상세 포함
            for tok in ("t0", "t1"):
                a, b = (t, f"short_{tok}_or"), (t, f"full_{tok}_or")
                if a in piv and b in piv:
                    comps.append({"subset": sname, **ev.paired_summary(piv[a].values, piv[b].values,
                                  f"H2 {t} {tok}: full - short")})
    out = pd.DataFrame(comps)
    if len(out):
        out["recall_note"] = ev.RECALL_NOTE
        out["claim_scope"] = ev.CLAIM_SCOPE
    return per, out


def error_cases(queries: pd.DataFrame, full: dict, gold: pd.DataFrame, min_rel: int = ev.RELEVANCE_PRIMARY_MIN) -> pd.DataFrame:
    rows = []
    for pid, g in gold[gold["relevance"] >= min_rel].groupby("pair_id"):
        tops = {f"{t}_{s}": set(top_ids(full, f"{pid}_{t[0].upper()}", f"{s}_t0_or")) for t in QTYPES for s in ("short", "full")}
        for sid in g["servId"]:
            in_top = {k: sid in v for k, v in tops.items()}
            rows.append({"pair_id": pid, "servId": sid, **in_top, "flags": "|".join(ev.mechanical_error_flags(in_top)),
                         "primary_cause": "", "reviewer": ""})
    return pd.DataFrame(rows)


def _git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def main(master_path: Path = MASTER, query_path: Path = QUERIES, out: Path = OUT, refresh_sheets: bool = False) -> dict:
    master = qs.load_master(master_path)
    queries_wide = pd.read_csv(query_path, dtype=str, encoding="utf-8-sig", keep_default_na=False)
    queries = load_queries(query_path)
    index = qs.KeywordIndex(master)
    conditions = qs.PRIMARY_CONDITIONS + qs.SENSITIVITY_CONDITIONS
    results, stats, full = run_search(index, queries, conditions)

    tables, review = out / "tables", out / "review"
    tables.mkdir(parents=True, exist_ok=True)
    review.mkdir(parents=True, exist_ok=True)
    results.to_csv(tables / "q3_search_results.csv", **CSV)
    stats.to_csv(tables / "q3_query_stats.csv", **CSV)
    pair_overlap(queries, full, conditions).to_csv(tables / "q3_pair_overlap.csv", **CSV)
    elig = elig_auto_top10(queries, full, master)
    elig.to_csv(tables / "q3_elig_auto_top10.csv", **CSV)
    pd.DataFrame([{"code": k, "definition": v} for k, v in ev.ERROR_TYPES.items()]).to_csv(tables / "q3_error_codebook.csv", **CSV)
    corpus = qc.load_corpus() if qc.CORPUS.exists() else None
    checks = qc.query_checks(queries_wide, corpus)
    checks.to_csv(tables / "q3_query_checks.csv", **CSV)

    pool = build_pool(queries, full)
    pool.to_csv(review / "q3_pool_key.csv", **CSV)
    kept_sheets = []

    def write_sheet(name: str, df: pd.DataFrame, always_keep: bool = False) -> None:
        # 검수 시트는 사람이 채우는 파일이라 기본적으로 덮어쓰지 않는다. 추가 양식은 refresh여도 보존한다.
        path = review / f"{name}.csv"
        if path.exists() and (not refresh_sheets or always_keep):
            kept_sheets.append(name)
            return
        df.to_csv(path, **CSV)

    pd.DataFrame(ev.INTENT_CODEBOOK, columns=["item", "value", "definition"]).to_csv(review / "q3_intent_codebook.csv", **CSV)
    write_sheet("q3_intent_review_sheet", intent_sheet(queries_wide))
    write_sheet("q3_manual_additions", pd.DataFrame(columns=["pair_id", "servId", "how_found", "evidence", "reviewer",
                                                             "reviewed_at", "judge_type"]), always_keep=True)
    r1, r2 = review / "q3_intent_review_R1.csv", review / "q3_intent_review_R2.csv"
    if r1.exists() and r2.exists():
        rd = dict(dtype=str, encoding="utf-8-sig", keep_default_na=False)
        intent_agreement(pd.read_csv(r1, **rd), pd.read_csv(r2, **rd)).to_csv(tables / "q3_intent_agreement.csv", **CSV)

    rr1, rr2 = review / "q3_relevance_R1.csv", review / "q3_relevance_R2.csv"
    if rr1.exists() and rr2.exists():
        rd = dict(dtype=str, encoding="utf-8-sig", keep_default_na=False)
        relevance_agreement(pd.read_csv(rr1, **rd), pd.read_csv(rr2, **rd)).to_csv(tables / "q3_relevance_agreement.csv", **CSV)

    status = {"stage": "의도 검수 대기", "intent_final_exists": False, "n_intent_pairs": 0, "n_pairs_excluded_mismatch": 0,
              "relevance_final_exists": False, "n_human_judgments": 0, "metrics": "산출 보류(사람 정답 없음)",
              "claim_scope": ev.CLAIM_SCOPE, "recall_note": ev.RECALL_NOTE}
    ifinal = review / "q3_intent_review_final.csv"
    intent = None
    if ifinal.exists():
        intent = ev.load_human_intent(pd.read_csv(ifinal, dtype=str, encoding="utf-8-sig", keep_default_na=False))
        usable = set(intent.loc[intent["intent_equivalence"] != "불일치", "pair_id"])
        status.update(stage="관련성 검수 대기", intent_final_exists=True, n_intent_pairs=len(intent),
                      n_pairs_excluded_mismatch=int((intent["intent_equivalence"] == "불일치").sum()))
        write_sheet("q3_relevance_sheet", relevance_sheet(queries_wide, pool, master, usable))

    final = review / "q3_relevance_final.csv"
    efinal = review / "q3_eligibility_final.csv"
    status['eligibility_metrics'] = '산출 보류(사람 자격 합의본 없음)'
    if efinal.exists() and (not final.exists() or intent is None):
        raise ValueError('자격 집계에는 의도·관련성 합의본이 먼저 필요합니다')
    status["relevance_final_exists"] = final.exists()
    if final.exists() and intent is None:
        raise ValueError("관련성 합의본이 있지만 의도 검수 합의본(q3_intent_review_final.csv)이 없음. 의도 검수를 먼저 확정한다.")
    if final.exists():
        gold = ev.load_human_relevance(pd.read_csv(final, dtype=str, encoding="utf-8-sig", keep_default_na=False))
        status["n_human_judgments"] = len(gold)
        if len(gold):
            gold = gold[gold["pair_id"].isin(usable)]
            per, comp = compute_metrics(queries, full, gold, conditions, intent)
            per.to_csv(tables / "q3_metrics_by_query.csv", **CSV)
            comp.to_csv(tables / "q3_metrics_paired.csv", **CSV)
            error_cases(queries, full, gold).to_csv(tables / "q3_error_cases.csv", **CSV)
            esheet = review / "q3_eligibility_sheet.csv"
            if not esheet.exists():  # 사람이 채우기 시작한 시트는 덮어쓰지 않는다
                eligibility_sheet(gold, queries_wide, master).to_csv(esheet, **CSV)
            status.update(stage="지표 산출", metrics="산출(풀 기준 제한적 Recall@10)")
            if efinal.exists():
                elig = er.aggregate(queries, full, gold, intent,
                                    pd.read_csv(efinal, dtype=str, encoding='utf-8-sig', keep_default_na=False),
                                    qs.PRIMARY_CONDITIONS)
                elig.to_csv(tables / 'q3_eligibility_by_query.csv', **CSV)
                status['eligibility_metrics'] = ('산출' if len(elig) and elig.status.eq('산출').any()
                                                   else '산출 보류(사람 자격 검수 미완료 또는 분모 0)')
    pd.DataFrame([status]).to_csv(tables / "q3_metric_status.csv", **CSV)

    meta = {"created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "baseline_version": qs.BASELINE_VERSION, "code_commit": _git_commit(),
            "master_path": str(master_path.relative_to(ROOT)) if master_path.is_relative_to(ROOT) else str(master_path),
            "master_sha256": qs.file_sha256(master_path), "master_rows": len(master),
            "queries_path": str(query_path.relative_to(ROOT)) if query_path.is_relative_to(ROOT) else str(query_path),
            "queries_sha256": qs.file_sha256(query_path), "n_pairs": int(queries["pair_id"].nunique()),
            "conditions": [c.cond_id for c in conditions], "pool_conditions": [c.cond_id for c in POOL_CONDITIONS],
            "pool_depth": POOL_DEPTH, "k": ev.K, "seed": ev.SEED,
            "pool_size": int(pool[["pair_id", "servId"]].drop_duplicates().shape[0]),
            "df_max": qs.DF_MAX,
            "corpus_checked": corpus is not None,
            "corpus_overlap_n": int(checks["corpus_overlap"].sum()) if "corpus_overlap" in checks else None, "review_sheets_kept_existing": kept_sheets, **status}
    (out / "q3_run_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--master", type=Path, default=MASTER)
    ap.add_argument("--queries", type=Path, default=QUERIES)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--refresh-sheets", action="store_true",
                    help="기존 의도·관련성 검수 시트를 새로 만든다(질의셋 변경 후, 검수 시작 전에만)")
    a = ap.parse_args()
    print(json.dumps(main(a.master, a.queries, a.out, a.refresh_sheets), ensure_ascii=False, indent=2))
