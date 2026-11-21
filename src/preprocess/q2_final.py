"""Q2 최종 분석 1단계: 검수자 원본 CSV의 기록 오류를 고친 사본과 수정 로그를 만든다. 원본은 바꾸지 않는다.

입력  logs/validation/q2_match/human/review_R*.csv, logs/validation/q2_item_rules/human/item_review_R*.csv,
      review_assignment.csv, gov24_search_index.csv
출력  logs/validation/q2_final/fixed/*_fixed.csv (고친 셀이 있는 파일만), logs/validation/q2_final/fix_log.csv
규칙  1 gov_id 지수 표기 복원(세 조건 모두 통과 시), 2 h_alt_gov_id 지수 표기 복원(후보 정확히 1개일 때),
      3 h_basis·h_conflict 구분자를 '|'로 통일, 4 item R1 G-period-ssis-02 증거 칸 위치 교정,
      5 R3 item_no 98 검수자·일시 공란은 고치지 않고 기록, 6 그 밖의 검증 문제는 고치지 않고 기록
실행: python -m src.preprocess.q2_final
"""
from __future__ import annotations

import re
import shutil
import tempfile
from math import comb, sqrt
from pathlib import Path

import numpy as np
import pandas as pd

from ..collect._common import ROOT
from . import q2_item_review as IR
from . import q2_codebook as CB
from . import q2_items_v3 as V
from . import q2_review as MR

LOG_COLS = ["file", "row_key", "column", "original", "corrected", "basis", "status"]
EXP = re.compile(r"^(\d)(?:\.(\d+))?E\+(\d+)$")


def _read(p) -> pd.DataFrame:
    return pd.read_csv(p, dtype=str, encoding="utf-8-sig", keep_default_na=False)


def _norm(s: str) -> str:
    """지수 표기를 끝자리 0 없는 형태로 통일(엑셀은 3.40E+11 을 3.4E+11 로 쓰기도 함)."""
    m = EXP.match(s.strip())
    mant = f"{m.group(1)}.{m.group(2)}" if m.group(2) else m.group(1)
    if "." in mant:
        mant = mant.rstrip("0").rstrip(".")
    return f"{mant}E+{int(m.group(3)):02d}"


def exp_matches(original: str, num: str) -> bool:
    """원본 지수 표기가 12자리 id를 원본과 같은 유효숫자(최소 3자리)로 표기한 값과 같은지."""
    m = EXP.match(original.strip())
    if not m or not num.isdigit():
        return False
    d = max(len(m.group(2) or ""), 2)
    return _norm(f"{int(num):.{d}E}".replace("E+", "E+")) == _norm(original)


def _ws(s: str) -> str:
    return re.sub(r"\s+", "", s)


def _entry(file, key, col, orig, corr, basis, status) -> dict:
    return {"file": file, "row_key": key, "column": col, "original": orig, "corrected": corr, "basis": basis, "status": status}


def fix_reviewer_files(root=ROOT) -> pd.DataFrame:
    match_dir = root / "logs" / "validation" / "q2_match"
    item_dir = root / "logs" / "validation" / "q2_item_rules"
    out = root / "logs" / "validation" / "q2_final"
    asg = _read(match_dir / "review_assignment.csv").set_index("item_no")
    idx = _read(match_dir / "gov24_search_index.csv")
    name_of = dict(zip(idx["서비스ID"], idx["서비스명"]))
    log: list[dict] = []
    fixed: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}

    for r in MR.REVIEWERS:
        fname = f"review_{r}.csv"
        orig = _read(match_dir / "human" / fname)
        df = orig.copy()
        for i, row in orig.iterrows():
            key, g = row["item_no"], row["gov_id"]
            if "E" in g.upper():
                if key not in asg.index:
                    log.append(_entry(fname, key, "gov_id", g, "", "배정 파일에 item_no 없음", "unresolved"))
                else:
                    a = asg.loc[key]
                    c_a = row["servId"] == a["servId"]
                    c_b = exp_matches(g, a["gov_id"])
                    c_c = row["gov_name"] == name_of.get(a["gov_id"])
                    if c_a and c_b and c_c:
                        df.at[i, "gov_id"] = a["gov_id"]
                        log.append(_entry(fname, key, "gov_id", g, a["gov_id"],
                                          "지수 표기 복원: servId 일치, 배정 gov_id의 지수 표기가 원본과 일치, gov_name이 해당 id의 서비스명과 일치", "fixed"))
                    else:
                        log.append(_entry(fname, key, "gov_id", g, "",
                                          f"복원 조건 실패(servId={c_a}, 지수표기={c_b}, 서비스명={c_c})", "unresolved"))
            h = row["h_alt_gov_id"]
            if "E" in h.upper():
                q, n = _ws(row["h_alt_query"]), _ws(row["h_note"])
                cands, pool = [], 0
                for sid, nm in name_of.items():
                    if not exp_matches(h, sid):
                        continue
                    pool += 1
                    w = _ws(nm)
                    if w and ((q and (w in q or q in w)) or (n and w in n)):
                        cands.append(sid)
                if len(cands) == 1:
                    df.at[i, "h_alt_gov_id"] = cands[0]
                    log.append(_entry(fname, key, "h_alt_gov_id", h, cands[0],
                                      "지수 표기가 같은 id 중 h_alt_query·h_note와 서비스명이 부분 일치하는 후보 1개", "fixed"))
                else:
                    log.append(_entry(fname, key, "h_alt_gov_id", h, "", f"지수 표기 일치 {pool}개 중 서비스명 부분 일치 후보 {len(cands)}개: {'|'.join(cands)}", "unresolved"))
            for c in ("h_basis", "h_conflict"):
                v = row[c]
                if "," in v:
                    toks = [t.strip() for t in v.split(",")]
                    if all(t in MR.CODES for t in toks):
                        df.at[i, c] = "|".join(toks)
                        log.append(_entry(fname, key, c, v, "|".join(toks),
                                          "구분자 형식 통일(q2_review.validate는 '|' 구분, 모든 토큰이 CODES 허용값)", "fixed"))
                    else:
                        log.append(_entry(fname, key, c, v, "", "CODES에 없는 토큰 포함", "unresolved"))
            if r == "R3" and key == "98" and row["h_same"].strip() and not row["reviewer"].strip() and not row["reviewed_at"].strip():
                log.append(_entry(fname, key, "reviewer,reviewed_at", "", "", "h_same은 입력됐으나 reviewer, reviewed_at 공란. 채우지 않음", "note_unchanged"))
        fixed[fname] = (df, orig)

    for r in IR.REVIEWERS:
        fname = f"item_review_{r}.csv"
        orig = _read(item_dir / "human" / fname)
        df = orig.copy()
        for i, row in orig.iterrows():
            if r == "R1" and row["sample_id"] == "G-period-ssis-02":
                txt = row["h_meta"]
                if txt.strip() and txt not in IR.META_H and not row["h_evidence"].strip() and txt in row["designated_text"]:
                    df.at[i, "h_evidence"] = txt
                    df.at[i, "h_meta"] = ""
                    why = ("h_meta 허용값은 해당없음/판단불가뿐(q2_item_review.META_H). 텍스트는 designated_text([apply])의 원문 부분 문자열이고 "
                           "h_evidence가 비어 있어 증거가 한 칸 왼쪽에 입력된 것으로 봄")
                    log.append(_entry(fname, row["sample_id"], "h_meta", txt, "", why, "fixed"))
                    log.append(_entry(fname, row["sample_id"], "h_evidence", "", txt, why, "fixed"))
                else:
                    log.append(_entry(fname, row["sample_id"], "h_meta", txt, "", "전제 불성립(원문 부분 문자열 아님 등)", "unresolved"))
        fixed[fname] = (df, orig)

    # 규칙 6: 고친 사본에 남은 검증 문제는 고치지 않고 기록
    asg_i = _read(item_dir / "item_review_assignment.csv")
    for r in MR.REVIEWERS:
        for p in MR.validate(fixed[f"review_{r}.csv"][0], r).itertuples(index=False):
            log.append(_entry(f"review_{r}.csv", p.item_no, "(validate)", "", "", p.problem, "unresolved"))
        a = asg_i[asg_i["reviewer_primary"].eq(r) | asg_i["reviewer_second"].eq(r)]
        for p in IR.validate_rows(fixed[f"item_review_{r}.csv"][0], r, set(a["sample_id"])).itertuples(index=False):
            log.append(_entry(f"item_review_{r}.csv", p.sample_id, "(validate_rows)", "", "", p.problem, "unresolved"))

    log_df = pd.DataFrame(log, columns=LOG_COLS)
    (out / "fixed").mkdir(parents=True, exist_ok=True)
    for f in (out / "fixed").glob("*_fixed.csv"):
        f.unlink()
    for fname, (df, orig) in fixed.items():
        if not df.equals(orig):
            df.to_csv(out / "fixed" / fname.replace(".csv", "_fixed.csv"), index=False, encoding="utf-8-sig", lineterminator="\n")
    log_df.to_csv(out / "fix_log.csv", index=False, encoding="utf-8-sig", lineterminator="\n")
    return log_df


# ================================================================== 통계 함수
Z = 1.959964
NOTE = "unresolved n건은 분모에서 제외"
NOTE_RULE = ("규칙 라벨 전수 집계라 분모는 미해결과 무관. n_unresolved는 이 항목의 사람 금표준 미해결 단위 수(참고)이며 "
             "item_rule_accuracy의 분모에서 제외됨")
OUTDIR = Path("outputs") / "04_q2" / "tables" / "final"
CONFIRMED = ("단독", "일치", "조정")
LABEL_C = "규칙 기반 추정(v3.1), 매칭은 자동 후보 A·B. 타당성은 item_rule_accuracy, match_accuracy 참조"


def wilson(k: int, n: int, z: float = Z) -> tuple[float, float]:
    if n <= 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def mcnemar_exact(b: int, c: int) -> float:
    """양측 정확 McNemar: b+c 에 대한 이항(p=.5) 양측 p."""
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def newcombe_paired_ci(a: int, b: int, c: int, d: int, z: float = Z) -> tuple[float, float]:
    """대응 비율 차 p1-p2 의 Newcombe(1998) 방법 10 score 구간. p1=(a+b)/n, p2=(a+c)/n, a=둘 다 예, d=둘 다 아니오."""
    n = a + b + c + d
    if n == 0:
        return float("nan"), float("nan")
    p1, p2 = (a + b) / n, (a + c) / n
    (l1, u1), (l2, u2) = wilson(a + b, n, z), wilson(a + c, n, z)
    A = (a + b) * (c + d) * (a + c) * (b + d)
    B = a * d - b * c
    C = B - n / 2 if B > n / 2 else (0 if B >= 0 else B)
    phi = 0.0 if A == 0 else C / sqrt(A)
    dl = p1 - p2
    lo = dl - sqrt(max(0.0, (p1 - l1) ** 2 + (u2 - p2) ** 2 - 2 * phi * (p1 - l1) * (u2 - p2)))
    hi = dl + sqrt(max(0.0, (u1 - p1) ** 2 + (p2 - l2) ** 2 - 2 * phi * (u1 - p1) * (p2 - l2)))
    return max(-1.0, lo), min(1.0, hi)


def fleiss(table) -> float:
    """Fleiss kappa. table: 단위 x 범주 평점 수(단위마다 평점자 수 동일)."""
    t = np.asarray(table, dtype=float)
    n, m = t.sum(axis=1)[0], t.shape[0]
    pi = ((t * t).sum(axis=1) - n) / (n * (n - 1))
    pe = ((t.sum(axis=0) / (m * n)) ** 2).sum()
    return float("nan") if pe == 1 else float((pi.mean() - pe) / (1 - pe))


def krippendorff_alpha_nominal(units) -> float:
    """명목 Krippendorff alpha. units: 단위별 평점 리스트(길이 2 이상)."""
    cats = sorted({v for u in units for v in u})
    k = {c: i for i, c in enumerate(cats)}
    o = np.zeros((len(cats), len(cats)))
    for u in units:
        for i, x in enumerate(u):
            for j, y in enumerate(u):
                if i != j:
                    o[k[x], k[y]] += 1 / (len(u) - 1)
    n, nc = o.sum(), o.sum(axis=1)
    if n <= 1:
        return float("nan")
    de = (n * n - (nc * nc).sum()) / (n * (n - 1))
    return float("nan") if de == 0 else float(1 - (n - np.trace(o)) / n / de)


def holm(p) -> list:
    """Holm 보정 p. nan 은 제외하고 나머지 개수로 보정, nan 은 그대로."""
    p = np.asarray(p, dtype=float)
    ok = np.where(~np.isnan(p))[0]
    out = np.full(len(p), np.nan)
    run = 0.0
    for r, i in enumerate(sorted(ok, key=lambda i: p[i])):
        run = max(run, min(1.0, (len(ok) - r) * p[i]))
        out[i] = run
    return list(out)


# ================================================================== 2단계 집계(임시 사본)
def aggregate_effective(root=ROOT) -> dict:
    """고친 사본이 있으면 그것, 없으면 원본을 검수자 파일로 쓰는 임시 로그 사본에서 기존 집계 함수를 실행한다. 실제 로그 폴더에는 쓰지 않는다."""
    fx = root / "logs" / "validation" / "q2_final" / "fixed"
    with tempfile.TemporaryDirectory() as td:
        m, it = Path(td) / "q2_match", Path(td) / "q2_item_rules"
        shutil.copytree(root / "logs" / "validation" / "q2_match", m, ignore=shutil.ignore_patterns("ai"))
        shutil.copytree(root / "logs" / "validation" / "q2_item_rules", it)
        for r in MR.REVIEWERS:
            if (fx / f"review_{r}_fixed.csv").exists():
                shutil.copyfile(fx / f"review_{r}_fixed.csv", m / "human" / f"review_{r}.csv")
        for r in IR.REVIEWERS:
            if (fx / f"item_review_{r}_fixed.csv").exists():
                shutil.copyfile(fx / f"item_review_{r}_fixed.csv", it / "human" / f"item_review_{r}.csv")
        rm = MR.aggregate(valid=m, match_path=root / "data" / "processed" / "q2_match.csv")
        ri = IR.aggregate_items(rules_dir=it, write=True)
        jm, _ = MR.load_judgements(m)
        rows_i = pd.concat([_read(it / "human" / f"item_review_{r}.csv").assign(reviewer_file=r) for r in IR.REVIEWERS], ignore_index=True)
        return {"res_m": rm, "res_i": ri, "jd_m": jm, "rows_i": rows_i,
                "jd_i": {(x["reviewer_file"], x["sample_id"]): IR.row_label(x) for x in rows_i.to_dict("records")},
                "asg_m": _read(m / "review_assignment.csv"), "asg_i": _read(it / "item_review_assignment.csv"),
                "gold": _read(it / "item_gold_sample.csv"), "strata": _read(it / "gold_strata.csv")}


def adjudication_tables(agg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """판정 확정표(등급·점수·규칙 라벨·AI 판정 없음). 단독/일치/조정은 확정, 불일치·미판정은 미해결."""
    basis = {"단독": "단독 검수 1차 판정 채택(q2_review final_source 규칙)", "일치": "중복 검수 두 판정 일치",
             "조정": "중복 검수 불일치, 제3자 또는 합의 판정(final_method, final_by) 채택",
             "미해결": "중복 검수 불일치, 기존 규칙상 제3자 또는 합의 판정(final_method, final_by) 필요",
             "미판정": "판정 없음 또는 검증 오류로 판정 미인정"}
    cols = ["item_id", "field", "R1", "R2", "R3", "R4", "final", "final_source", "basis", "status"]

    def row(iid, field, a, b, lab_of, final, src):
        rv = {k: "" for k in MR.REVIEWERS}
        for k in (a, b):
            if k:
                rv[k] = lab_of(k, iid)
        return {"item_id": iid, "field": field, **rv, "final": final if src in CONFIRMED else "", "final_source": src,
                "basis": basis[src], "status": "confirmed" if src in CONFIRMED else "unresolved"}

    fm = agg["res_m"]["final"].set_index("item_no")
    jd = agg["jd_m"]
    m = pd.DataFrame([row(r.item_no, "h_same", r.reviewer_primary, r.reviewer_second, lambda k, i: jd.get((k, i), {}).get("h_same", ""),
                          fm.at[r.item_no, "final_same"], fm.at[r.item_no, "final_source"]) for r in agg["asg_m"].itertuples()], columns=cols)
    fi = agg["res_i"]["final"].set_index("sample_id")
    ji = agg["jd_i"]
    i = pd.DataFrame([row(r.sample_id, f"{fi.at[r.sample_id, 'item']}_{fi.at[r.sample_id, 'side']}", r.reviewer_primary, r.reviewer_second,
                          lambda k, s: ji.get((k, s), ""), fi.at[r.sample_id, "final_label"], fi.at[r.sample_id, "final_source"])
                      for r in agg["asg_i"].itertuples()], columns=cols)
    return m, i


def _join(parts) -> str:
    return "; ".join(f"{k}: {v.strip()}" for k, v in parts if v.strip())


def unresolved_lists(agg: dict, adj_m: pd.DataFrame, adj_i: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """사람 판정 요청 목록: 미해결 단위의 두 검수자 판정·근거 원문. 등급·규칙 라벨·AI 판정은 넣지 않는다."""
    jd, asg = agg["jd_m"], agg["asg_m"].set_index("item_no")
    rows = []
    for it in adj_m.loc[adj_m["status"].eq("unresolved"), "item_id"]:
        a = asg.loc[it]
        ka, kb = a["reviewer_primary"], a["reviewer_second"]
        ja, jb = jd.get((ka, it), {}), jd.get((kb, it), {})
        ev = lambda j: _join([("근거", j.get("h_basis", "")), ("충돌", j.get("h_conflict", "")), ("메모", j.get("h_note", ""))])
        ref = ja or jb
        rows.append({"item_id": it, "servId": a["servId"], "gov_id": a["gov_id"], "reviewer_a": ka, "label_a": ja.get("h_same", ""), "evidence_a": ev(ja),
                     "reviewer_b": kb, "label_b": jb.get("h_same", ""), "evidence_b": ev(jb),
                     "ssis_name": ref.get("ssis_name", ""), "gov_name": ref.get("gov_name", "")})
    um = pd.DataFrame(rows, columns=["item_id", "servId", "gov_id", "reviewer_a", "label_a", "evidence_a", "reviewer_b", "label_b", "evidence_b", "ssis_name", "gov_name"])
    ri = agg["rows_i"].set_index(["reviewer_file", "sample_id"])
    asg, gold = agg["asg_i"].set_index("sample_id"), agg["gold"].set_index("sample_id")
    rows = []
    for it in adj_i.loc[adj_i["status"].eq("unresolved"), "item_id"]:
        a, g = asg.loc[it], gold.loc[it]
        ka, kb = a["reviewer_primary"], a["reviewer_second"]
        x, y = ri.loc[(ka, it)], ri.loc[(kb, it)]
        rows.append({"item_id": it, "servId": g["servId"], "gov_id": g["gov_id"], "item": g["item"], "side": g["side"],
                     "reviewer_a": ka, "label_a": IR.row_label(x), "evidence_a": _join([("증거", x["h_evidence"]), ("메모", x["h_note"])]),
                     "reviewer_b": kb, "label_b": IR.row_label(y), "evidence_b": _join([("증거", y["h_evidence"]), ("메모", y["h_note"])]),
                     "designated_text": x["designated_text"][:300], "codebook_definition": x["codebook_definition"]})
    return um, pd.DataFrame(rows, columns=["item_id", "servId", "gov_id", "item", "side", "reviewer_a", "label_a", "evidence_a",
                                           "reviewer_b", "label_b", "evidence_b", "designated_text", "codebook_definition"])


# ================================================================== 3단계 통계표
def _ci(df: pd.DataFrame, k: str, n: str, name: str) -> pd.DataFrame:
    ci = [wilson(int(a), int(b)) for a, b in zip(df[k], df[n])]
    df[f"{name}_ci_low"], df[f"{name}_ci_high"] = [c[0] for c in ci], [c[1] for c in ci]
    return df


def match_accuracy(agg: dict, root=ROOT) -> pd.DataFrame:
    tiers = _read(root / "data" / "processed" / "q2_match.csv")
    t = agg["asg_m"].merge(agg["res_m"]["final"][["item_no", "final_same", "final_source"]], on="item_no")
    t["tier"] = t["servId"].map(tiers.drop_duplicates("servId").set_index("servId")["tier"])
    rows = []
    for tier in ["A", "B", "C", "U"]:
        g = t[t["tier"].eq(tier)]
        conf = g[g["final_source"].isin(CONFIRMED)]
        ns, nd, nu = (int(conf["final_same"].eq(k).sum()) for k in MR.SAME)
        nc, nun = len(conf), len(g) - len(conf)
        rows.append({"tier": tier, "n_sampled": len(g), "n_confirmed": nc, "n_same": ns, "n_diff": nd, "n_undecidable": nu, "n_unresolved": nun,
                     "acc": ns / nc if nc else np.nan, "n_excl_undecidable": ns + nd, "acc_excl_undecidable": ns / (ns + nd) if ns + nd else np.nan,
                     "acc_low_bound": ns / (nc + nun) if nc + nun else np.nan, "acc_high_bound": (ns + nun) / (nc + nun) if nc + nun else np.nan,
                     "method": "Wilson 95%"})
    df = _ci(pd.DataFrame(rows), "n_same", "n_confirmed", "acc")
    ex = [wilson(int(a), int(b)) for a, b in zip(df["n_same"], df["n_excl_undecidable"])]
    df["acc_excl_ci_low"], df["acc_excl_ci_high"] = [e[0] for e in ex], [e[1] for e in ex]
    ab = df[df["tier"].isin(["A", "B"])].set_index("tier")
    N = {k: int(((tiers["tier"] == k) & tiers["gov_id"].ne("")).sum()) for k in "AB"}
    W = {k: N[k] / (N["A"] + N["B"]) for k in "AB"}
    p = {k: ab.at[k, "acc"] for k in "AB"}
    est = sum(W[k] * p[k] for k in "AB")
    se = sqrt(sum(W[k] ** 2 * p[k] * (1 - p[k]) / ab.at[k, "n_confirmed"] for k in "AB"))
    df.loc[len(df)] = {"tier": "AB_weighted", "n_sampled": int(ab["n_sampled"].sum()), "n_confirmed": int(ab["n_confirmed"].sum()),
                       "n_same": int(ab["n_same"].sum()), "n_diff": int(ab["n_diff"].sum()), "n_undecidable": int(ab["n_undecidable"].sum()),
                       "n_unresolved": int(ab["n_unresolved"].sum()), "acc": est, "acc_ci_low": max(0.0, est - Z * se), "acc_ci_high": min(1.0, est + Z * se),
                       "method": f"층화 가중(정규근사), W_A={W['A']:.4f} (N_A={N['A']}), W_B={W['B']:.4f} (N_B={N['B']}), p_h=n_same/n_confirmed"}
    df["note"] = NOTE + ". 자동 매칭 결과는 정답으로 쓰지 않음. C, U 등급은 합쳐 전체 정확도를 내지 않음"
    return df


def item_rule_accuracy(agg: dict) -> pd.DataFrame:
    fin = agg["res_i"]["final"].merge(agg["gold"][["sample_id", "stratum", "v31_level", "v31_meta"]], on="sample_id")
    fin["v31"] = fin["v31_meta"].where(fin["v31_meta"].ne(""), fin["v31_level"])
    fin["agree"] = fin["final_label"].ne("") & fin["final_label"].eq(fin["v31"])
    st = agg["strata"].copy()
    st[["N_h", "n_h"]] = st[["N_h", "n_h"]].astype(int)
    sa = IR.weighted_accuracy(st, fin)       # 확정 단위(final_label 있음)만 사용
    rows = []
    for item in V.ITEMS:
        for side in list(IR.SIDES) + ["all"]:
            f = fin[fin["item"].eq(item) & (fin["side"].eq(side) | (side == "all"))]
            x = sa[sa["item"].eq(item) & (sa["side"].eq(side) | (side == "all"))]
            conf = f[f["final_label"].ne("")]
            has = x[x["n_final"].gt(0)]
            rows.append({"item": item, "side": side, "core_metric": item in CB.CORE_ITEMS, "n_units": len(f), "n_confirmed": len(conf),
                         "n_unresolved": len(f) - len(conf), "n_correct": int(conf["agree"].sum()),
                         "acc": conf["agree"].sum() / len(conf) if len(conf) else np.nan,
                         "weighted_acc": float((has["N_h"] * has["accuracy"]).sum() / has["N_h"].sum()) if len(has) else np.nan,
                         "n_strata": len(x), "n_strata_without_confirmed": len(x) - len(has),
                         "note": NOTE + ". 가중 정확도는 층 N_h 가중(확정 단위가 없는 층은 제외하고 재정규화)"})
    return _ci(pd.DataFrame(rows), "n_correct", "n_confirmed", "acc")


def load_items(root=ROOT) -> pd.DataFrame:
    """노트북 PART C 의 AB_nonshared: tier A/B 이고 gov_shared == 'False'."""
    d = _read(root / "data" / "processed" / "q2_items_v3_1.csv")
    d["lab"] = V.label(d)
    return d[d["tier"].isin(["A", "B"]) & d["gov_shared"].eq("False")]


def _paired_table(s: pd.DataFrame, unres: dict) -> pd.DataFrame:
    """플랫폼 쌍 비교표(explicitness_tables 와 보충 분석 S2 가 공유). s: load_items 형식 단위 표."""
    lv = IR.LEVELS
    p = s.pivot_table(index=["servId", "gov_id", "scope", "item"], columns="side", values="lab", aggfunc="first").reset_index()
    p = p[p["ssis"].isin(lv) & p["gov"].isin(lv)]
    rows = []
    for scope in ["text", "all"]:
        for item in V.ITEMS:
            g = p[p["scope"].eq(scope) & p["item"].eq(item)]
            ys, yg = g["ssis"].eq("명시"), g["gov"].eq("명시")
            a, b, cc, d = int((ys & yg).sum()), int((~ys & yg).sum()), int((ys & ~yg).sum()), int((~ys & ~yg).sum())
            n = len(g)
            lo, hi = newcombe_paired_ci(a, b, cc, d)
            rows.append({"scope": scope, "item": item, "core_metric": item in CB.CORE_ITEMS, "n_pairs": n, "n_both_explicit": a,
                         "b_gov_only": b, "c_ssis_only": cc, "n_both_not": d,
                         "p_ssis": (a + cc) / n if n else np.nan, "p_gov": (a + b) / n if n else np.nan, "diff": (b - cc) / n if n else np.nan,
                         "diff_ci_low": lo, "diff_ci_high": hi, "mcnemar_p": mcnemar_exact(b, cc), "n_unresolved": unres.get(item, 0), "note": NOTE_RULE})
    t = pd.DataFrame(rows)
    core = t["core_metric"]
    t["holm_p"] = np.nan
    t.loc[core, "holm_p"] = holm(t.loc[core, "mcnemar_p"])
    t["holm_family"] = np.where(core, "핵심 6항목 x 2범위 = 12검정", "")
    t["test_note"] = "두 플랫폼 비교라 정확 McNemar 사용(Cochran Q는 3개 이상 대응 비교용이라 해당 없음). 차이 = 보조금24 - SSIS. Newcombe(1998) 방법 10 구간. " + LABEL_C
    return t


def explicitness_tables(root=ROOT, unres: dict | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    unres = unres or {}
    s, lv = load_items(root), IR.LEVELS
    rows = []
    for scope in ["text", "all"]:
        for item in V.ITEMS:
            for side in IR.SIDES:
                g = s[s["scope"].eq(scope) & s["item"].eq(item) & s["side"].eq(side)]
                den, k = int(g["lab"].isin(lv).sum()), int(g["lab"].eq("명시").sum())
                rows.append({"scope": scope, "item": item, "side": side, "core_metric": item in CB.CORE_ITEMS, "n_units": len(g),
                             "n_explicit": k, "n_denominator": den, "share_explicit": k / den if den else np.nan,
                             "n_unresolved": unres.get(item, 0), "note": NOTE_RULE, "label": LABEL_C})
    c = _ci(pd.DataFrame(rows), "n_explicit", "n_denominator", "share")
    ref = pd.read_csv(root / "outputs" / "04_q2" / "tables" / "item_explicitness_distribution.csv", encoding="utf-8-sig")
    m = c.merge(ref[ref["pair_set"].eq("AB_nonshared")], on=["scope", "item", "side"])
    assert len(m) == len(c) == 28 and (m["n_explicit"] == m["n_명시"]).all() and (m["n_denominator"] == m["n_level_denominator"]).all(), "explicitness mismatch"
    t = _paired_table(s, unres)
    # text 대 all
    q = s.pivot_table(index=["servId", "gov_id", "item", "side"], columns="scope", values="lab", aggfunc="first").reset_index()
    q = q[q["text"].isin(lv) & q["all"].isin(lv)]
    rows = []
    for item in V.ITEMS:
        for side in IR.SIDES:
            g = q[q["item"].eq(item) & q["side"].eq(side)]
            n = len(g)
            ty, ay = g["text"].eq("명시"), g["all"].eq("명시")
            add, lost = int((~ty & ay).sum()), int((ty & ~ay).sum())
            rows.append({"item": item, "side": side, "core_metric": item in CB.CORE_ITEMS, "n": n, "k_text": int(ty.sum()), "k_all": int(ay.sum()),
                         "share_text": ty.mean() if n else np.nan, "share_all": ay.mean() if n else np.nan,
                         "k_added_by_all": add, "added_by_all": add / n if n else np.nan, "k_lost_in_all": lost, "lost_in_all": lost / n if n else np.nan,
                         "mcnemar_p": mcnemar_exact(add, lost), "n_unresolved": unres.get(item, 0), "note": NOTE_RULE})
    d = pd.DataFrame(rows)
    for k, nm in (("k_text", "text"), ("k_all", "all"), ("k_added_by_all", "added"), ("k_lost_in_all", "lost")):
        d = _ci(d, k, "n", nm)
    core = d["core_metric"]
    d["holm_p"] = np.nan
    d.loc[core, "holm_p"] = holm(d.loc[core, "mcnemar_p"])
    d["holm_family"] = np.where(core, "핵심 6항목 x 2측 = 12검정", "")
    d["test_note"] = np.where(d["k_added_by_all"] + d["k_lost_in_all"] == 0, "두 범위 판정 동일", "")
    d["label"] = LABEL_C
    return c, t, d


def _boot_kappa(units: list, n_boot: int = MR.BOOT_N, seed: int = MR.BOOT_SEED) -> tuple[float, float]:
    cats = sorted({v for u in units for v in u})
    t = np.array([[u.count(c) for c in cats] for u in units])
    if len(t) < MR.MIN_CI_N:
        return float("nan"), float("nan")
    idx = np.random.default_rng(seed).integers(0, len(t), size=(n_boot, len(t)))
    v = np.array([fleiss(t[i]) for i in idx], dtype=float)
    v = v[~np.isnan(v)]
    return tuple(float(x) for x in np.percentile(v, [2.5, 97.5])) if len(v) else (float("nan"), float("nan"))


def agreement_tables(agg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    jm, ji = agg["jd_m"], agg["jd_i"]
    specs = [("match", [(r.reviewer_primary, r.reviewer_second, jm.get((r.reviewer_primary, r.item_no), {}).get("h_same", ""),
                         jm.get((r.reviewer_second, r.item_no), {}).get("h_same", "")) for r in agg["asg_m"].itertuples() if r.reviewer_second]),
             ("item", [(r.reviewer_primary, r.reviewer_second, ji.get((r.reviewer_primary, r.sample_id), ""), ji.get((r.reviewer_second, r.sample_id), ""))
                       for r in agg["asg_i"].itertuples() if r.reviewer_second])]
    rows, pairs = [], []
    for name, dbl in specs:
        both = [x for x in dbl if x[2] and x[3]]
        u = [[x[2], x[3]] for x in both]
        ag, n = sum(x[2] == x[3] for x in both), len(both)
        cats = sorted({v for x in u for v in x})
        lo, hi = wilson(ag, n)
        klo, khi = _boot_kappa(u) if u else (float("nan"),) * 2
        rows.append({"table": name, "n_double": len(dbl), "n_both": n, "n_agree": ag, "n_disagree": n - ag, "pct_agree": ag / n if n else np.nan,
                     "pct_ci_low": lo, "pct_ci_high": hi, "fleiss_kappa": fleiss([[x.count(c) for c in cats] for x in u]) if u else np.nan,
                     "kappa_ci_low": klo, "kappa_ci_high": khi, "krippendorff_alpha": krippendorff_alpha_nominal(u) if u else np.nan,
                     "note": "평점자 2명, 단위마다 구성이 달라 Fleiss kappa와 Krippendorff alpha 사용(kappa 95% 구간은 단위 bootstrap 2000회, seed q2_review.BOOT_SEED). "
                             "4명이 같은 표본 전체를 보지 않아 4인 통계는 내지 않음. 검수자 원 판정 기준 일치도, n_disagree = 검수자 불일치 건수(매칭 6, 항목 11), 불일치 건은 제3자 판정으로 확정되어 최종 미해결 0건, 일치도에는 조정 결과를 반영하지 않음"})
        pairs += [{"table": name, "pair": "-".join(sorted(x[:2])), "agree": float(x[2] == x[3]) if x[2] and x[3] else np.nan} for x in dbl]
    bp = pd.DataFrame(pairs).groupby(["table", "pair"]).agg(n=("agree", "count"), n_agree=("agree", "sum"), n_double=("agree", "size")).reset_index()
    bp["n_agree"] = bp["n_agree"].astype(int)
    bp["note"] = "검수자 쌍별 건수만 기술. n은 양쪽 판정이 있는 중복 단위"
    return pd.DataFrame(rows), bp


def unresolved_summary(adj_m: pd.DataFrame, adj_i: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame([{"table": n, "n_total": len(a), "n_confirmed": int(a["status"].eq("confirmed").sum()), "n_unresolved": int(a["status"].eq("unresolved").sum()),
                          "note": "기존 규칙상 불일치는 제3자 또는 합의 판정이 필요하며, 다수결, AI, 자동 판정으로 확정하지 않음"}
                         for n, a in (("match", adj_m), ("item", adj_i))])


# ================================================================== 보충 분석 S1~S7 (supp_*.csv, 기존 표는 바꾸지 않음)
MC_DRAWS, MC_SEED = 20000, 20261110
MC_METHOD = "층별 Jeffreys 사후 MC 결합(근사)"
MC_MEDIAN = "층별 Jeffreys 사후 MC 중앙값"
SAMPLE_PROP = "표본 비율"
SUPP_NOTE = "참고 추정. 규칙 v3.1 라벨 기반 결과의 사람 기준 보정이며 확정 값 아님"
DEFAULT_RULES = {"period": ["gov_deadline_default"], "documents": ["gov_docs_ambiguous", "gov_docs_na_consistent"]}
GOV_SCREEN_KEY = ("logs", "validation", "q2_item_rules", "gov24_screen_key.csv")
GOV_SCREEN_CHECK = ("logs", "validation", "q2_item_rules", "human", "gov24_screen_check.csv")


def jeffreys_mc_summary(ks, ms, ws, draws: int = MC_DRAWS, seed: int = MC_SEED) -> tuple[float, float, float]:
    """층 비율 p_h ~ Beta(k_h+.5, m_h-k_h+.5) 를 가중합(W_h 합 1로 정규화)해 2.5/97.5 백분위. 모든 m_h > 0 이어야 함."""
    ks, ms, ws = np.asarray(ks, float), np.asarray(ms, float), np.asarray(ws, float)
    ws = ws / ws.sum()
    rng = np.random.default_rng(seed)
    d = np.zeros(draws)
    for k, m, w in zip(ks, ms, ws):
        d += w * rng.beta(k + 0.5, m - k + 0.5, size=draws)
    lo, med, hi = np.percentile(d, [2.5, 50, 97.5])
    return float(lo), float(med), float(hi)


def jeffreys_mc_ci(ks, ms, ws, draws: int = MC_DRAWS, seed: int = MC_SEED) -> tuple[float, float]:
    lo, _, hi = jeffreys_mc_summary(ks, ms, ws, draws, seed)
    return lo, hi


def calibrated_explicitness(strata: pd.DataFrame, fin: pd.DataFrame) -> pd.DataFrame:
    """S1. strata: item, side, stratum, N_h. fin: item, side, stratum, final_label(확정 라벨, 비면 미확정).
    p_cal = sum_h W_h k_h/m_h (m_h>0 인 층만, N_h 로 재정규화). p_rule = 같은 층에서 규칙 명시 층 N / 수준 층(명시, 부분, 미명시) N 합."""
    rows = []
    for item in V.ITEMS:
        for side in IR.SIDES:
            st = strata[strata["item"].eq(item) & strata["side"].eq(side)]
            rec, det = [], []
            for r in st.itertuples():
                f = fin[fin["item"].eq(item) & fin["side"].eq(side) & fin["stratum"].eq(r.stratum)]
                lab = f["final_label"]
                n_conf, m, k = int(lab.ne("").sum()), int(lab.isin(IR.LEVELS).sum()), int(lab.eq("명시").sum())
                rec.append((r.stratum, int(r.N_h), n_conf, m, k))
                det.append(f"{r.stratum}: N={int(r.N_h)}, 확정={n_conf}, m={m}, k={k}")
            use = [x for x in rec if x[3] > 0]
            drop = [x[0] for x in rec if x[3] == 0]
            N_used = sum(x[1] for x in use)
            p_plugin = sum(x[1] / N_used * x[4] / x[3] for x in use) if use else np.nan
            lo, p_cal, hi = jeffreys_mc_summary([x[4] for x in use], [x[3] for x in use], [x[1] for x in use]) if use else (np.nan,) * 3
            lv = [x for x in use if x[0] in IR.LEVELS]
            N_lv = sum(x[1] for x in lv)
            p_rule = sum(x[1] for x in lv if x[0] == "명시") / N_lv if N_lv else np.nan
            min_m = min((x[3] for x in use), default=0)
            rows.append({"item": item, "side": side, "core_metric": item in CB.CORE_ITEMS, "N_total": sum(x[1] for x in rec), "N_used": N_used,
                         "n_strata_used": len(use), "n_strata_dropped": len(drop), "dropped_strata": ", ".join(drop),
                         "n_confirmed": sum(x[2] for x in rec), "m_total": sum(x[3] for x in use), "k_total": sum(x[4] for x in use), "min_m_h": min_m,
                         "p_rule": p_rule, "p_cal": p_cal, "estimate_definition": MC_MEDIAN, "p_plugin": p_plugin, "p_cal_ci_low": lo, "p_cal_ci_high": hi, "diff_cal_minus_rule": p_cal - p_rule,
                         "status": "참고 추정(표본 부족)" if min_m < 5 else "참고 추정",
                         "ci_method": MC_METHOD, "strata_detail": "; ".join(det),
                         "note": f"{SUPP_NOTE}. p_cal = 같은 MC 추출의 사후 중앙값(구간 안), p_plugin = 단순 층 가중값 sum W_h k_h/m_h, W_h = N_h/sum N_h(m_h>0인 층). 사람 메타 라벨(해당없음, 판단불가)은 m_h에서 제외. "
                                 "p_rule = 같은 재정규화에서 규칙 명시 층 N/수준 층(명시, 부분, 미명시) N 합(규칙 meta 층은 분모에서 제외). "
                                 f"구간 = Jeffreys 사후 {MC_DRAWS}회, seed {MC_SEED}. 보정 명시율의 플랫폼 간 비교는 표본 부족으로 검정하지 않음"})
    return pd.DataFrame(rows)


def tierA_paired_tests(root=ROOT, pt: pd.DataFrame | None = None, unres: dict | None = None) -> pd.DataFrame:
    """S2. 쌍 집합만 tier A, gov_shared False 로 바꾼 platform_paired_tests. Holm 은 이 표의 핵심 12검정 안에서."""
    d = _read(root / "data" / "processed" / "q2_items_v3_1.csv")
    d["lab"] = V.label(d)
    s = d[d["tier"].eq("A") & d["gov_shared"].eq("False")]
    t = _paired_table(s, unres or {})
    ab = pt.set_index(["scope", "item"])
    t["diff_AB_nonshared"] = [ab.at[(r.scope, r.item), "diff"] for r in t.itertuples()]
    t["n_pairs_AB_nonshared"] = [int(ab.at[(r.scope, r.item), "n_pairs"]) for r in t.itertuples()]
    t["direction_same_as_AB"] = [bool(np.sign(r.diff) == np.sign(r.diff_AB_nonshared)) for r in t.itertuples()]
    t["pair_set"] = "tier A, gov_shared False"
    t["tierA_note"] = "A 등급도 사람 검수 13/14 동일(95% CI 0.69~0.99) — A 안에도 매칭 오류 가능"
    t["holm_family"] = np.where(t["core_metric"], "핵심 6항목 x 2범위 = 12검정(A 등급 표 안)", "")
    return t


def gov24_screen(root=ROOT) -> pd.DataFrame:
    """S3a. 보조금24 화면 점검 45건을 규칙 가지별로만 집계(목적 표본, 합산 비율 없음). 자유 서술(h_note)은 코딩하지 않음."""
    key, chk = _read(root.joinpath(*GOV_SCREEN_KEY)), _read(root.joinpath(*GOV_SCREEN_CHECK))
    x = chk.merge(key[["check_id", "branch", "rule_id"]], on="check_id", how="left")
    assert x["branch"].notna().all() and len(x) == 45
    docs_na = {"해당 없음", "해당없음", "구비서류 없음"}
    specs = [("page_found", "h_page_found", lambda v: v.eq("예")),
             ("page_deadline_always", "h_page_deadline", lambda v: v.eq("상시신청")),
             ("page_docs_none", "h_page_docs", lambda v: v.isin(docs_na)),
             ("docs_match_api", "h_docs_match_api", lambda v: v.eq("일치")),
             ("docs_mismatch_api", "h_docs_match_api", lambda v: v.eq("불일치")),
             ("deadline_match_api", "h_deadline_match_api", lambda v: v.eq("일치"))]
    rows = []
    for br in ["docs_contradicted", "docs_ambiguous", "deadline_default"]:
        g = x[x["branch"].eq(br)]
        r = {"branch": br, "rule_id": g["rule_id"].iloc[0], "n": len(g)}
        for name, col, fn in specs:
            k = int(fn(g[col].str.strip()).sum())
            lo, hi = wilson(k, len(g))
            r.update({f"k_{name}": k, f"share_{name}": k / len(g), f"{name}_ci_low": lo, f"{name}_ci_high": hi})
        r["note"] = "규칙 가지별 목적 표본, 전체 일반화 불가. 가지 합산 비율은 내지 않음. 자유 서술(h_note)은 코딩하지 않음. Wilson 95% CI"
        rows.append(r)
    return pd.DataFrame(rows)


def default_rule_counts(s: pd.DataFrame) -> pd.DataFrame:
    """S3b 근거. gov 측 all 범위, 기본값 규칙(rule_id)별 라벨 건수(명시 포함)."""
    g = s[s["scope"].eq("all") & s["side"].eq("gov")]
    rows = []
    for item, rules in DEFAULT_RULES.items():
        for rid in rules:
            h = g[g["item"].eq(item) & g["rule_id"].eq(rid)]
            vc = h["lab"].value_counts()
            rows.append({"item": item, "side": "gov", "scope": "all", "rule_id": rid, "n_units": len(h), "n_explicit": int(vc.get("명시", 0)),
                         "n_partial": int(vc.get("부분", 0)), "n_not_explicit": int(vc.get("미명시", 0)),
                         "n_meta_na": int(vc.get("해당없음", 0)), "n_meta_undecidable": int(vc.get("판단불가", 0)),
                         "note": "AB_nonshared 쌍. 명시 단위만 기본값 제외 민감도 대상. 해당없음, 판단불가는 원래 분모에서 빠짐"})
    return pd.DataFrame(rows)


def default_sensitivity(s: pd.DataFrame, items=("period",)) -> pd.DataFrame:
    """S3b. text 대 all(gov 측)에서 기본값 규칙의 all 명시를 비명시로 본 민감도. s: load_items 형식(rule_id 열 포함)."""
    g = s[s["side"].eq("gov")]
    rows = []
    for item in items:
        rules = DEFAULT_RULES[item]
        q = g[g["item"].eq(item)].pivot_table(index=["servId", "gov_id"], columns="scope", values=["lab", "rule_id"], aggfunc="first")
        q.columns = [f"{a}_{b}" for a, b in q.columns]
        q = q[q["lab_text"].isin(IR.LEVELS) & q["lab_all"].isin(IR.LEVELS)]
        n = len(q)
        ty = q["lab_text"].eq("명시")
        dflt = q["lab_all"].eq("명시") & q["rule_id_all"].isin(rules)
        variants = {"original": q["lab_all"].eq("명시"), "sensitivity": q["lab_all"].eq("명시") & ~dflt}
        r = {"item": item, "side": "gov", "pair_set": "AB_nonshared", "n": n, "default_rule_ids": ", ".join(rules),
             "k_default_explicit_in_n": int(dflt.sum()), "share_text": ty.mean() if n else np.nan}
        for nm, ay in variants.items():
            add, lost = int((~ty & ay).sum()), int((ty & ~ay).sum())
            lo, hi = wilson(add, n)
            r.update({f"k_all_{nm}": int(ay.sum()), f"share_all_{nm}": ay.mean(), f"k_added_by_all_{nm}": add, f"added_by_all_{nm}": add / n,
                      f"added_ci_low_{nm}": lo, f"added_ci_high_{nm}": hi, f"k_lost_in_all_{nm}": lost, f"mcnemar_p_{nm}": mcnemar_exact(add, lost)})
        r["note"] = ("기본값 규칙의 all 범위 명시를 비명시(기본값 제외)로 바꿔 다시 계산. 참고 민감도이며 코드북 결정 전 값. 사람 금표준은 text 범위만 검증. "
                     "text 범위 단위는 바꾸지 않음")
        rows.append(r)
    return pd.DataFrame(rows)


def agreement_binary(agg: dict, ag: pd.DataFrame) -> pd.DataFrame:
    """S4. 이분 일치도(명시 대 비명시, 동일 대 비동일). 기존 agreement 값은 다범주(기존) 행으로 나란히 둠."""
    jm, ji = agg["jd_m"], agg["jd_i"]
    bm = lambda v: "동일" if v == "동일" else "비동일"
    bi = lambda v: None if v in IR.META_H else ("명시" if v == "명시" else "비명시")
    specs = [("match", [(jm.get((r.reviewer_primary, r.item_no), {}).get("h_same", ""), jm.get((r.reviewer_second, r.item_no), {}).get("h_same", ""))
                        for r in agg["asg_m"].itertuples() if r.reviewer_second], bm, "동일 대 비동일(다름, 판단불가)"),
             ("item", [(ji.get((r.reviewer_primary, r.sample_id), ""), ji.get((r.reviewer_second, r.sample_id), ""))
                       for r in agg["asg_i"].itertuples() if r.reviewer_second], bi, "명시 대 비명시(부분, 미명시)")]
    rows = []
    for name, dbl, fn, scheme in specs:
        both = [(a, b) for a, b in dbl if a and b]
        mapped = [(fn(a), fn(b)) for a, b in both]
        nmeta = sum(x is None or y is None for x, y in mapped)
        u = [[x, y] for x, y in mapped if x is not None and y is not None]
        cats = sorted({v for z in u for v in z})
        n, ags = len(u), sum(x == y for x, y in u)
        lo, hi = wilson(ags, n)
        klo, khi = _boot_kappa(u) if u else (float("nan"),) * 2
        rows.append({"table": name, "category_scheme": "이분", "n_double": len(dbl), "n_both": n, "n_meta_excluded": nmeta, "n_agree": ags, "n_disagree": n - ags,
                     "pct_agree": ags / n if n else np.nan, "pct_ci_low": lo, "pct_ci_high": hi,
                     "fleiss_kappa": fleiss([[z.count(c) for c in cats] for z in u]) if u else np.nan, "kappa_ci_low": klo, "kappa_ci_high": khi,
                     "krippendorff_alpha": krippendorff_alpha_nominal(u) if u else np.nan,
                     "note": f"{scheme}. 일치도는 검수 원 판정 기준이라 미해결 범위를 적용하지 않음. 메타 라벨(해당없음, 판단불가)이 있는 항목 단위는 n_meta_excluded로 별도 집계. "
                             "kappa 구간은 단위 bootstrap 2000회"})
    old = ag.drop(columns=["n_unresolved"], errors="ignore").assign(category_scheme="다범주(기존)", n_meta_excluded=np.nan)
    old["note"] = "기존 agreement.csv 값 참조(다범주: 항목은 명시, 부분, 미명시, 해당없음, 판단불가 / 매칭은 동일, 다름, 판단불가)"
    out = pd.concat([pd.DataFrame(rows), old], ignore_index=True)
    order = {"match": 0, "item": 1}
    out["_o"] = out["table"].map(order) * 2 + out["category_scheme"].eq("다범주(기존)").astype(int)
    return out.sort_values("_o", kind="stable").drop(columns="_o").reset_index(drop=True)


def weighted_accuracy_ci(agg: dict, ma: pd.DataFrame, ra: pd.DataFrame, root=ROOT) -> pd.DataFrame:
    """S5. 정확도 구간: 비가중은 Wilson, 층 가중은 층별 Jeffreys 사후 MC 결합. estimate_type, ci_method 를 행마다 표기."""
    rows = []
    for r in ma[ma["tier"].isin(["A", "B"])].itertuples():
        lo, hi = wilson(int(r.n_same), int(r.n_confirmed))
        rows.append({"table": "match", "key": r.tier, "side": "", "estimate_type": "비가중", "estimate": r.acc, "estimate_plugin": r.acc, "estimate_definition": SAMPLE_PROP, "ci_low": lo, "ci_high": hi, "ci_method": "Wilson",
                     "n_confirmed": int(r.n_confirmed), "n_strata_used": 1, "n_strata_dropped": 0, "old_normal_ci_low": np.nan, "old_normal_ci_high": np.nan,
                     "note": "등급별 정확도, 표본 그대로"})
    tiers = _read(root / "data" / "processed" / "q2_match.csv")
    ab = ma[ma["tier"].isin(["A", "B"])].set_index("tier")
    N = {k: int(((tiers["tier"] == k) & tiers["gov_id"].ne("")).sum()) for k in "AB"}
    w = ma[ma["tier"].eq("AB_weighted")].iloc[0]
    lo, med, hi = jeffreys_mc_summary([ab.at[k, "n_same"] for k in "AB"], [ab.at[k, "n_confirmed"] for k in "AB"], [N[k] for k in "AB"])
    rows.append({"table": "match", "key": "AB_weighted", "side": "", "estimate_type": "층 가중", "estimate": med, "estimate_plugin": w["acc"], "estimate_definition": MC_MEDIAN, "ci_low": lo, "ci_high": hi, "ci_method": MC_METHOD,
                 "n_confirmed": int(w["n_confirmed"]), "n_strata_used": 2, "n_strata_dropped": 0, "old_normal_ci_low": w["acc_ci_low"], "old_normal_ci_high": w["acc_ci_high"],
                 "note": f"층 A, B, W_h = N_h/(N_A+N_B)(N_A={N['A']}, N_B={N['B']}), p_h = n_same/n_confirmed. estimate = MC 사후 중앙값, estimate_plugin = 단순 층 가중값. old_normal_* = 기존 정규근사(상한 1 절단) 비교용"})
    fin = agg["res_i"]["final"].merge(agg["gold"][["sample_id", "stratum", "v31_level", "v31_meta"]], on="sample_id")
    fin["v31"] = fin["v31_meta"].where(fin["v31_meta"].ne(""), fin["v31_level"])
    fin["agree"] = fin["final_label"].ne("") & fin["final_label"].eq(fin["v31"])
    st = agg["strata"].copy()
    st[["N_h", "n_h"]] = st[["N_h", "n_h"]].astype(int)
    sa = IR.weighted_accuracy(st, fin)
    for r in ra.itertuples():
        x = sa[sa["item"].eq(r.item) & (sa["side"].eq(r.side) | (r.side == "all"))]
        has = x[x["n_final"].gt(0)]
        lo, hi = wilson(int(r.n_correct), int(r.n_confirmed))
        rows.append({"table": "item", "key": r.item, "side": r.side, "estimate_type": "비가중", "estimate": r.acc, "estimate_plugin": r.acc, "estimate_definition": SAMPLE_PROP, "ci_low": lo, "ci_high": hi, "ci_method": "Wilson",
                     "n_confirmed": int(r.n_confirmed), "n_strata_used": np.nan, "n_strata_dropped": np.nan, "old_normal_ci_low": np.nan, "old_normal_ci_high": np.nan,
                     "note": "규칙 v3.1 정확도, 표본 그대로"})
        wlo, wmed, whi = jeffreys_mc_summary(has["n_agree"], has["n_final"], has["N_h"]) if len(has) else (np.nan,) * 3
        rows.append({"table": "item", "key": r.item, "side": r.side, "estimate_type": "층 가중", "estimate": wmed, "estimate_plugin": r.weighted_acc, "estimate_definition": MC_MEDIAN, "ci_low": wlo, "ci_high": whi, "ci_method": MC_METHOD,
                     "n_confirmed": int(r.n_confirmed), "n_strata_used": len(has), "n_strata_dropped": len(x) - len(has), "old_normal_ci_low": np.nan, "old_normal_ci_high": np.nan,
                     "note": "gold_strata N_h 가중, 확정 단위가 없는 층은 제외하고 재정규화. 층 표본 3~4건이라 구간은 근사"})
    return pd.DataFrame(rows)


def rule_accuracy_bounds(ra: pd.DataFrame) -> pd.DataFrame:
    """S6. 미해결 단위가 모두 틀림(하한) 또는 모두 맞음(상한)일 때 규칙 정확도 범위. 일치도에는 적용하지 않음."""
    d = ra[["item", "side", "core_metric", "n_units", "n_confirmed", "n_unresolved", "n_correct", "acc"]].copy()
    tot = (d["n_confirmed"] + d["n_unresolved"]).replace(0, np.nan)
    d["acc_low_bound"] = d["n_correct"] / tot
    d["acc_high_bound"] = (d["n_correct"] + d["n_unresolved"]) / tot
    d["note"] = "acc_low_bound = n_correct/(n_confirmed+n_unresolved), acc_high_bound = (n_correct+n_unresolved)/(n_confirmed+n_unresolved). 일치도에는 적용하지 않음"
    return d


def agreement_pair_wilson(bp: pd.DataFrame) -> pd.DataFrame:
    """S7. 검수자 쌍별 일치율 + Wilson. 쌍별 kappa 는 내지 않음."""
    d = bp[["table", "pair", "n", "n_agree"]].copy()
    d["pct_agree"] = d["n_agree"] / d["n"]
    ci = [wilson(int(a), int(b)) for a, b in zip(d["n_agree"], d["n"])]
    d["pct_ci_low"], d["pct_ci_high"] = [c[0] for c in ci], [c[1] for c in ci]
    d["note"] = "쌍별 n 4~5건, 구간 매우 넓음. 쌍별 kappa는 내지 않음"
    return d


def build_supp(root=ROOT, tabs: dict | None = None, agg: dict | None = None) -> dict:
    tabs = tabs if tabs is not None else build_all(root)
    agg = agg if agg is not None else aggregate_effective(root)
    fin = agg["res_i"]["final"].merge(agg["gold"][["sample_id", "stratum"]], on="sample_id")
    st = agg["strata"].copy()
    st["N_h"] = st["N_h"].astype(int)
    adj_i = tabs["adjudication_item"]
    unres = adj_i[adj_i["status"].eq("unresolved")]["field"].str.split("_").str[0].value_counts().to_dict()
    d = _read(root / "data" / "processed" / "q2_items_v3_1.csv")
    d["lab"] = V.label(d)
    s = d[d["tier"].isin(["A", "B"]) & d["gov_shared"].eq("False")]
    return {"supp_calibrated_explicitness": calibrated_explicitness(st, fin),
            "supp_tierA_paired_tests": tierA_paired_tests(root, tabs["platform_paired_tests"], unres),
            "supp_gov24_screen": gov24_screen(root),
            "supp_default_sensitivity": default_sensitivity(s),
            "supp_default_rule_counts": default_rule_counts(s),
            "supp_agreement_binary": agreement_binary(agg, tabs["agreement"]),
            "supp_weighted_accuracy_ci": weighted_accuracy_ci(agg, tabs["match_accuracy"], tabs["item_rule_accuracy"], root),
            "supp_rule_accuracy_bounds": rule_accuracy_bounds(tabs["item_rule_accuracy"]),
            "supp_agreement_by_pair": agreement_pair_wilson(tabs["agreement_by_pair"])}


def build_all(root=ROOT) -> dict:
    agg = aggregate_effective(root)
    adj_m, adj_i = adjudication_tables(agg)
    um, ui = unresolved_lists(agg, adj_m, adj_i)
    unres = adj_i[adj_i["status"].eq("unresolved")]["field"].str.split("_").str[0].value_counts().to_dict()
    ex, pt, tv = explicitness_tables(root, unres)
    ag, bp = agreement_tables(agg)
    return {"adjudication_match": adj_m, "adjudication_item": adj_i, "unresolved_match": um, "unresolved_item": ui,
            "match_accuracy": match_accuracy(agg, root), "item_rule_accuracy": item_rule_accuracy(agg),
            "explicitness_by_platform": ex, "platform_paired_tests": pt, "text_vs_all": tv,
            "agreement": ag, "agreement_by_pair": bp, "unresolved_summary": unresolved_summary(adj_m, adj_i)}


def write_all(root=ROOT) -> dict:
    tabs = build_all(root)
    out = root / OUTDIR
    out.mkdir(parents=True, exist_ok=True)
    for k, v in tabs.items():
        v.to_csv(out / f"{k}.csv", index=False, encoding="utf-8-sig", lineterminator="\n")
    supp = build_supp(root, tabs)
    for k, v in supp.items():
        v.to_csv(out / f"{k}.csv", index=False, encoding="utf-8-sig", lineterminator="\n")
    tabs.update(supp)
    return tabs


if __name__ == "__main__":
    lg = fix_reviewer_files()
    print(lg["status"].value_counts().to_string())
    tabs = write_all()
    for k in ("unresolved_summary", "match_accuracy", "agreement"):
        print(f"\n[{k}]\n{tabs[k].to_string()}")
