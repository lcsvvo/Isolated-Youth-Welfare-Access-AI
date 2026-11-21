"""Q2 보완 분석: 서비스 단위 신청 정보 완성도, 청년 대상(Q1 엄격 S2) 세부, 양쪽 내용 비교.

기존 결과(q2_final)를 다시 계산하지 않는다. 규칙 v3.1 라벨(q2_items_v3_1.csv)을 서비스(쌍) 단위로 묶어 본다.
  - 완성도: 한 쌍의 핵심 6항목 중 '명시'인 항목 수. SSIS 단독, 보조금24 단독, 두 출처 통합(항목별 더 높은 수준)
  - 분모: 해당없음은 필요한 항목에서 뺀다. 부분, 판단불가, 미명시는 부족으로 세고 따로 구분한다
  - 청년: Q1 PR #42 기준 엄격 S2 서비스 사본(q2_q1_youth_layer.csv)과 servId로 연결. Q1 코드와 마스터는 읽기만 한다
  - 내용 비교: 양쪽 원문에서 뽑은 값(연령 경계, 소득 기준, 신청 경로, 신청 기한, 전화번호)의 일치, 보완, 상이 후보.
    자동 후보 쌍의 '상이'는 동일 사업 여부가 확인되지 않았으므로 충돌이 아니라 후보다. 충돌 서술은 사람이 동일로 확정한 쌍에 한정하고, 공식 기준 확인 결과와 구분해 적는다

명시 여부는 설명문에 정보가 적혀 있는지에 대한 규칙 판정이다. 실제 자격 충족이나 신청 가능 여부가 아니다.

실행: python -m src.preprocess.q2_completeness  → outputs/04_q2/tables/final/svc_*.csv, youth_*.csv, content_*.csv
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from ..collect._common import PROCESSED, ROOT
from . import q2_codebook as CB
from . import q2_final as QF
from . import q2_items_v3 as V
from .ssis_master import extract_age

OUTDIR = QF.OUTDIR
ITEMS6 = list(CB.CORE_ITEMS)                       # 연령, 소득, 신청방법, 신청기간, 구비서류, 문의처
ITEMS7 = ITEMS6 + ["residence"]                    # 거주는 보조 지표
GROUPS = {"eligibility": ["age", "income"], "application": ["channel", "period", "documents", "contact"]}
SOURCES = ["ssis", "gov", "union"]
# 통합 시 항목 수준: 더 높은 쪽을 취한다. 해당없음(신청 절차 없음)은 부분보다 위에 둔다(필요 없음이 확인된 상태)
RANK = {"명시": 4, "해당없음": 3, "부분": 2, "판단불가": 1, "미명시": 0}
INV = {v: k for k, v in RANK.items()}
DEFAULT_RULES = {"gov_deadline_default"}           # 보조금24 신청기한 '상시신청' 기본값 후보
YOUTH = PROCESSED / "q2_q1_youth_layer.csv"
AREA_KO = {"q21_02": "② 일상생활 회복", "q21_03": "③ 혼자 하는 활동", "q21_05": "⑤ 전문 상담·치료", "q21_06": "⑥ 또래·멘토 상담",
           "q21_07": "⑦ 교육·학습", "q21_08": "⑧ 취업·일 경험", "q21_10": "⑩ 경제적 지원", "q21_11": "⑪ 주거"}
COUNSEL = ["q21_05", "q21_06"]
JUDGED_BY = "사람 검수 완료"
NOTE_RULE = "규칙 v3.1 라벨 기반. 명시 = 설명문에 적혀 있음(자격 충족이나 신청 가능 여부 아님). 사람 금표준은 text 범위 128단위뿐"


# ================================================================== 입력
def load(root=ROOT) -> dict:
    d = V.read(root / "data" / "processed" / "q2_items_v3_1.csv")
    d["lab"] = V.label(d)
    assert d["lab"].isin(RANK).all(), sorted(set(d["lab"]) - set(RANK))
    q = V.read(root / "data" / "processed" / "q2_match.csv")
    y = V.read(root / "data" / "processed" / "q2_q1_youth_layer.csv")
    return {"items": d, "match": q, "youth": y}


def confirmed_pairs(adj_match: pd.DataFrame, root=ROOT) -> pd.DataFrame:
    """사람 검수 72쌍의 확정 판정(동일, 다름, 판단불가)과 servId, gov_id."""
    asg = V.read(root / "logs" / "validation" / "q2_match" / "review_assignment.csv")
    out = adj_match[["item_id", "final", "final_source"]].merge(asg[["item_no", "servId", "gov_id"]], left_on="item_id", right_on="item_no",
                                                                validate="1:1")
    out = out[["servId", "gov_id", "final", "final_source"]].rename(columns={"final": "human_final"})
    pr = post_review(root)
    fix = {(r.servId, r.gov_id): r.post_review_final for r in pr.itertuples()}
    key = list(zip(out["servId"], out["gov_id"]))
    assert set(fix) <= set(key), "사후 정정 쌍이 검수 72쌍에 없음"
    out["post_review"] = [k in fix for k in key]
    out["final"] = [fix.get(k, v) for k, v in zip(key, out["human_final"])]
    return out


POST_REVIEW = ROOT / "logs" / "validation" / "q2_final" / "match_post_review.csv"


def post_review(root=ROOT) -> pd.DataFrame:
    """검수 확정 뒤 내용 대조에서 드러난 매칭 정정. 검수 원본과 조정 판정표는 그대로 두고 이 분석 층에서만 반영한다."""
    p = root / "logs" / "validation" / "q2_final" / "match_post_review.csv"
    cols = ["servId", "gov_id", "human_final", "post_review_final"]
    return V.read(p) if p.exists() else pd.DataFrame(columns=cols)


def match_accuracy_post_review(ft: dict, conf: pd.DataFrame, root=ROOT) -> pd.DataFrame:
    """사후 정정을 반영한 등급별 매칭 정확도. 기존 match_accuracy(검수 판정 그대로)와 나란히 둔다."""
    q = V.read(root / "data" / "processed" / "q2_match.csv")
    tier = conf["servId"].map(q.set_index("servId")["tier"])
    old = ft["match_accuracy"].set_index("tier")
    rows = []
    for t in "ABCU":
        g = conf[tier.eq(t)]
        k, n = int(g["final"].eq("동일").sum()), len(g)
        lo, hi = QF.wilson(k, n)
        rows.append({"tier": t, "n_confirmed": n, "n_same": k, "n_same_before": int(old.at[t, "n_same"]), "n_post_review": int(g["post_review"].sum()),
                     "acc": k / n, "acc_ci_low": lo, "acc_ci_high": hi, "acc_before": old.at[t, "acc"], "method": "Wilson 95%"})
    N = {t: int(((q["tier"] == t) & q["gov_id"].ne("")).sum()) for t in "AB"}
    ab = {r["tier"]: r for r in rows if r["tier"] in "AB"}
    lo, med, hi = QF.jeffreys_mc_summary([ab[t]["n_same"] for t in "AB"], [ab[t]["n_confirmed"] for t in "AB"], [N[t] for t in "AB"])
    _, med0, _ = QF.jeffreys_mc_summary([ab[t]["n_same_before"] for t in "AB"], [ab[t]["n_confirmed"] for t in "AB"], [N[t] for t in "AB"])
    rows.append({"tier": "AB_weighted", "n_confirmed": sum(ab[t]["n_confirmed"] for t in "AB"), "n_same": sum(ab[t]["n_same"] for t in "AB"),
                 "n_same_before": sum(ab[t]["n_same_before"] for t in "AB"), "n_post_review": sum(ab[t]["n_post_review"] for t in "AB"),
                 "acc": med, "acc_ci_low": lo, "acc_ci_high": hi, "acc_before": med0, "method": "층 N_h 가중, 층별 Jeffreys 사후 MC 중앙값과 95% 구간"})
    out = pd.DataFrame(rows)
    out["note"] = "acc_before = 검수 판정 그대로의 값(match_accuracy.csv, supp_weighted_accuracy_ci.csv). 사후 정정 = match_post_review.csv"
    return out


def wide(d: pd.DataFrame, default_as_missing: bool = False) -> pd.DataFrame:
    """쌍 × (scope, side, item) 순위 표. default_as_missing이면 보조금24 기본값 후보 규칙의 명시를 미명시로 본다."""
    x = d.copy()
    if default_as_missing:
        hit = x["rule_id"].isin(DEFAULT_RULES) & x["lab"].eq("명시")
        x.loc[hit, "lab"] = "미명시"
    x["r"] = x["lab"].map(RANK)
    w = x.pivot(index=["servId", "gov_id"], columns=["scope", "side", "item"], values="r")
    for scope in ("text", "all"):
        for item in ITEMS7:
            w[(scope, "union", item)] = np.maximum(w[(scope, "ssis", item)], w[(scope, "gov", item)])
    return w.sort_index(axis=1)


def populations(q: pd.DataFrame, conf: pd.DataFrame, youth: pd.DataFrame) -> dict[str, set]:
    p = q[q["tier"].ne("U") & q["gov_id"].ne("")]
    key = lambda f: set(zip(f["servId"], f["gov_id"]))
    ab = p[p["tier"].isin(["A", "B"]) & p["gov_shared"].eq("False")]
    same = key(conf[conf["final"].eq("동일")]) & key(p)
    ys = set(youth["servId"])
    yab = ab[ab["servId"].isin(ys)]
    return {"AB_nonshared": key(ab), "A_nonshared": key(ab[ab["tier"].eq("A")]), "ABC_all": key(p), "human_same": same,
            "human_same_AB": same & key(ab), "youth_S2_AB": key(yab), "youth_S2_A": key(yab[yab["tier"].eq("A")]),
            "nonyouth_AB": key(ab) - key(yab)}


POP_KO = {"AB_nonshared": "자동 후보 A·B 비공유(주)", "A_nonshared": "자동 후보 A 비공유(민감도)", "ABC_all": "자동 후보 A·B·C 전체(참고)",
          "human_same": "사람 확정 동일(A·B·C)", "human_same_AB": "사람 확정 동일(A·B 비공유)", "youth_S2_AB": "청년 엄격 S2 ∩ A·B 비공유",
          "youth_S2_A": "청년 엄격 S2 ∩ A 비공유", "nonyouth_AB": "엄격 S2 밖 ∩ A·B 비공유"}


# ================================================================== 서비스 단위 지표
def service_metrics(w: pd.DataFrame, scope: str, items=ITEMS6, lenient: bool = False) -> pd.DataFrame:
    """쌍 × 출처별 필요한 항목 수, 명시 수, 부족 수. lenient는 부분도 확인으로 센다(민감도)."""
    ok_rank = {4, 2} if lenient else {4}
    rows = []
    for src in SOURCES:
        m = w[scope][src][list(items)]
        ok = m.isin(ok_rank)
        na = m.eq(3)
        f = pd.DataFrame({"source": src, "n_applicable": len(items) - na.sum(axis=1), "n_explicit": ok.sum(axis=1),
                          "n_partial": 0 if lenient else m.eq(2).sum(axis=1), "n_undecidable": m.eq(1).sum(axis=1),
                          "n_missing": m.eq(0).sum(axis=1)})
        f["n_short"] = f["n_applicable"] - f["n_explicit"]
        f["complete"] = f["n_short"].eq(0)
        for g, its in GROUPS.items():
            its = [i for i in its if i in items]
            f[f"complete_{g}"] = (ok[its] | na[its]).all(axis=1)
        rows.append(f.reset_index())
    return pd.concat(rows, ignore_index=True)


def _sel(f: pd.DataFrame, keys: set) -> pd.DataFrame:
    return f[[k in keys for k in zip(f["servId"], f["gov_id"])]]


VARIANTS = {  # (기본값 후보 처리, 부분 인정, 항목 세트)
    "main": (False, False, ITEMS6), "default_excluded": (True, False, ITEMS6), "partial_counted": (False, True, ITEMS6),
    "with_residence": (False, False, ITEMS7), "without_age": (False, False, [i for i in ITEMS6 if i != "age"])}
VAR_KO = {"main": "기준(명시만, 핵심 6항목)", "default_excluded": "'상시신청' 기본값 후보 제외", "partial_counted": "부분도 확인으로 인정",
          "with_residence": "거주 포함 7항목(보조)", "without_age": "연령 제외 5항목(청년 비교용)"}


def completeness_summary(d: pd.DataFrame, pops: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """모집단 × 범위 × 변형 × 출처별 요약과 부족 항목 수 분포."""
    summ, dist = [], []
    for var, (dflt, len_, items) in VARIANTS.items():
        w = wide(d, dflt)
        for scope in ("text", "all"):
            sm = service_metrics(w, scope, items, len_)
            for pop, keys in pops.items():
                g = _sel(sm, keys)
                for src in SOURCES:
                    h = g[g["source"].eq(src)]
                    n, k = len(h), int(h["complete"].sum())
                    lo, hi = QF.wilson(k, n)
                    row = {"population": pop, "scope": scope, "variant": var, "source": src, "n_pairs": n, "n_items": len(items),
                           "mean_applicable": h["n_applicable"].mean(), "mean_explicit": h["n_explicit"].mean(),
                           "mean_short": h["n_short"].mean(), "median_short": h["n_short"].median(),
                           "n_complete": k, "share_complete": k / n if n else np.nan, "complete_ci_low": lo, "complete_ci_high": hi,
                           "share_short_ge3": h["n_short"].ge(3).mean()}
                    for grp in GROUPS:
                        kk = int(h[f"complete_{grp}"].sum())
                        glo, ghi = QF.wilson(kk, n)
                        row.update({f"n_complete_{grp}": kk, f"share_complete_{grp}": kk / n if n else np.nan,
                                    f"{grp}_ci_low": glo, f"{grp}_ci_high": ghi})
                    summ.append(row)
                    if var == "main":
                        vc = h["n_short"].value_counts()
                        dist += [{"population": pop, "scope": scope, "source": src, "n_short": s, "n_pairs": int(vc.get(s, 0)),
                                  "share": vc.get(s, 0) / n if n else np.nan} for s in range(len(items) + 1)]
    s = pd.DataFrame(summ)
    s["population_label"], s["variant_label"], s["note"] = s["population"].map(POP_KO), s["variant"].map(VAR_KO), NOTE_RULE
    t = pd.DataFrame(dist)
    t["note"] = "n_short = 필요한 항목(해당없음 제외) 중 명시가 아닌 항목 수. 핵심 6항목, 기준 변형"
    return s, t


def item_status(d: pd.DataFrame, pops: dict) -> pd.DataFrame:
    """모집단 × 범위 × 항목 × 출처별 상태 건수(명시, 부분, 미명시, 해당없음, 판단불가)."""
    w = wide(d)
    rows = []
    for pop, keys in pops.items():
        g = w[[k in keys for k in w.index]]
        for scope in ("text", "all"):
            for item in ITEMS7:
                for src in SOURCES:
                    vc = g[(scope, src, item)].map(INV).value_counts()
                    n_app = len(g) - int(vc.get("해당없음", 0))
                    rows.append({"population": pop, "scope": scope, "item": item, "source": src, "core_metric": item in ITEMS6, "n_pairs": len(g),
                                 **{f"n_{k}": int(vc.get(k, 0)) for k in RANK}, "n_applicable": n_app,
                                 "share_explicit_of_applicable": vc.get("명시", 0) / n_app if n_app else np.nan})
    t = pd.DataFrame(rows)
    t["note"] = "분모 n_applicable = 쌍 수 − 해당없음. 판단불가도 분모에 남긴다(서비스 단위 지표와 같은 분모). 기존 explicitness_by_platform은 판단불가를 분모에서 뺀다"
    return t


def integration_gain(d: pd.DataFrame, pops: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """통합으로 새로 명시가 되는 항목. 항목별(누가 보태는가)과 서비스별(몇 항목이 늘어나는가)."""
    by_item, by_svc = [], []
    for var in ("main", "default_excluded"):
        w = wide(d, VARIANTS[var][0])
        for pop, keys in pops.items():
            g = w[[k in keys for k in w.index]]
            n = len(g)
            for scope in ("text", "all"):
                s, v, u = (g[scope][x][ITEMS7] for x in SOURCES)
                es, ev = s.eq(4), v.eq(4)
                need = ~u.eq(3)                                   # 통합 기준 해당없음이 아닌 항목
                for item in ITEMS7:
                    a, b = es[item], ev[item]
                    nd = int(need[item].sum())
                    by_item.append({"population": pop, "scope": scope, "variant": var, "item": item, "core_metric": item in ITEMS6, "n_pairs": n,
                                    "n_needed": nd, "n_both": int((a & b).sum()), "n_ssis_only": int((a & ~b).sum()),
                                    "n_gov_only": int((~a & b).sum()), "n_neither": int((~a & ~b & need[item]).sum()),
                                    "share_union": (a | b).sum() / nd if nd else np.nan,
                                    "gain_for_ssis_user": (~a & b).sum() / nd if nd else np.nan,
                                    "gain_for_gov_user": (a & ~b).sum() / nd if nd else np.nan,
                                    "share_neither": (~a & ~b & need[item]).sum() / nd if nd else np.nan})
                k6 = ITEMS6
                ns, ng, nu = es[k6].sum(axis=1), ev[k6].sum(axis=1), (es[k6] | ev[k6]).sum(axis=1)
                best = np.maximum(ns, ng)
                for ref, gain in {"vs_ssis": nu - ns, "vs_gov": nu - ng, "vs_better_single": nu - best}.items():
                    k = int(gain.ge(1).sum())
                    lo, hi = QF.wilson(k, n)
                    by_svc.append({"population": pop, "scope": scope, "variant": var, "reference": ref, "n_pairs": n,
                                   "mean_gain_items": gain.mean(), "n_gain_ge1": k, "share_gain_ge1": k / n if n else np.nan,
                                   "gain_ci_low": lo, "gain_ci_high": hi, "share_gain_ge2": gain.ge(2).mean(),
                                   **{f"n_gain_{j}": int(gain.eq(j).sum()) for j in range(7)}})
    a, b = pd.DataFrame(by_item), pd.DataFrame(by_svc)
    a["note"] = "gain_for_ssis_user = SSIS 비명시이고 보조금24 명시인 쌍 / 통합 기준 필요한 쌍. 자동 후보 쌍은 동일 사업이 아닐 수 있음"
    b["note"] = "핵심 6항목. vs_better_single = 통합 명시 수 − 두 출처 중 더 많은 쪽의 명시 수"
    return a, b


# ================================================================== 청년(Q1 엄격 S2)
def youth_tables(d: pd.DataFrame, q: pd.DataFrame, youth: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(1) 자동 후보 등급 분포: 전체 대 엄격 S2, (2) 영역별 완성도, (3) 영역별 항목 부족."""
    ys = set(youth["servId"])
    q = q.assign(group=np.where(q["servId"].isin(ys), "youth_S2_strict", "other"))
    cov = []
    for grp, g in [("all_ssis", q), ("youth_S2_strict", q[q["group"].eq("youth_S2_strict")]), ("other", q[q["group"].eq("other")])]:
        vc = g["tier"].value_counts()
        ab = g[g["tier"].isin(["A", "B"]) & g["gov_shared"].eq("False")]
        lo, hi = QF.wilson(len(ab), len(g))
        cov.append({"group": grp, "n_services": len(g), **{f"n_tier_{t}": int(vc.get(t, 0)) for t in "ABCU"},
                    "n_AB_nonshared": len(ab), "share_AB_nonshared": len(ab) / len(g), "ci_low": lo, "ci_high": hi,
                    "share_tier_U": vc.get("U", 0) / len(g)})
    cov = pd.DataFrame(cov)
    cov["note"] = "등급은 자동 후보 기준. U는 '자동 후보 없음'이며 보조금24에 없다는 뜻이 아님(사람 검수 U 15쌍 중 1쌍 동일)"

    ab = q[q["tier"].isin(["A", "B"]) & q["gov_shared"].eq("False")]
    y = youth.merge(ab[["servId", "gov_id", "tier"]], on="servId", validate="1:1")
    areas = {a: set(zip(y.loc[y[a].eq("1"), "servId"], y.loc[y[a].eq("1"), "gov_id"])) for a in AREA_KO}
    areas["counsel_05_06"] = areas["q21_05"] | areas["q21_06"]
    areas["youth_S2_AB"] = set(zip(y["servId"], y["gov_id"]))
    label = {**AREA_KO, "counsel_05_06": "⑤+⑥ 상담(합산)", "youth_S2_AB": "엄격 S2 전체"}
    rows, irows = [], []
    for var in ("main", "default_excluded"):
        w = wide(d, VARIANTS[var][0])
        for scope in ("text", "all"):
            sm = service_metrics(w, scope)
            for a, keys in areas.items():
                g = _sel(sm, keys)
                n = len(keys)
                row = {"area": a, "area_label": label[a], "scope": scope, "variant": var, "n_pairs": n,
                       "n_supply_S2": int(youth[a].eq("1").sum()) if a in AREA_KO else (len(youth) if a == "youth_S2_AB" else int((youth[COUNSEL] == "1").any(axis=1).sum())),
                       "small_n": n < 20}
                for src in SOURCES:
                    h = g[g["source"].eq(src)]
                    k = int(h["complete"].sum())
                    lo, hi = QF.wilson(k, n)
                    row.update({f"mean_explicit_{src}": h["n_explicit"].mean(), f"mean_short_{src}": h["n_short"].mean(),
                                f"n_complete_{src}": k, f"share_complete_{src}": k / n if n else np.nan,
                                f"complete_{src}_ci_low": lo, f"complete_{src}_ci_high": hi,
                                f"share_application_{src}": h["complete_application"].mean(), f"share_eligibility_{src}": h["complete_eligibility"].mean()})
                row["gain_vs_better_single"] = row["mean_explicit_union"] - _best_mean(g)
                rows.append(row)
                if var == "main":
                    gw = w[[k in keys for k in w.index]]
                    for item in ITEMS6:
                        u, s_, v_ = (gw[(scope, x, item)] for x in ("union", "ssis", "gov"))
                        need = int((~u.eq(3)).sum())
                        irows.append({"area": a, "area_label": label[a], "scope": scope, "item": item, "n_pairs": n, "n_needed": need,
                                      "n_explicit_ssis": int(s_.eq(4).sum()), "n_explicit_gov": int(v_.eq(4).sum()), "n_explicit_union": int(u.eq(4).sum()),
                                      "n_short_union": int((~u.isin([4, 3])).sum()),
                                      "share_short_union": (~u.isin([4, 3])).sum() / need if need else np.nan, "small_n": n < 20})
    t, it = pd.DataFrame(rows), pd.DataFrame(irows)
    t["note"] = "영역은 Q1 PR #42(0ffe286) 엄격 S2 × v4_main. 한 서비스가 여러 영역에 속할 수 있음. n_pairs < 20은 참고만"
    it["note"] = "share_short_union = 통합해도 명시가 아닌 쌍 / 필요한 쌍"
    return cov, t, it


def _best_mean(g: pd.DataFrame) -> float:
    p = g.pivot(index=["servId", "gov_id"], columns="source", values="n_explicit")
    return float(np.maximum(p["ssis"], p["gov"]).mean()) if len(p) else np.nan


# ================================================================== 내용 비교
_PHONE = re.compile(r"(?<!\d)(0\d{1,2})[-)\s.]?(\d{3,4})[-\s.]?(\d{4})(?!\d)|(?<!\d)(1[0-9]{3})[-\s]?(\d{4})(?!\d)|(?<![\d-])(1[0-9]{2})(?![\d-])")
_MEDIAN = re.compile(r"중위\s*소득(?:\s*\([^)]{0,12}\))?\s*(?:의)?\s*(\d{2,3})\s*%")
_STATUS = {"기초수급": r"기초\s*(?:생활)?\s*(?:보장)?\s*수급|생계\s*[·,ㆍ]?\s*의료\s*(?:급여)?\s*수급|국민기초", "차상위": r"차상위", "소득무관": r"소득\s*(?:과|에)?\s*(?:무관|관계\s*없)"}
_CH = {"방문": r"방문|내방|주민센터|행정복지센터|읍[·ㆍ/]?면[·ㆍ/]?동|읍면사무소", "온라인": r"온라인|인터넷|누리집|홈페이지|복지로|정부24|https?://|www\.|앱",
       "우편": r"우편", "전화": r"전화|유선", "팩스": r"팩스|FAX|fax", "신청불필요": r"신청\s*불필요|신청\s*(?:절차|사항)\s*없|별도\s*신청\s*(?:없|불필요|사항\s*없)|직권"}
_EVENT = re.compile(r"(\d{1,3})\s*(일|개월|년)\s*(?:이내|내에|안에)")
_ALWAYS = re.compile(r"상시|수시|연중|언제든지")
_DATED = re.compile(r"\d{4}\s*[.년]\s*\d{1,2}\s*[.월]|\d{1,2}\s*월|\d{1,2}\.\s*\d{1,2}\.")


def phones(t: str) -> frozenset:
    return frozenset("".join(x for x in m if x) for m in _PHONE.findall(V.nfkc(t)))


def age_value(F: dict) -> tuple:
    lo, hi, *_ = extract_age([F.get("target", ""), F.get("criteria", "")])
    return () if (lo, hi) == ("N", "N") else (str(lo), str(hi))


def income_value(F: dict) -> frozenset:
    t = V.nfkc(F.get("target", "") + "\n" + F.get("criteria", ""))
    out = {f"중위{n}%" for n in _MEDIAN.findall(t)}
    out |= {k for k, p in _STATUS.items() if re.search(p, t)}
    return frozenset(out)


def channel_value(F: dict) -> frozenset:
    t = V.nfkc(F.get("apply", "") + "\n" + F.get("apply_cat", ""))
    return frozenset(k for k, p in _CH.items() if re.search(p, t))


def period_value(F: dict) -> frozenset:
    """신청 본문과 신청기한 필드의 기한 표현. '상시신청' 필드 값은 기본값 후보라 본문의 상시와 구분한다."""
    t, dl = V.nfkc(F.get("apply", "")), V.nfkc(F.get("deadline", "")).strip()
    out = {f"{n}{u}이내" for n, u in _EVENT.findall(t + "\n" + dl)}
    if _ALWAYS.search(t) or (dl and dl != "상시신청" and _ALWAYS.search(dl)):
        out.add("상시")
    if dl == "상시신청":
        out.add("상시(필드 기본값 후보)")
    if _DATED.search(t) or _DATED.search(dl):
        out.add("날짜 지정")
    return frozenset(out)


def _cmp_sets(a: frozenset, b: frozenset) -> str:
    if not a and not b:
        return "양쪽 값 없음"
    if not a or not b:
        return "한쪽만 값"
    if a == b:
        return "일치"
    return "보완(포함 관계)" if (a <= b or b <= a) else "상이 후보"


def _cmp_age(a: tuple, b: tuple) -> str:
    if not a and not b:
        return "양쪽 값 없음"
    if not a or not b:
        return "한쪽만 값"
    if a == b:
        return "일치"
    one_bound = any(x == y and x != "N" for x, y in zip(a, b)) and any("N" in (x, y) for x, y in zip(a, b))
    return "보완(포함 관계)" if one_bound else "상이 후보"


def _cmp_period(a: frozenset, b: frozenset) -> str:
    ev_a, ev_b = {x for x in a if x.endswith("이내")}, {x for x in b if x.endswith("이내")}
    core = lambda s: {x for x in s if x != "상시(필드 기본값 후보)"}
    if not a and not b:
        return "양쪽 값 없음"
    if not core(a) and not core(b):
        return "기본값 후보만"
    if not core(a) or not core(b):
        return "한쪽만 값"
    if ev_a and ev_b:
        return "일치" if ev_a == ev_b else ("보완(포함 관계)" if (ev_a <= ev_b or ev_b <= ev_a) else "상이 후보")
    if core(a) == core(b):
        return "일치"
    return "보완(포함 관계)" if (core(a) <= core(b) or core(b) <= core(a)) else "상이 후보"


EXTRACT = {"age": age_value, "income": income_value, "channel": channel_value, "period": period_value,
           "contact": lambda F: phones(F.get("contact", "") + "\n" + F.get("apply", ""))}
COMPARE = {"age": _cmp_age, "income": _cmp_sets, "channel": _cmp_sets, "period": _cmp_period, "contact": lambda a, b: (
    "양쪽 값 없음" if not a and not b else "한쪽만 값" if not a or not b else "일치" if a == b else "보완(포함 관계)" if a & b else "상이 후보")}
CLASSES = ["일치", "보완(포함 관계)", "상이 후보", "한쪽만 값", "기본값 후보만", "양쪽 값 없음"]
CONTENT_ITEMS = list(EXTRACT)
_SHOW = lambda v: " ~ ".join(v) if isinstance(v, tuple) else ", ".join(sorted(v))


def content_compare(q: pd.DataFrame, sf: pd.DataFrame, gf: pd.DataFrame) -> pd.DataFrame:
    """자동 후보 쌍(A, B, C) × 항목별 양쪽 추출값과 비교 유형. 범위는 all(구조화 필드 포함)."""
    p = q[q["tier"].ne("U") & q["gov_id"].ne("")]
    rows = []
    for r in p.itertuples(index=False):
        S, G = V.side_fields(sf, gf, r.servId, r.gov_id, "ssis"), V.side_fields(sf, gf, r.servId, r.gov_id, "gov")
        for item in CONTENT_ITEMS:
            a, b = EXTRACT[item](S), EXTRACT[item](G)
            rows.append({"servId": r.servId, "gov_id": r.gov_id, "tier": r.tier, "gov_shared": r.gov_shared, "item": item,
                         "ssis_value": _SHOW(a), "gov_value": _SHOW(b), "compare": COMPARE[item](a, b)})
    return pd.DataFrame(rows)


def content_summary(cc: pd.DataFrame, pops: dict) -> pd.DataFrame:
    rows = []
    for pop in ("AB_nonshared", "A_nonshared", "human_same", "human_same_AB", "youth_S2_AB"):
        g = _sel(cc, pops[pop])
        for item in CONTENT_ITEMS:
            h = g[g["item"].eq(item)]
            vc = h["compare"].value_counts()
            both = int(sum(vc.get(c, 0) for c in CLASSES[:3]))
            k = int(vc.get("상이 후보", 0))
            lo, hi = QF.wilson(k, both)
            rows.append({"population": pop, "population_label": POP_KO[pop], "item": item, "n_pairs": len(h),
                         **{f"n_{c}": int(vc.get(c, 0)) for c in CLASSES}, "n_both_valued": both,
                         "share_differ_of_both": k / both if both else np.nan, "differ_ci_low": lo, "differ_ci_high": hi,
                         "identity_checked": pop.startswith("human_same")})
    t = pd.DataFrame(rows)
    t["note"] = ("추출값 비교(규칙). 상이 후보는 값이 다르게 뽑힌 쌍이며 충돌 확정이 아님. identity_checked=False 모집단은 동일 사업 여부 미확인. "
                 "분모 n_both_valued = 양쪽 모두 값이 뽑힌 쌍")
    return t


def content_cases(cc: pd.DataFrame, pops: dict, sf, gf, master: pd.DataFrame, gov: pd.DataFrame, review: pd.DataFrame | None = None) -> pd.DataFrame:
    """사람이 동일로 확정한 쌍의 항목별 비교와 양쪽 원문, 출처 URL. review가 있으면 원문 대조 판정을 붙인다."""
    g = _sel(cc, pops["human_same"])
    g = g[g["compare"].isin(["보완(포함 관계)", "상이 후보"])].copy()
    FIELD = {"age": ["target", "criteria"], "income": ["target", "criteria"], "channel": ["apply", "apply_cat"], "period": ["apply", "deadline"],
             "contact": ["contact"]}
    clip = lambda s: re.sub(r"\s+", " ", V.nfkc(s)).strip()[:600]
    txt = lambda row, item: " / ".join(f"[{k}] {clip(row[k])}" for k in FIELD[item] if k in row and clip(row[k]))
    g["ssis_text"] = [txt(sf.loc[r.servId], r.item) for r in g.itertuples()]
    g["gov_text"] = [txt(gf.loc[r.gov_id], r.item) for r in g.itertuples()]
    mi, gi = master.set_index("servId"), gov.set_index("서비스ID")
    g["ssis_name"], g["ssis_url"] = g["servId"].map(mi["servNm"]), g["servId"].map(mi["detail_url"])
    g["gov_name"], g["gov_url"] = g["gov_id"].map(gi["서비스명"]), g["gov_id"].map(gi["상세조회URL"])
    g["collected"] = "SSIS 2026-10-30, 보조금24 2026-10-31"
    if review is not None and len(review):
        g = g.merge(review[["servId", "gov_id", "item", *REVIEW_COLS]], on=["servId", "gov_id", "item"], how="left")
    cols = ["servId", "gov_id", "tier", "gov_shared", "item", "compare", "ssis_value", "gov_value", *REVIEW_COLS,
            "ssis_name", "gov_name", "ssis_text", "gov_text", "ssis_url", "gov_url", "collected"]
    return g[[c for c in cols if c in g]].sort_values(["item", "compare", "servId"]).reset_index(drop=True)


REVIEW = ROOT / "logs" / "validation" / "q2_final" / "content_conflict_review.csv"
# 수집 당시 플랫폼 간 차이(사실)와 공식 기준 확인 결과, 최종 판정을 나눠 적는다
DIFFS = ["값 상이", "한쪽 추가 기재", "표현 차이", "번호 상이"]
VERDICTS = ["실제 충돌", "확인 불가", "상호 보완", "표현 차이", "번호 상이(충돌 아님)"]
MISMATCH = "매칭 오류(비교 제외)"
REVIEW_COLS = ["platform_diff", "platform_diff_note", "official_check", "official_finding", "official_source", "official_checked_at",
               "final_verdict", "judged_by"]


def verdict_summary(cases: pd.DataFrame, n_same: int) -> pd.DataFrame:
    """사람 확정 동일 쌍의 판정 건수(항목별, any 행은 쌍 단위). 수집 당시 차이와 최종 판정을 함께 센다."""
    if "final_verdict" not in cases:
        return pd.DataFrame()
    pairs = lambda h: h[["servId", "gov_id"]].drop_duplicates().shape[0]
    rows = []
    for item in CONTENT_ITEMS + ["any"]:
        h = cases if item == "any" else cases[cases["item"].eq(item)]
        cnt = (lambda x: pairs(x)) if item == "any" else len
        row = {"item": item, "n_human_same_pairs": n_same, "n_flagged": cnt(h)}
        for d in DIFFS:
            row[f"diff_{d}"] = cnt(h[h["platform_diff"].eq(d)])
        for v in VERDICTS:
            row[f"n_{v}"] = cnt(h[h["final_verdict"].eq(v)])
        k = row["diff_값 상이"]
        row["share_value_diff_of_same"] = k / n_same if n_same else np.nan
        row["value_diff_ci_low"], row["value_diff_ci_high"] = QF.wilson(k, n_same)
        rows.append(row)
    t = pd.DataFrame(rows)
    t["note"] = ("분모 = 사람 확정 동일 쌍(사후 정정 반영). diff_* = 수집 당시 플랫폼 간 차이 유형, n_* = 공식 기준 확인을 거친 최종 판정. "
                 "규칙이 보완 또는 상이 후보로 표시한 행만 검수해 건수는 하한. 표본이 작아 전체 비율로 일반화하지 않음")
    return t


# ================================================================== 묶음
def build_all(root=ROOT, final_tabs: dict | None = None) -> dict:
    L = load(root)
    ft = final_tabs or QF.build_all(root)
    conf = confirmed_pairs(ft["adjudication_match"], root)
    pops = populations(L["match"], conf, L["youth"])
    summ, dist = completeness_summary(L["items"], pops)
    gi, gs = integration_gain(L["items"], pops)
    cov, ya, yi = youth_tables(L["items"], L["match"], L["youth"])
    m, g, sf, gf = V.load_inputs()
    cc = content_compare(L["match"], sf, gf)
    review = V.read(REVIEW) if REVIEW.exists() else None
    cases = content_cases(cc, pops, sf, gf, m, g, review)
    out = {"svc_completeness_summary": summ, "svc_short_distribution": dist, "svc_item_status": item_status(L["items"], pops),
           "svc_gain_by_item": gi, "svc_gain_by_service": gs, "youth_match_coverage": cov, "youth_area_completeness": ya,
           "youth_area_item_short": yi, "content_compare_summary": content_summary(cc, pops), "content_cases_human_same": cases}
    vs = verdict_summary(cases, len(pops["human_same"]))
    if len(vs):
        out["content_verdict_summary"] = vs
    out["match_accuracy_post_review"] = match_accuracy_post_review(ft, conf, root)
    out["_populations"] = pd.DataFrame([{"population": k, "label": POP_KO[k], "n_pairs": len(v)} for k, v in pops.items()])
    return out


def write_all(root=ROOT) -> dict:
    tabs = build_all(root)
    out = root / OUTDIR
    out.mkdir(parents=True, exist_ok=True)
    for k, v in tabs.items():
        v.to_csv(out / f"{k.lstrip('_') if k != '_populations' else 'svc_populations'}.csv", index=False, encoding="utf-8-sig", lineterminator="\n")
    return tabs


if __name__ == "__main__":
    T = write_all()
    s = T["svc_completeness_summary"]
    print(s[(s.population == "AB_nonshared") & (s.variant == "main")][["scope", "source", "n_pairs", "mean_explicit", "share_complete"]].round(3).to_string(index=False))
