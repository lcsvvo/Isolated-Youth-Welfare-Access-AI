"""STEP 02 사람 라벨 수집·평가.

라벨러는 차수별 시트를 채워 outputs/02_labeling/labels/<round>_<labeler>.csv로 저장한다.
값 규칙: h_q21_xx = 예/아니오/불명, h_youth_target = 예/아니오/확인필요, h_apply_barrier·h_use_outing = 0~3/N,
         h_job_relevance = 0/1/2/N, h_job_reason = 규칙누락/실제무관/정보부족/규칙정답.
'불명'·'확인필요'는 지표 계산의 분모에서 뺀다.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .sampling import Q21

YES, NO = "예", "아니오"


def load_labels(folder: Path) -> pd.DataFrame:
    files = sorted(folder.glob("*.csv"))
    if not files:
        return pd.DataFrame()
    df = pd.concat([pd.read_csv(f, dtype=str, encoding="utf-8-sig").assign(file=f.name) for f in files], ignore_index=True)
    return df[df["labeler"].fillna("").str.strip().ne("")]


def primary(labels: pd.DataFrame, round_name: str) -> pd.DataFrame:
    """서비스당 1행: 라벨러 이름순 첫 라벨. 이중 라벨 불일치는 agreement()로 따로 본다."""
    d = labels[labels["round"].eq(round_name)].sort_values(["servId", "labeler"])
    return d.drop_duplicates("servId")


def _rule_hit(key: pd.DataFrame, q: str) -> pd.Series:
    return key["need_area_q21"].fillna("N").str.split("|").map(lambda v: q in v)


def q1_precision(lab: pd.DataFrame, key: pd.DataFrame) -> pd.DataFrame:
    """규칙이 영역 k를 붙인 서비스 중 사람이 '예'로 본 비율 (S·P 표본, 비가중)."""
    d = key.merge(lab, on="servId", how="inner")
    rows = []
    for q in Q21:
        h = d.loc[_rule_hit(d, q), f"h_{q}"]
        h = h[h.isin([YES, NO])]
        rows.append({"need_area": q, "n_rule_positive_labeled": len(h), "precision": (h == YES).mean() if len(h) else np.nan})
    return pd.DataFrame(rows)


def q1_recall(lab: pd.DataFrame, key: pd.DataFrame) -> pd.DataFrame:
    """S 층화 표본 가중 추정: 사람 '예' 서비스 중 규칙이 잡은 비율과 놓친 추정 건수."""
    d = key[key["in_S"]].merge(lab, on="servId", how="inner")
    rows = []
    for q in Q21:
        v = d[d[f"h_{q}"].isin([YES, NO])]
        pos = v[f"h_{q}"].eq(YES)
        w = v["s_weight"].astype(float)
        hit = _rule_hit(v, q)
        est_pos, est_hit = float((w * pos).sum()), float((w * (pos & hit)).sum())
        rows.append({"need_area": q, "n_labeled_S": len(v), "n_human_positive_S": int(pos.sum()),
                     "est_positive_services": est_pos, "est_missed_by_rule": est_pos - est_hit,
                     "recall": est_hit / est_pos if est_pos else np.nan})
    return pd.DataFrame(rows)


def job_reason(lab: pd.DataFrame, key: pd.DataFrame) -> pd.DataFrame:
    """J 층별 사람 판정 분포와 층 가중치를 적용한 모집단 추정 건수."""
    d = key[key["in_J"]].merge(lab, on="servId", how="inner")
    if d.empty:
        return pd.DataFrame()
    t = d.groupby(["j_stratum", "h_job_reason", "h_job_relevance"], dropna=False).agg(n=("servId", "count"),
                                                                                       weight=("j_weight", "first")).reset_index()
    t["est_services"] = t["n"] * t["weight"].astype(float)
    return t


def agreement(labels: pd.DataFrame, round_name: str, cols: list[str]) -> pd.DataFrame:
    """이중 라벨 서비스의 컬럼별 일치율과 Cohen's kappa (라벨러 2명 기준)."""
    d = labels[labels["round"].eq(round_name)]
    multi = d.groupby("servId").filter(lambda g: g["labeler"].nunique() >= 2).sort_values(["servId", "labeler"])
    rows = []
    for c in cols:
        a = multi.groupby("servId")[c].agg(lambda s: tuple(s.iloc[:2]))
        pairs = [p for p in a if len(p) == 2 and all(isinstance(x, str) and x for x in p)]
        if not pairs:
            rows.append({"column": c, "n_pairs": 0, "agreement": np.nan, "kappa": np.nan})
            continue
        x, y = map(np.array, zip(*pairs))
        po = (x == y).mean()
        cats = set(x) | set(y)
        pe = sum((x == k).mean() * (y == k).mean() for k in cats)
        rows.append({"column": c, "n_pairs": len(pairs), "agreement": po, "kappa": (po - pe) / (1 - pe) if pe < 1 else np.nan})
    return pd.DataFrame(rows)


def ordinal_agreement(lab: pd.DataFrame, key: pd.DataFrame, rule_col: str, human_col: str, k: int = 4) -> dict:
    """순서형 라벨(0~k-1/N)의 규칙 대 사람 비교: 정확도·quadratic weighted kappa(둘 다 N이 아닌 건)와 N 비율."""
    d = key[["servId", rule_col]].merge(lab[["servId", human_col]], on="servId", how="inner")
    d = d[d[human_col].fillna("").ne("")]
    both = d[d[rule_col].ne("N") & d[human_col].ne("N")]
    out = {"attribute": rule_col, "n_labeled": len(d), "n_both_valued": len(both),
           "rule_N_rate": d[rule_col].eq("N").mean() if len(d) else np.nan,
           "human_N_rate": d[human_col].eq("N").mean() if len(d) else np.nan,
           "accuracy_incl_N": (d[rule_col] == d[human_col]).mean() if len(d) else np.nan,
           "accuracy_valued": (both[rule_col] == both[human_col]).mean() if len(both) else np.nan, "qwk": np.nan}
    if len(both):
        x, y = both[rule_col].astype(int).to_numpy(), both[human_col].astype(int).to_numpy()
        o = np.zeros((k, k))
        for i, j in zip(x, y):
            o[i, j] += 1
        w = np.array([[(i - j) ** 2 / (k - 1) ** 2 for j in range(k)] for i in range(k)])
        e = np.outer(o.sum(1), o.sum(0)) / o.sum()
        out["qwk"] = 1 - (w * o).sum() / (w * e).sum() if (w * e).sum() else np.nan
    return out


# ------------------------------------------------------------------ 2026-11-04 보완
def refresh_rule_values(key: pd.DataFrame, master: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """채점 키의 표본 설계 컬럼(층·가중치)은 그대로 두고, 규칙 값만 현재 마스터 값으로 바꾼다.
    규칙을 고친 뒤 같은 표본으로 개정 전후를 비교할 때 쓴다. 키 파일 자체는 수정하지 않는다."""
    keep = [c for c in key.columns if c not in cols or c == "servId"]
    return key[keep].merge(master[["servId", *cols]], on="servId", how="left")


def q1_multilabel_f1(lab: pd.DataFrame, key: pd.DataFrame, areas: list[str] | None = None) -> pd.DataFrame:
    """다중 라벨 영역의 영역별 precision·recall·F1과 micro·macro F1 (비가중, 사람 '예/아니오'만 사용).
    S·P를 합친 표본은 무작위 표본이 아니므로 이 값은 '표본 내' 지표다. 모집단 재현율은 q1_recall(S 가중)을 쓴다."""
    areas = areas or Q21
    d = key.merge(lab, on="servId", how="inner")
    rows, tp_all, fp_all, fn_all = [], 0, 0, 0
    for q in areas:
        v = d[d[f"h_{q}"].isin([YES, NO])]
        h, r = v[f"h_{q}"].eq(YES), _rule_hit(v, q)
        tp, fp, fn = int((h & r).sum()), int((~h & r).sum()), int((h & ~r).sum())
        p = tp / (tp + fp) if tp + fp else np.nan
        rc = tp / (tp + fn) if tp + fn else np.nan
        if tp + fp + fn == 0:
            f1 = np.nan                     # 사람·규칙 모두 양성이 없음: 정의되지 않음
        elif tp == 0:
            f1 = 0.0                        # 양성이 있었는데 하나도 맞히지 못함
        else:
            f1 = 2 * p * rc / (p + rc)
        rows.append({"need_area": q, "n_labeled": len(v), "tp": tp, "fp": fp, "fn": fn,
                     "precision": p, "recall": rc, "f1": f1})
        tp_all, fp_all, fn_all = tp_all + tp, fp_all + fp, fn_all + fn
    t = pd.DataFrame(rows)
    mp = tp_all / (tp_all + fp_all) if tp_all + fp_all else np.nan
    mr = tp_all / (tp_all + fn_all) if tp_all + fn_all else np.nan
    micro = (0.0 if tp_all == 0 else 2 * mp * mr / (mp + mr)) if tp_all + fp_all + fn_all else np.nan
    t.attrs["micro_f1"], t.attrs["macro_f1"] = micro, float(t["f1"].mean(skipna=True)) if t["f1"].notna().any() else np.nan
    return t


def youth_scope_confusion(lab: pd.DataFrame, key: pd.DataFrame, weight_col: str | None = "r_weight") -> pd.DataFrame:
    """규칙 youth_scope × 사람 h_youth_scope 혼동표(건수와 층 가중 추정 건수).
    핵심 오류는 사람 C(청년 비해당)를 규칙이 A·B로 본 경우와, 사람 A·B를 규칙이 C로 본 경우다."""
    d = key.merge(lab[["servId", "h_youth_scope"]], on="servId", how="inner")
    d = d[d["h_youth_scope"].isin(["A", "B", "U", "C"])]
    w = d[weight_col].astype(float) if weight_col and weight_col in d else pd.Series(1.0, index=d.index)
    t = d.assign(w=w).groupby(["youth_scope", "h_youth_scope"]).agg(n=("servId", "count"), est=("w", "sum")).reset_index()
    return t
