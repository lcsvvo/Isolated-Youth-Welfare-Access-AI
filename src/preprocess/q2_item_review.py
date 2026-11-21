"""Q2 정보 항목 규칙 v3.1 사람 금표준(gold) 표본: 추출·배정·검수 파일 생성, 판정 집계, v3.1 성능 요약.

원칙
  - 금표준 판정은 사람만 한다. AI 판정(ai_ 열)을 만들거나 읽지 않는다.
  - 독립성: 파일럿(v3_pilot_sample)·검증(v3_validation_sample)·item_rule_sample에 한 번이라도 나온 (servId, gov_id) 쌍은
    항목과 상관없이 모두 모집단에서 뺀다. 기존 검수 시트(item_review_sheet.csv)는 개발 세트이고 금표준이 아니다.
  - 모집단: A·B 등급, 보조금24 공유 아님, 본문(text) 범위의 v3.1 판정 단위(쌍, side, 항목). 거주도 뽑되 core_metric=False 로 표시.
  - 표본: 항목 7 x side 2마다 10단위(v3.1 수준·메타 층별, 층당 최대 3), 시드 20261108. 층별 모집단 크기 N_h 와 표본 크기 n_h 는
    gold_strata.csv 에 저장해 나중에 층 가중 성능을 낼 때 쓴다.
  - 중복 검수 30단위(항목당 약 4개)를 R1~R4에 나눠 일치도를 본다. 검수자 파일에는 중복 여부·층·v3/v3.1 결과·ai_ 판정을 싣지 않는다.

폴더 구분
  - logs/validation/q2_item_rules/human/   사람에게 주는 파일만 (item_gold_index.csv, item_review_R1~R4.csv, gov24_screen_check.csv)
  - logs/validation/q2_item_rules/         중복 배정(item_review_assignment.csv), 층·v3.1 결과(item_gold_sample.csv, gold_strata.csv),
                                           조정·최종·일치도·성능 파일

일치도 해석 (aggregate_items)
  - 중복 30단위는 항목별 kappa 를 안정적으로 낼 만큼 많지 않다. 항목별로는 중복 단위 수와 일치 건수만 기술하고 항목별 kappa 는 내지 않는다.
  - 전체 kappa 와 일치율(항목 단위 percentile bootstrap 95% 신뢰구간)은 양쪽 판정이 있는 중복 단위가 20건 이상일 때만 낸다.
  - 불일치는 조정 파일에 두 원 판정을 그대로 두고, h_final·final_method·final_by 셋이 모두 채워졌을 때만 최종으로 인정한다.
  - 성능(v3.1 대비 사람 최종 판정)은 최종 판정이 있는 단위만 쓰며, 항목 x side 의 모든 단위가 판정되기 전에는 '미완료'다.

실행: python -m src.preprocess.q2_item_review build | aggregate
"""
from __future__ import annotations

import json
import sys
from itertools import combinations

import numpy as np
import pandas as pd

from . import q2_codebook as CB
from . import q2_items_v3 as V
from . import q2_items_v31 as V31
from .q2_review import boot_ci, kappa, pct_agree

RULES_DIR = V.RULES_DIR
HUMAN = RULES_DIR / "human"
SEED_GOLD = 20261108
SEED_ASSIGN = 20261109          # 배정·중복 선택 (고정)
N_PER_CELL = 10
N_DOUBLE = 30
REVIEWERS = ["R1", "R2", "R3", "R4"]
SIDES = ("ssis", "gov")
LEVELS = ["명시", "부분", "미명시"]
META_H = ["해당없음", "판단불가"]
LABELS = LEVELS + META_H
FINAL_METHOD = ["제3자", "합의"]
MIN_BOTH = 20
BOOT_SEED = 20261110

P_SAMPLE = RULES_DIR / "item_gold_sample.csv"
P_STRATA = RULES_DIR / "gold_strata.csv"
P_ASSIGN = RULES_DIR / "item_review_assignment.csv"
P_REPORT = RULES_DIR / "gold_build_report.json"
P_INDEX = HUMAN / "item_gold_index.csv"
P_ADJ = RULES_DIR / "item_review_adjudication.csv"
P_FINAL = RULES_DIR / "item_gold_final.csv"
P_AGREE = RULES_DIR / "item_review_agreement.csv"
P_AGREE_ITEM = RULES_DIR / "item_review_agreement_by_item.csv"
P_CONF = RULES_DIR / "item_gold_confusion.csv"
P_PERF = RULES_DIR / "item_gold_performance.csv"
P_STRATUM_ACC = RULES_DIR / "item_gold_stratum_accuracy.csv"
P_PROBLEMS = RULES_DIR / "item_review_problems.csv"

REVIEW_COLS = ["sample_id", "item", "side", "designated_fields", "designated_text", "apply", "target", "criteria",
               "benefit", "summary", "codebook_definition"] + V.HUMAN_COLS
ADJ_FINAL = ["h_final", "final_method", "final_by", "final_note", "final_at"]
ADJ_COLS = ["sample_id", "item", "side", "reviewer_a", "label_a", "evidence_a", "reviewer_b", "label_b", "evidence_b"] + ADJ_FINAL
ASSIGN_COLS = ["sample_id", "reviewer_primary", "reviewer_second"]
AGREE_NOTE = ("중복 30단위는 항목별 kappa 를 안정적으로 낼 만큼 많지 않아 항목별로는 건수만 기술한다. 전체 kappa 는 양쪽 판정 20건 이상일 때만 "
              "낸다. 범주가 한쪽으로 쏠리면 일치율이 높아도 kappa 가 낮을 수 있어 일치율을 함께 본다.")

read = V.read


def r_path(r: str, human=HUMAN):
    return human / f"item_review_{r}.csv"


# ------------------------------------------------------------------ 정의 문구 (v3.1 조건 포함)
def definition31(item: str) -> str:
    """검수 시트용 정의: v3 요약 + 코드북 v3.1 경계 결정(해당 항목만, 짧게)."""
    rows = [r for r in CB.RULES_31 if r[0] == item]
    add = " / ".join(f"{r[1]}: {r[2]}" + (f" (제외 예: {r[4]})" if r[4] else "") for r in rows)
    return V.DEF_SHORT[item] + (f" [v3.1 결정] {add}" if add else "")


# ------------------------------------------------------------------ 독립성: 이미 쓴 쌍
def reviewed_pairs(rules_dir=RULES_DIR) -> pd.DataFrame:
    """파일럿·검증·item_rule_sample 에 나온 (servId, gov_id) 쌍과 출처 파일."""
    parts = []
    for name in ("v3_pilot_sample", "v3_validation_sample", "item_rule_sample"):
        p = rules_dir / f"{name}.csv"
        if p.exists():
            s = read(p)[["servId", "gov_id", "item", "side"]]
            parts.append(s.assign(file=name))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["servId", "gov_id", "item", "side", "file"])


def pilot_validation_overlap(rules_dir=RULES_DIR) -> dict:
    """파일럿 대 검증: (쌍, side, 항목) 단위 중복 수, 서로 다른 항목으로 양쪽에 나온 쌍 수."""
    a, b = (read(rules_dir / f"v3_{n}_sample.csv") for n in ("pilot", "validation"))
    ua, ub = V._unit_set(a), V._unit_set(b)
    pa, pb = set(zip(a["servId"], a["gov_id"])), set(zip(b["servId"], b["gov_id"]))
    both = pa & pb
    return {"units_pilot": len(ua), "units_validation": len(ub), "unit_overlap": len(ua & ub),
            "pairs_pilot": len(pa), "pairs_validation": len(pb), "pairs_in_both": len(both)}


def gold_population(df31: pd.DataFrame, rules_dir=RULES_DIR) -> tuple[pd.DataFrame, dict]:
    pop = V.population(df31)
    used = reviewed_pairs(rules_dir)
    ex = set(zip(used["servId"], used["gov_id"]))
    keep = pd.Series([(a, b) not in ex for a, b in zip(pop["servId"], pop["gov_id"])], index=pop.index)
    pairs_before = set(zip(pop["servId"], pop["gov_id"]))
    after = pop[keep]
    rep = {"pairs_excluded_all_sources": len(ex),
           "pairs_excluded_by_file": {n: int(used.loc[used["file"].eq(n)].drop_duplicates(["servId", "gov_id"]).shape[0])
                                      for n in used["file"].unique()},
           "pop_pairs_before": len(pairs_before), "pop_pairs_excluded": len(pairs_before & ex),
           "pop_pairs_after": len(set(zip(after["servId"], after["gov_id"]))),
           "pop_units_before": int(len(pop)), "pop_units_after": int(len(after))}
    return after, rep


# ------------------------------------------------------------------ 추출
def draw_gold(pop: pd.DataFrame, seed: int = SEED_GOLD) -> pd.DataFrame:
    """항목 x side 마다 V.draw(층별 번갈아, 층당 3). sample_id 는 블록 안에서 섞은 뒤 매겨 층 순서가 드러나지 않게 한다."""
    smp = V.draw(pop, seed, set(), per_item_side=N_PER_CELL)
    rng = np.random.default_rng(seed + 1)
    parts = []
    for item in V.ITEMS:
        for side in SIDES:
            b = smp[smp["item"].eq(item) & smp["side"].eq(side)]
            b = b.iloc[rng.permutation(len(b))].copy()
            b["sample_id"] = [f"G-{item}-{side}-{k:02d}" for k in range(1, len(b) + 1)]
            parts.append(b)
    out = pd.concat(parts, ignore_index=True)
    out["core_metric"] = out["item"].isin(CB.CORE_ITEMS)
    return out


def strata_table(pop: pd.DataFrame, smp: pd.DataFrame) -> pd.DataFrame:
    N = pop.groupby(["item", "side", "stratum"]).size().rename("N_h")
    n = smp.groupby(["item", "side", "stratum"]).size().rename("n_h")
    t = pd.concat([N, n], axis=1).fillna(0).astype(int).reset_index()
    t["item"] = pd.Categorical(t["item"], V.ITEMS)
    t["core_metric"] = t["item"].astype(str).isin(CB.CORE_ITEMS)
    return t.sort_values(["item", "side", "stratum"]).reset_index(drop=True).assign(item=lambda x: x["item"].astype(str))


# ------------------------------------------------------------------ 배정
def assign_primary(smp: pd.DataFrame, seed: int = SEED_ASSIGN) -> pd.Series:
    """항목 x side 블록마다 섞어 검수자에게 돌아가며 배정. 시작 위치를 블록 사이에 이어 가서 검수자별 합계를 맞춘다."""
    rng = np.random.default_rng(seed)
    prim, offset = pd.Series("", index=smp.index), 0
    for item in V.ITEMS:
        for side in SIDES:
            idx = smp.index[smp["item"].eq(item) & smp["side"].eq(side)].to_numpy()
            idx = idx[rng.permutation(len(idx))]
            for i, p in enumerate(idx):
                prim[p] = REVIEWERS[(offset + i) % len(REVIEWERS)]
            offset += len(idx)
    return prim


def pick_doubles(smp: pd.DataFrame, prim: pd.Series, n: int = N_DOUBLE, seed: int = SEED_ASSIGN + 1) -> list:
    """항목마다 n/7개(나머지는 무작위 항목에 1개씩 더), 항목 안에서 ssis·gov 반반. 1차 검수자의 중복 건수가 고르게 퍼지게 고른다."""
    rng = np.random.default_rng(seed)
    base, extra = divmod(n, len(V.ITEMS))
    plus = set(rng.choice(V.ITEMS, size=extra, replace=False)) if extra else set()
    cnt, chosen = {r: 0 for r in REVIEWERS}, []
    for item in rng.permutation(V.ITEMS):
        q = base + (item in plus)
        sides = [SIDES[i % 2] for i in range(q)] if q % 2 == 0 else [SIDES[i % 2] for i in range(q - 1)] + [SIDES[int(rng.integers(2))]]
        for side in sides:
            cand = [p for p in smp.index[smp["item"].eq(item) & smp["side"].eq(side)] if p not in chosen]
            cand = [cand[i] for i in rng.permutation(len(cand))]
            if not cand:
                continue
            p = min(cand, key=lambda x: cnt[prim[x]])
            chosen.append(p)
            cnt[prim[p]] += 1
    return chosen


def assign_second(prim: pd.Series, doubles: list) -> pd.Series:
    """2차 검수자: 1차와 달라야 하고, 2차 부담 편차를 검수자 쌍 조합 편차보다 우선해 줄인다."""
    sec = pd.Series("", index=prim.index)
    load, pair = {r: 0 for r in REVIEWERS}, {}
    for p in doubles:
        cand = [r for r in REVIEWERS if r != prim[p]]
        r2 = min(cand, key=lambda r: (load[r], pair.get(frozenset((prim[p], r)), 0), REVIEWERS.index(r)))
        sec[p] = r2
        load[r2] += 1
        pair[frozenset((prim[p], r2))] = pair.get(frozenset((prim[p], r2)), 0) + 1

    def cost() -> int:
        ld = pd.Series([sec[p] for p in doubles]).value_counts().reindex(REVIEWERS, fill_value=0)
        pr = pd.Series([frozenset((prim[p], sec[p])) for p in doubles]).value_counts()
        pr = pr.reindex([frozenset(c) for c in combinations(REVIEWERS, 2)], fill_value=0)
        return 100 * int((ld ** 2).sum()) + int((pr ** 2).sum())

    best, improved = cost(), True
    while improved:
        improved = False
        for p in doubles:
            cur = sec[p]
            for r in REVIEWERS:
                if r in (cur, prim[p]):
                    continue
                sec[p] = r
                c = cost()
                if c < best:
                    best, cur, improved = c, r, True
                else:
                    sec[p] = cur
    return sec


def make_assignment(smp: pd.DataFrame, seed: int = SEED_ASSIGN) -> pd.DataFrame:
    prim = assign_primary(smp, seed)
    doubles = pick_doubles(smp, prim, N_DOUBLE, seed + 1)
    sec = assign_second(prim, doubles)
    return pd.DataFrame({"sample_id": smp["sample_id"].to_numpy(), "reviewer_primary": prim.to_numpy(),
                         "reviewer_second": sec.to_numpy()})[ASSIGN_COLS]


# ------------------------------------------------------------------ 검수자 파일
def review_frame(smp: pd.DataFrame, sf, gf) -> pd.DataFrame:
    """사람용 전체 열. 본문은 자르지 않는다(NFKC 원문). v3·v3.1·ai_ 열과 층·중복 여부는 넣지 않는다."""
    rows = []
    for r in smp.itertuples(index=False):
        F = V.side_fields(sf, gf, r.servId, r.gov_id, r.side)
        d = {"sample_id": r.sample_id, "item": r.item, "side": r.side,
             "designated_fields": "+".join(CB.FIELDS[r.item]["text"]),
             "designated_text": "\n".join(f"[{k}] {F.get(k, '')}" for k in CB.FIELDS[r.item]["text"])}
        for k in V.SHEET_TEXT:
            d[k] = F.get(k, "")
        d["codebook_definition"] = definition31(r.item)
        rows.append(d)
    out = pd.DataFrame(rows)
    for c in V.HUMAN_COLS:
        out[c] = ""
    return out[REVIEW_COLS]


def guard_human(human=HUMAN) -> None:
    """검수자 파일에 판정(h_ 열, reviewer, reviewed_at)이 하나라도 있으면 아무것도 덮어쓰지 않는다."""
    for r in REVIEWERS:
        p = r_path(r, human)
        if p.exists():
            prev = read(p)
            cols = [c for c in prev.columns if c.startswith("h_") or c in ("reviewer", "reviewed_at")]
            if cols and prev[cols].ne("").any().any():
                raise ValueError(f"판정값이 있어 덮어쓰지 않습니다: {p}")


def build(rules_dir=RULES_DIR, df31: pd.DataFrame | None = None, seed: int = SEED_GOLD) -> dict:
    human = rules_dir / "human"
    guard_human(human)
    df31 = read(V31.OUT) if df31 is None else df31
    pop, rep = gold_population(df31, rules_dir)
    smp = draw_gold(pop, seed)
    strata = strata_table(pop, smp)
    asg = make_assignment(smp)
    _, _, sf, gf = V.load_inputs()
    full = review_frame(smp, sf, gf)
    human.mkdir(parents=True, exist_ok=True)

    keep = ["sample_id", "item", "side", "servId", "gov_id", "tier", "source", "stratum", "core_metric", "level", "meta", "rule_id", "evidence"]
    gs = smp[keep].rename(columns={"level": "v31_level", "meta": "v31_meta", "rule_id": "v31_rule_id", "evidence": "v31_evidence"})
    gs.to_csv(rules_dir / P_SAMPLE.name, index=False, encoding="utf-8-sig")
    strata.to_csv(rules_dir / P_STRATA.name, index=False, encoding="utf-8-sig")
    asg.to_csv(rules_dir / P_ASSIGN.name, index=False, encoding="utf-8-sig")
    smp[["sample_id", "item", "side", "servId", "gov_id"]].sort_values("sample_id").to_csv(
        human / P_INDEX.name, index=False, encoding="utf-8-sig")
    for k, r in enumerate(REVIEWERS):
        ids = asg.loc[asg["reviewer_primary"].eq(r) | asg["reviewer_second"].eq(r), "sample_id"]
        full[full["sample_id"].isin(ids)].sample(frac=1, random_state=SEED_ASSIGN + 10 + k).to_csv(
            r_path(r, human), index=False, encoding="utf-8-sig")

    dbl = asg[asg["reviewer_second"].ne("")]
    rep.update({
        "seed_gold": seed, "units": int(len(smp)),
        "units_by_item_side": {f"{i}|{s}": int(n) for (i, s), n in smp.groupby(["item", "side"]).size().items()},
        "primary_by_reviewer": {r: int(asg["reviewer_primary"].eq(r).sum()) for r in REVIEWERS},
        "second_by_reviewer": {r: int(dbl["reviewer_second"].eq(r).sum()) for r in REVIEWERS},
        "doubles": int(len(dbl)),
        "doubles_by_item": {i: int(n) for i, n in smp.set_index("sample_id").loc[dbl["sample_id"], "item"].value_counts().sort_index().items()},
        "doubles_by_pair": {"-".join(sorted(k)): int(n) for k, n in
                            pd.Series([frozenset(x) for x in zip(dbl["reviewer_primary"], dbl["reviewer_second"])]).value_counts().items()},
        "file_rows": {r: int((asg["reviewer_primary"].eq(r) | asg["reviewer_second"].eq(r)).sum()) for r in REVIEWERS}})
    rep["pilot_vs_validation"] = pilot_validation_overlap(rules_dir)
    (rules_dir / P_REPORT.name).write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    return rep


# ------------------------------------------------------------------ 판정 읽기·검증
def row_label(r) -> str:
    """한 행의 판정 라벨. h_level 과 h_meta 중 정확히 하나가 허용값으로 채워졌을 때만."""
    lv, mt = r["h_level"].strip(), r["h_meta"].strip()
    if lv and mt:
        return ""
    if lv in LEVELS:
        return lv
    return mt if mt in META_H else ""


def validate_rows(rows: pd.DataFrame, reviewer: str, allowed: set | None = None) -> pd.DataFrame:
    bad = []
    for r in rows.to_dict("records"):
        why = []
        lv, mt = r["h_level"].strip(), r["h_meta"].strip()
        if lv not in LEVELS + [""]:
            why.append(f"h_level={lv!r}")
        if mt not in META_H + [""]:
            why.append(f"h_meta={mt!r}")
        if lv and mt:
            why.append("h_level 과 h_meta 동시 입력")
        if allowed is not None and r["sample_id"] not in allowed:
            why.append("배정되지 않은 단위")
        if why:
            bad.append({"reviewer_file": reviewer, "sample_id": r["sample_id"], "problem": "; ".join(why)})
    return pd.DataFrame(bad, columns=["reviewer_file", "sample_id", "problem"])


def load_judgements(asg: pd.DataFrame, human=HUMAN) -> tuple[dict, pd.DataFrame]:
    """{(검수자, sample_id): (라벨, 증거)} 와 오류 행. 오류 행·빈 행은 판정 없음."""
    jd, bad = {}, []
    for r in REVIEWERS:
        p = r_path(r, human)
        if not p.exists():
            continue
        f = read(p)
        allowed = set(asg.loc[asg["reviewer_primary"].eq(r) | asg["reviewer_second"].eq(r), "sample_id"])
        bad.append(validate_rows(f, r, allowed))
        for row in f.to_dict("records"):
            lab = row_label(row) if row["sample_id"] in allowed else ""
            if lab:
                jd[(r, row["sample_id"])] = (lab, row["h_evidence"].strip())
    return jd, pd.concat(bad, ignore_index=True) if bad else pd.DataFrame(columns=["reviewer_file", "sample_id", "problem"])


def both_table(asg: pd.DataFrame, items: pd.Series, jd: dict) -> pd.DataFrame:
    """중복 단위의 1·2차 판정. 한쪽이라도 없으면 j1 또는 j2 가 빈 문자열."""
    d = asg[asg["reviewer_second"].ne("")].copy()
    d["item"] = d["sample_id"].map(items)
    d["j1"] = [jd.get((r, s), ("", ""))[0] for r, s in zip(d["reviewer_primary"], d["sample_id"])]
    d["j2"] = [jd.get((r, s), ("", ""))[0] for r, s in zip(d["reviewer_second"], d["sample_id"])]
    return d


# ------------------------------------------------------------------ 조정
def validate_adjudication(adj: pd.DataFrame) -> pd.DataFrame:
    bad = []
    for r in adj.to_dict("records"):
        why = []
        if r["h_final"].strip() not in LABELS + [""]:
            why.append(f"h_final={r['h_final']!r}")
        if r["final_method"].strip() not in FINAL_METHOD + [""]:
            why.append(f"final_method={r['final_method']!r}")
        if r["h_final"].strip() in LABELS and not r["final_method"].strip():
            why.append("final_method 없음(최종값 미인정)")
        if r["h_final"].strip() in LABELS and not r["final_by"].strip():
            why.append("final_by 없음(최종값 미인정)")
        if why:
            bad.append({"reviewer_file": "adjudication", "sample_id": r["sample_id"], "problem": "; ".join(why)})
    return pd.DataFrame(bad, columns=["reviewer_file", "sample_id", "problem"])


def final_accepted(r) -> bool:
    return (r["h_final"].strip() in LABELS and r["final_method"].strip() in FINAL_METHOD and bool(r["final_by"].strip()))


def update_adjudication(current: pd.DataFrame, path=P_ADJ) -> pd.DataFrame:
    """조정 파일 갱신. 조정 칸이 하나라도 채워진 행은 보존하고, 미조정 불일치는 원 판정을 새로 쓴다. 더는 불일치가 아닌 빈 행은 뺀다."""
    prev = read(path) if path.exists() else pd.DataFrame(columns=ADJ_COLS)
    for c in ADJ_COLS:
        if c not in prev.columns:
            prev[c] = ""
    prev = prev[ADJ_COLS]
    filled = prev[prev[ADJ_FINAL].apply(lambda s: s.str.strip()).ne("").any(axis=1)]
    new = current[~current["sample_id"].isin(filled["sample_id"])]
    out = pd.concat([filled, new], ignore_index=True).sort_values("sample_id", kind="stable").reset_index(drop=True)
    out.to_csv(path, index=False, encoding="utf-8-sig")
    return out


# ------------------------------------------------------------------ 집계
def weighted_accuracy(strata: pd.DataFrame, fin: pd.DataFrame) -> pd.DataFrame:
    """항목 x side x 층별 일치 수. fin: sample_id 별 v31 라벨(v31)과 사람 최종 라벨(final_label, 비면 미판정), item, side, stratum."""
    j = fin[fin["final_label"].ne("")]
    g = j.groupby(["item", "side", "stratum"]).agg(n_final=("final_label", "size"), n_agree=("agree", "sum")).reset_index()
    t = strata.merge(g, on=["item", "side", "stratum"], how="left").fillna({"n_final": 0, "n_agree": 0})
    t[["n_final", "n_agree"]] = t[["n_final", "n_agree"]].astype(int)
    t["accuracy"] = np.where(t["n_final"].gt(0), t["n_agree"] / t["n_final"].replace(0, np.nan), np.nan)
    return t


def performance_table(sa: pd.DataFrame) -> pd.DataFrame:
    """층 가중 정확도: sum_h N_h * acc_h / sum_h N_h. 모든 표본 단위가 판정되기 전(n_final < n_h)에는 미완료."""
    rows = []
    for item in V.ITEMS:
        for side in list(SIDES) + ["all"]:
            x = sa[sa["item"].eq(item)] if side == "all" else sa[sa["item"].eq(item) & sa["side"].eq(side)]
            if x.empty:
                continue
            done = bool((x["n_final"] == x["n_h"]).all() and x["n_h"].gt(0).all())
            acc = float((x["N_h"] * x["accuracy"]).sum() / x["N_h"].sum()) if done else np.nan
            rows.append({"item": item, "side": side, "core_metric": item in CB.CORE_ITEMS, "n_units": int(x["n_h"].sum()),
                         "n_final": int(x["n_final"].sum()), "status": "완료" if done else "미완료",
                         "weighted_accuracy": round(acc, 4) if done else np.nan})
    return pd.DataFrame(rows)


def aggregate_items(rules_dir=RULES_DIR, write: bool = True) -> dict:
    """판정 집계. 판정이 없으면 수치를 만들지 않고 상태만 남긴다.

    일치도: 중복 30단위는 항목별 kappa 를 안정적으로 낼 만큼 많지 않다. 항목별로는 중복 단위 수(n_double)·양쪽 판정 수(n_both)·일치 건수
    (n_agree)만 기술하고 항목별 kappa 는 내지 않는다. 전체 kappa 와 일치율은 양쪽 판정 단위가 MIN_BOTH(20) 이상일 때만 낸다.
    """
    human = rules_dir / "human"
    asg = read(rules_dir / P_ASSIGN.name)
    gs = read(rules_dir / P_SAMPLE.name)
    strata = read(rules_dir / P_STRATA.name)
    strata[["N_h", "n_h"]] = strata[["N_h", "n_h"]].astype(int)
    jd, bad = load_judgements(asg, human)
    both = both_table(asg, gs.set_index("sample_id")["item"], jd)
    judged_both = both[both["j1"].ne("") & both["j2"].ne("")]

    # 일치도
    by_item = []
    for item in V.ITEMS:
        b, jb = both[both["item"].eq(item)], judged_both[judged_both["item"].eq(item)]
        by_item.append({"item": item, "n_double": len(b), "n_both": len(jb), "n_agree": int((jb["j1"] == jb["j2"]).sum())})
    by_item = pd.DataFrame(by_item)
    n_both = len(judged_both)
    ok = n_both >= MIN_BOTH
    k = kappa(judged_both["j1"], judged_both["j2"], LABELS) if ok else np.nan
    pa = pct_agree(judged_both["j1"], judged_both["j2"]) if ok else np.nan
    rows = []
    for metric, est, fn in (("kappa", k, lambda a, b: kappa(a, b, LABELS)), ("pct_agree", pa, pct_agree)):
        lo, hi = boot_ci(judged_both["j1"], judged_both["j2"], fn, seed=BOOT_SEED) if ok and not np.isnan(est) else (np.nan, np.nan)
        rows.append({"metric": metric, "estimate": round(est, 4) if not np.isnan(est) else np.nan,
                     "ci_low": round(lo, 4) if not np.isnan(lo) else np.nan, "ci_high": round(hi, 4) if not np.isnan(hi) else np.nan,
                     "n_double": len(both), "n_both": n_both, "status": "산출" if ok and not np.isnan(est) else "미완료",
                     "interpretation_note": AGREE_NOTE})
    agree = pd.DataFrame(rows)

    # 조정
    dis = judged_both[judged_both["j1"] != judged_both["j2"]]
    cur = pd.DataFrame([{"sample_id": r.sample_id, "item": r.item, "side": gs.set_index("sample_id").at[r.sample_id, "side"],
                         "reviewer_a": r.reviewer_primary, "label_a": r.j1, "evidence_a": jd[(r.reviewer_primary, r.sample_id)][1],
                         "reviewer_b": r.reviewer_second, "label_b": r.j2, "evidence_b": jd[(r.reviewer_second, r.sample_id)][1],
                         **{c: "" for c in ADJ_FINAL}} for r in dis.itertuples()], columns=ADJ_COLS)
    adj = update_adjudication(cur, rules_dir / P_ADJ.name) if write else cur
    bad = pd.concat([bad, validate_adjudication(adj)], ignore_index=True)
    adj_ok = {r["sample_id"]: r["h_final"].strip() for r in adj.to_dict("records") if final_accepted(r)}

    # 최종 판정
    fin = []
    for r in asg.itertuples():
        j1 = jd.get((r.reviewer_primary, r.sample_id), ("", ""))[0]
        if not r.reviewer_second:
            lab, src = (j1, "단독") if j1 else ("", "미판정")
        else:
            j2 = jd.get((r.reviewer_second, r.sample_id), ("", ""))[0]
            if not (j1 and j2):
                lab, src = "", "미판정"
            elif j1 == j2:
                lab, src = j1, "일치"
            elif r.sample_id in adj_ok:
                lab, src = adj_ok[r.sample_id], "조정"
            else:
                lab, src = "", "미해결"
        fin.append({"sample_id": r.sample_id, "final_label": lab, "final_source": src})
    fin = gs[["sample_id", "item", "side", "servId", "gov_id", "stratum", "core_metric", "v31_level", "v31_meta"]].merge(
        pd.DataFrame(fin), on="sample_id", validate="1:1")
    fin["v31"] = fin["v31_meta"].where(fin["v31_meta"].ne(""), fin["v31_level"])
    fin["agree"] = fin["final_label"].ne("") & fin["final_label"].eq(fin["v31"])
    out_final = fin.assign(final_level=np.where(fin["final_label"].isin(LEVELS), fin["final_label"], ""),
                           final_meta=np.where(fin["final_label"].isin(META_H), fin["final_label"], ""))[
        ["sample_id", "item", "side", "servId", "gov_id", "final_label", "final_level", "final_meta", "final_source"]]

    # 성능 (최종 판정이 있는 단위만)
    j = fin[fin["final_label"].ne("")]
    conf = j.groupby(["item", "v31", "final_label"]).size().rename("n").reset_index().rename(columns={"v31": "v31_label", "final_label": "human_label"})
    sa = weighted_accuracy(strata, fin)
    perf = performance_table(sa)
    res = {"judged_units": len(jd), "n_final": int(len(j)), "n_units": int(len(fin)), "agreement": agree, "agreement_by_item": by_item,
           "adjudication": adj, "final": out_final, "confusion": conf, "stratum_accuracy": sa, "performance": perf, "problems": bad,
           "complete": bool(len(j) == len(fin))}
    if write:
        out_final.to_csv(rules_dir / P_FINAL.name, index=False, encoding="utf-8-sig")
        agree.to_csv(rules_dir / P_AGREE.name, index=False, encoding="utf-8-sig")
        by_item.to_csv(rules_dir / P_AGREE_ITEM.name, index=False, encoding="utf-8-sig")
        conf.to_csv(rules_dir / P_CONF.name, index=False, encoding="utf-8-sig")
        perf.to_csv(rules_dir / P_PERF.name, index=False, encoding="utf-8-sig")
        sa.to_csv(rules_dir / P_STRATUM_ACC.name, index=False, encoding="utf-8-sig")
        bad.to_csv(rules_dir / P_PROBLEMS.name, index=False, encoding="utf-8-sig")
    return res


def report(res: dict) -> None:
    if res["judged_units"] == 0:
        print("사람 판정 없음: 일치도·성능 집계 보류 (미완료)")
        return
    print(f"판정 {res['judged_units']}건, 최종 판정 단위 {res['n_final']}/{res['n_units']}" + ("" if res["complete"] else " (미완료)"))
    print(res["agreement_by_item"].to_string(index=False))
    print(res["agreement"][["metric", "estimate", "ci_low", "ci_high", "n_both", "status"]].to_string(index=False))
    if len(res["problems"]):
        print(f"입력 오류 {len(res['problems'])}건 → {P_PROBLEMS.relative_to(P_PROBLEMS.parents[3])}")
    print(res["performance"].to_string(index=False))


def main(cmd: str) -> None:
    if cmd == "build":
        rep = build()
        print(f"금표준 {rep['units']}단위, 제외한 기존 표본 쌍 {rep['pairs_excluded_all_sources']}개 "
              f"(모집단 쌍 {rep['pop_pairs_before']} → {rep['pop_pairs_after']}), 중복 {rep['doubles']}단위")
    elif cmd == "aggregate":
        report(aggregate_items())
    else:
        raise SystemExit(f"알 수 없는 명령: {cmd}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "aggregate")
