"""Q1 본 분석: 수요(2023 실태조사 Q21 지원 필요도) 쪽 계산과 입력 검증.

질문: 은둔 청년은 어떤 영역의 지원을 얼마나 필요로 하고, 고립 보조집단과는 연령 구성을 맞춰도 다른가?
      여러 영역이 동시에 필요한가(필요 영역 수·영역 쌍)? 외출 정도(sq3)·은둔 기간(q4)에 따라 다른가?

- 모든 비율은 비가중 응답자 비율이고 문항별 유효 응답(1~4)만 분모에 둔다. 결측을 0으로 바꾸지 않는다.
- 구간(Wilson·표준화 차이)은 이 응답자 집단 안의 표집변동만 나타낸다. 모집단 추정 구간이 아니다.
- 은둔 집단과 고립 보조집단은 추출 기준(은둔: 외출·이유 기준, 고립: 연령만)과 Q21 문구('은둔생활을…'/'고립생활을…')가 다르다.
- 고립 시트의 q4는 '고립 기간'이라 은둔 기간과 정의가 달라 집단 비교에 쓰지 않는다.
- 공급(서비스 건수)과 나누는 지표는 만들지 않는다. 비교는 q1_compare에서 순위로만 한다.

실행은 노트북 notebooks/03_q1.ipynb에서 한다(이 모듈은 파일을 쓰지 않는다).
"""
from __future__ import annotations

import hashlib
import itertools
import json
import math
import re
import subprocess
import warnings

import numpy as np
import pandas as pd

from ..collect._common import PROCESSED, ROOT
from ..preprocess import survey_2023
from ..preprocess.keywords import Q21_LABELS
from .q1_supply import AREAS, DEMAND_AREAS, wilson

GROUPS = {"hikikomori": "은둔", "isolated_aux": "고립 보조"}
MEASURES = {"very_needed_4": {"4"}, "needed_3_4": {"3", "4"}}
AGE_LABELS = {"1": "19-24세", "2": "25-29세", "3": "30-34세"}
SQ3_LABELS = {"1": "자기 방에서 거의 안 나옴", "2": "방에서 나오지만 집 밖으로 안 나감",
              "3": "인근 편의점 등에만 외출", "4": "취미생활을 위해서만 외출"}
Q4_LABELS = {"1": "3개월 미만", "2": "3~6개월 미만", "3": "6개월~1년 미만", "4": "1~3년 미만",
             "5": "3~5년 미만", "6": "5~10년 미만", "7": "10년 이상"}
HQ1_LABELS = {"1": "남", "2": "여"}
Z95 = 1.959964
MISSING = {".", ""}
META_PATH = PROCESSED / "services_master.meta.json"
EXPECTED_TYPES = {"living", "expense_relief", "asset", "lump_sum", "compensation",
                  "unrestricted_cash", "form_only", "earmarked", "org_target", "none"}


# ----------------------------------------------------------------------------- 입력
def _clean(s: pd.Series) -> pd.Series:
    return s.where(~s.isin(MISSING))


def load_demand() -> tuple[pd.DataFrame, pd.DataFrame]:
    """STEP 00 수요 테이블에 기간(q4)·성별(hq1)을 붙인다. '.'·빈칸은 결측.
    고립 시트의 q4는 고립 기간이므로 q4_isolation으로 이름을 바꾸고 분석에 쓰지 않는다."""
    hiki = pd.read_csv(PROCESSED / "demand_2023_hikikomori.csv", dtype=str, encoding="utf-8-sig")
    iso = pd.read_csv(PROCESSED / "demand_2023_isolated_aux.csv", dtype=str, encoding="utf-8-sig")
    out = []
    for df, sheet, rename in ((hiki, "(데이터)은둔", {}), (iso, "(데이터)고립", {"q4": "q4_isolation"})):
        extra = survey_2023.load_sheet(sheet)[["pid", "q4", "hq1"]].rename(columns=rename)
        merged = df.merge(extra, on="pid", how="left", validate="one_to_one")
        for c in extra.columns.drop("pid"):
            merged[c] = _clean(merged[c])
        out.append(merged)
    return out[0], out[1]


def _git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, encoding="utf-8")


def _sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _codebook_text(sheet: str, var: str) -> str:
    cb = survey_2023.load_sheet(sheet)
    note = cb.loc[cb["변수명"].eq(var), "비고"].dropna()
    return str(note.iloc[0]) if len(note) else ""


def validate_inputs(m: pd.DataFrame, hiki: pd.DataFrame, iso: pd.DataFrame) -> pd.DataFrame:
    """분석 전 입력 점검표. status는 실제 비교 결과(pass/fail)이고, 판단이 필요한 정보는 note로 남긴다."""
    rows: list[dict] = []

    def add(check, expected, observed, status=None, note=""):
        if status is None:
            status = "pass" if str(expected) == str(observed) else "fail"
        rows.append({"check": check, "expected": str(expected), "observed": str(observed), "status": status, "note": note})

    def subset(check, values, allowed, note=""):
        bad = sorted(set(values) - set(allowed), key=str)
        add(check, f"subset of {sorted(allowed, key=str)}", f"unexpected={bad}" if bad else "all within", "fail" if bad else "pass", note)

    meta = json.loads(META_PATH.read_text(encoding="utf-8"))
    # --- 마스터
    add("master rows", 5288, len(m))
    add("master cols == meta.cols", meta["cols"], m.shape[1])
    add("servId duplicates", 0, int(m["servId"].duplicated().sum()))
    src = m["source"].value_counts().to_dict()
    add("source central", 461, src.get("central", 0))
    add("source local", 4827, src.get("local", 0))
    raw_fields = ["summary_text", "target_text", "criteria_text", "benefit_text", "apply_text"]
    need = raw_fields + [f"{c.removesuffix('_text')}_text_raw" for c in raw_fields]
    miss = [c for c in need if c not in m.columns]
    add("원문 필드 존재(5개 + *_raw 5개)", "missing=[]", f"missing={miss}", "pass" if not miss else "fail")
    subset("youth_scope values", m["youth_scope"].dropna().unique(), {"A", "B", "U", "C"})
    add("youth_scope distribution", "(기록용)", json.dumps(m["youth_scope"].value_counts().to_dict(), ensure_ascii=False), "note",
        "A 청년 특화 / B 연령 겹침 / U 판단 불가 / C 제외")
    subset("q21_10_tier values", m["q21_10_tier"].dropna().unique(), {"main", "candidate", "excluded"})
    subset("q21_10_type values", m["q21_10_type"].dropna().unique(), EXPECTED_TYPES)
    subset("q21_07_beneficiary values", m["q21_07_beneficiary"].dropna().unique(), {"dependent", "self", "both", "N"})
    toks = set(m["need_area_q21"].fillna("N").str.split("|").explode())
    subset("need_area_q21 tokens", toks, set(AREAS) | {"N"})
    has7 = m["need_area_q21"].fillna("N").str.split("|").map(lambda v: "q21_07" in v)
    add("⑦이 아닌데 q21_07_beneficiary 값이 있는 행", 0, int((m["q21_07_beneficiary"].notna() & ~has7).sum()))
    # --- 조사
    add("은둔 응답자", 4448, len(hiki))
    add("고립 보조 응답자", 3032, len(iso))
    add("pid 중복(은둔)", 0, int(hiki["pid"].duplicated().sum()))
    add("pid 중복(고립)", 0, int(iso["pid"].duplicated().sum()))
    add("은둔·고립 pid 겹침", 0, len(set(hiki["pid"]) & set(iso["pid"])))
    subset("연령 코드 sq1a1 (은둔)", hiki["sq1a1"].dropna().unique(), set(AGE_LABELS))
    subset("연령 코드 sq1a1 (고립)", iso["sq1a1"].dropna().unique(), set(AGE_LABELS))
    subset("은둔 sq3 코드", hiki["sq3"].dropna().unique(), set(SQ3_LABELS))
    for gname, df in (("hikikomori", hiki), ("isolated_aux", iso)):
        for q in DEMAND_AREAS:
            subset(f"{q} 유효값 ({gname})", df[q].dropna().unique(), {"1", "2", "3", "4"})
            add(f"{q} 결측 수 ({gname})", "(기록용)", int(df[q].isna().sum()), "note", "분모에서 제외하며 0으로 바꾸지 않음")
    qcol = hiki["q4"]
    subset("q4 유효값 (은둔)", qcol.dropna().unique(), set(Q4_LABELS))
    add("q4 결측 수 (은둔)", "(기록용)", int(qcol.isna().sum()), "note", "별도 수준으로 만들지 않고 제외")
    t_h, t_i = _codebook_text("(코드북)은둔", "q4"), _codebook_text("(코드북)고립", "q4")
    add("은둔 코드북 q4 문구", "은둔 생활을 한지 총 얼마나 되었나요?", t_h)
    add("고립 코드북 q4 문구", "(기록용)", t_i, "note",
        "고립 시트 q4는 '고립 생활' 기간이라 은둔 기간과 정의가 달라 집단 비교에 쓰지 않음(q4_isolation)")
    for gname, sheet, pref in (("은둔", "(코드북)은둔", "은둔생활을 벗어나기위한"), ("고립", "(코드북)고립", "고립생활을 벗어나기위한")):
        txt = _codebook_text(sheet, "q211")
        add(f"Q21 문구 ({gname} 코드북)", f"startswith '{pref}'", txt, "note" if txt.startswith(pref) else "fail",
            "은둔 '은둔생활을…' vs 고립 '고립생활을…': 문구가 달라 직접 비교에 한계")
    for gname, df in (("hikikomori", hiki), ("isolated_aux", iso)):
        subset(f"hq1 유효값 ({gname})", df["hq1"].dropna().unique(), set(HQ1_LABELS))
        add(f"hq1 결측 수 ({gname})", "(기록용)", int(df["hq1"].isna().sum()), "note")
    # --- 메타
    add("meta.rows == 파일 행수", meta["rows"], len(m))
    add("meta.columns == 파일 열 순서", "identical", "identical" if meta["columns"] == list(m.columns) else "different",
        "pass" if meta["columns"] == list(m.columns) else "fail")
    for kind in ("inputs_sha256", "code_sha256"):
        bad = [k for k, v in meta[kind].items() if not (ROOT / k).exists() or _sha256(ROOT / k) != v]
        add(f"meta.{kind} 파일 해시 일치", "mismatch=[]", f"mismatch={bad}", "pass" if not bad else "fail")
    add("meta.code_dirty", False, meta["code_dirty"], "pass" if meta["code_dirty"] is False else "fail")
    head = _git("rev-parse", "--short", "HEAD").stdout.strip()
    anc = _git("merge-base", "--is-ancestor", meta["code_commit"], "HEAD")
    if anc.returncode in (0, 1):
        add("meta.code_commit가 HEAD의 조상", True, anc.returncode == 0, "pass" if anc.returncode == 0 else "fail")
    else:
        add("meta.code_commit가 HEAD의 조상", True, f"git 오류: {anc.stderr.strip()}", "note", "git 확인 불가")
    if head != meta["code_commit"]:
        add("meta.code_commit vs HEAD", meta["code_commit"], head, "note",
            "마스터 생성 이후 커밋이 더 쌓임(마스터 입력 코드의 해시는 위에서 일치 확인)")
    spec = (ROOT / "docs" / "services_master_spec.md").read_text(encoding="utf-8")
    key_row = next((line for line in spec.splitlines() if line.startswith("| 키 |")), "")
    doc_cols = [int(n) for n in re.findall(r"(\d+)열", key_row)]
    status = "pass" if meta["cols"] in doc_cols else "note"
    add("services_master_spec.md 열 수 표기", meta["cols"], doc_cols, status,
        "Q1 기준 마스터와 앱 영역 v3 적용본의 스키마를 함께 표기")
    return pd.DataFrame(rows, columns=["check", "expected", "observed", "status", "note"])


# ----------------------------------------------------------------------------- 영역별 수요
def demand_by_group(hiki: pd.DataFrame, iso: pd.DataFrame) -> pd.DataFrame:
    """q1_supply.demand_table을 두 집단에 적용. missing_n은 전체 − 유효 응답, ci_overlap_next는 바로 아래 순위 영역과 구간 겹침."""
    from .q1_supply import demand_table
    parts = []
    for gname, df, fn in (("hikikomori", hiki, "demand_2023_hikikomori.csv"), ("isolated_aux", iso, "demand_2023_isolated_aux.csv")):
        t = demand_table(PROCESSED / fn)
        t.insert(0, "group", gname)
        t["n_total"] = len(df)
        t["missing_n"] = t["n_total"] - t["n_respondents"]
        # 파일을 다시 읽은 값과 메모리 프레임이 같은지 확인
        chk = [int(df[a].isin(MEASURES[me]).sum()) for a, me in zip(t["area"], t["measure"])]
        if chk != t["k"].tolist():
            raise AssertionError(f"{gname}: demand_table과 프레임의 k가 다릅니다")
        parts.append(t)
    t = pd.concat(parts, ignore_index=True)
    t["ci_overlap_next"] = pd.array([pd.NA] * len(t), dtype="boolean")
    for _, g in t.groupby(["group", "measure"]):
        g = g.sort_values(["rate", "area"], ascending=[False, True])
        lo, hi = g["ci95_lo"].to_numpy(), g["ci95_hi"].to_numpy()
        for pos in range(len(g) - 1):
            t.loc[g.index[pos], "ci_overlap_next"] = bool(lo[pos] <= hi[pos + 1])
    t["primary_measure"] = t["measure"].eq("very_needed_4")
    return t


# ----------------------------------------------------------------------------- 직접 표준화
def age_weights(df: pd.DataFrame) -> dict[str, float]:
    """연령대(sq1a1) 응답자 구성비(합 1). 문항 결측과 무관하게 전체 응답자 기준."""
    n = df["sq1a1"].value_counts()
    return {a: float(n.get(a, 0)) / len(df) for a in AGE_LABELS}


def direct_standardize(df: pd.DataFrame, q: str, ok_values: set[str], weights: dict[str, float]) -> dict:
    """연령대별 유효응답 기준 비율 p_a, 표준화율 Σw_a p_a, 분산 Σw_a² p_a(1-p_a)/n_a.
    어떤 연령층의 유효 n이 0이면 값은 NaN이고 status에 표시한다."""
    n_a, k_a, p_a = {}, {}, {}
    for a in AGE_LABELS:
        v = df.loc[df["sq1a1"].eq(a), q].dropna()
        n_a[a], k_a[a] = len(v), int(v.isin(ok_values).sum())
        p_a[a] = k_a[a] / n_a[a] if n_a[a] else math.nan
    if any(n == 0 for n in n_a.values()):
        return {"std": math.nan, "var": math.nan, "se": math.nan, "n_a": n_a, "k_a": k_a, "p_a": p_a, "status": "empty_stratum"}
    std = sum(weights[a] * p_a[a] for a in AGE_LABELS)
    var = sum(weights[a] ** 2 * p_a[a] * (1 - p_a[a]) / n_a[a] for a in AGE_LABELS)
    return {"std": std, "var": var, "se": math.sqrt(var), "n_a": n_a, "k_a": k_a, "p_a": p_a, "status": "ok"}


def _crude(df: pd.DataFrame, q: str, ok: set[str]) -> tuple[int, float]:
    v = df[q].dropna()
    return len(v), (v.isin(ok).mean() if len(v) else math.nan)


def age_standardized_comparison(hiki: pd.DataFrame, iso: pd.DataFrame) -> pd.DataFrame:
    """은둔 − 고립 보조집단 차이: 조율과 연령 표준화율. 표준인구 pooled(주, 두 집단 합산 응답자)·hikikomori(민감도, 은둔 구성).
    차이의 95% 구간은 정규근사이며 이 표본 안의 표집변동만 반영한다."""
    both = pd.concat([hiki[["sq1a1"]], iso[["sq1a1"]]], ignore_index=True)
    standards = {"pooled": age_weights(both), "hikikomori": age_weights(hiki)}
    rows = []
    for q in DEMAND_AREAS:
        for meas, ok in MEASURES.items():
            n_h, c_h = _crude(hiki, q, ok)
            n_i, c_i = _crude(iso, q, ok)
            for sname, w in standards.items():
                sh, si = direct_standardize(hiki, q, ok, w), direct_standardize(iso, q, ok, w)
                d = sh["std"] - si["std"]
                se = math.sqrt(sh["var"] + si["var"])
                rows.append({"area": q, "area_label": Q21_LABELS[q], "measure": meas, "standard": sname,
                             "n_hiki": n_h, "n_iso": n_i, "crude_hiki": c_h, "crude_iso": c_i,
                             "crude_diff_pp": 100 * (c_h - c_i), "std_hiki": sh["std"], "std_iso": si["std"],
                             "std_diff_pp": 100 * d, "std_diff_ci95_lo_pp": 100 * (d - Z95 * se), "std_diff_ci95_hi_pp": 100 * (d + Z95 * se),
                             "crude_minus_std_pp": 100 * (c_h - c_i) - 100 * d})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- 필요 영역 수
def need_count(df: pd.DataFrame, measure: str) -> pd.Series:
    """응답자별 필요 영역 수(0~10). Q21 10문항이 모두 유효한 응답자만 값이 있고 나머지는 NaN(0이 아님)."""
    v = df[DEMAND_AREAS]
    return v.isin(MEASURES[measure]).sum(axis=1).astype(float).where(v.notna().all(axis=1))


def is_straightline(df: pd.DataFrame) -> pd.Series:
    """10문항을 모두 유효하게 답했고 값이 전부 같은 응답자(동일 응답)."""
    v = df[DEMAND_AREAS]
    return v.notna().all(axis=1) & v.nunique(axis=1).eq(1)


def _groups(hiki, iso):
    return (("hikikomori", hiki), ("isolated_aux", iso))


def need_count_distribution(hiki: pd.DataFrame, iso: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for g, df in _groups(hiki, iso):
        for meas in MEASURES:
            c = need_count(df, meas).dropna().astype(int)
            n = len(c)
            for k in range(11):
                nk = int((c == k).sum())
                rows.append({"group": g, "measure": meas, "n_areas": k, "n_respondents": nk, "share": nk / n if n else math.nan,
                             "share_at_least": int((c >= k).sum()) / n if n else math.nan,
                             "n_valid": n, "n_excluded_missing": len(df) - n})
    return pd.DataFrame(rows)


def need_count_summary(hiki: pd.DataFrame, iso: pd.DataFrame) -> pd.DataFrame:
    """필요 영역 수 요약. excl_straightline = 10문항 응답이 모두 같은 응답자 제외(탐색적 민감도).
    straightline_share·all_four_share는 10문항 완전 응답자 기준 비율(sample과 무관한 참고값)."""
    rows = []
    for g, df in _groups(hiki, iso):
        sl = is_straightline(df)
        complete = df[DEMAND_AREAS].notna().all(axis=1)
        all4 = complete & df[DEMAND_AREAS].eq("4").all(axis=1)
        for meas in MEASURES:
            c = need_count(df, meas)
            for sample, keep in (("all", complete), ("excl_straightline", complete & ~sl)):
                x = c[keep]
                rows.append({"group": g, "measure": meas, "sample": sample, "n_valid": len(x), "mean": x.mean(), "median": x.median(),
                             "q1": x.quantile(0.25), "q3": x.quantile(0.75), "share_0": (x == 0).mean(), "share_10": (x == 10).mean(),
                             "straightline_share": sl.sum() / complete.sum(), "all_four_share": all4.sum() / complete.sum()})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- 영역 쌍 동시 필요
def _phi(n11, n10, n01, n00) -> float:
    den = math.sqrt((n11 + n10) * (n01 + n00) * (n11 + n01) * (n10 + n00))
    return (n11 * n00 - n10 * n01) / den if den else math.nan


def pair_cooccurrence(hiki: pd.DataFrame, iso: pd.DataFrame) -> pd.DataFrame:
    """45개 영역 쌍 × 집단 × 측정 × 표본. 분모는 두 문항이 모두 유효한 응답자(쌍마다 다름)."""
    rows = []
    for g, df in _groups(hiki, iso):
        keep_all = pd.Series(True, index=df.index)
        keep_ex = ~is_straightline(df)
        for meas, ok in MEASURES.items():
            for sample, keep in (("all", keep_all), ("excl_straightline", keep_ex)):
                d = df[keep]
                for a, b in itertools.combinations(DEMAND_AREAS, 2):
                    valid = d[a].notna() & d[b].notna()
                    xa, xb = d.loc[valid, a].isin(ok), d.loc[valid, b].isin(ok)
                    n = int(valid.sum())
                    n11, n10, n01 = int((xa & xb).sum()), int((xa & ~xb).sum()), int((~xa & xb).sum())
                    n00 = n - n11 - n10 - n01
                    na, nb = n11 + n10, n11 + n01
                    lo, hi = wilson(n11, n)
                    pa, pb = (na / n, nb / n) if n else (math.nan, math.nan)
                    rows.append({"group": g, "measure": meas, "sample": sample, "area_a": a, "area_b": b,
                                 "label_a": Q21_LABELS[a], "label_b": Q21_LABELS[b], "n_valid_pair": n, "n_both": n11,
                                 "joint_rate": n11 / n if n else math.nan, "joint_ci95_lo": lo, "joint_ci95_hi": hi,
                                 "rate_a": pa, "rate_b": pb,
                                 "cond_b_given_a": n11 / na if na else math.nan, "cond_a_given_b": n11 / nb if nb else math.nan,
                                 "lift": (n11 / n) / (pa * pb) if n and pa * pb else math.nan, "phi": _phi(n11, n10, n01, n00)})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- 은둔 하위집단
def subgroup_demand(hiki: pd.DataFrame) -> pd.DataFrame:
    """sq3(외출 정도)·q4(은둔 기간) 수준별 영역 수요. 변수 결측자는 제외(별도 수준 아님).
    age_std는 은둔 전체 연령 구성으로 직접표준화하고, 비교 기준 all은 같은 영역의 은둔 전체 조율이다."""
    w = age_weights(hiki)
    rows = []
    for var, labels in (("sq3", SQ3_LABELS), ("q4", Q4_LABELS)):
        for lv, lab in labels.items():
            sub = hiki[hiki[var].eq(lv)]
            for q in DEMAND_AREAS:
                for meas, ok in MEASURES.items():
                    n_all, rate_all = _crude(hiki, q, ok)
                    v = sub[q].dropna()
                    n, k = len(v), int(v.isin(ok).sum())
                    lo, hi = wilson(k, n)
                    st = direct_standardize(sub, q, ok, w)
                    rate = k / n if n else math.nan
                    rows.append({"variable": var, "level": lv, "level_label": lab, "n_level": len(sub), "area": q, "area_label": Q21_LABELS[q],
                                 "measure": meas, "n_valid": n, "k": k, "rate": rate, "ci95_lo": lo, "ci95_hi": hi,
                                 "diff_vs_all_pp": 100 * (rate - rate_all), "age_std_rate": st["std"],
                                 "age_std_diff_vs_all_pp": 100 * (st["std"] - rate_all), "min_age_stratum_n": min(st["n_a"].values())})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- 부록: 로지스틱 회귀
def _bh(p: pd.Series) -> pd.Series:
    """Benjamini-Hochberg 보정 p. 결측 p는 제외하고 계산한다."""
    from statsmodels.stats.multitest import multipletests
    out = pd.Series(math.nan, index=p.index)
    ok = p.notna()
    if ok.any():
        out[ok] = multipletests(p[ok].to_numpy(), method="fdr_bh")[1]
    return out


def _term_info(term: str) -> tuple[str, str]:
    mt = re.match(r'C\((\w+), Treatment\("([^"]+)"\)\)\[T\.([^\]]+)\]', term)
    if not mt:
        return term, term
    var, ref, lv = mt.groups()
    labs = {"sq3": SQ3_LABELS, "q4": Q4_LABELS, "sq1a1": AGE_LABELS, "hq1": HQ1_LABELS,
            "group": {"hikikomori": "은둔 (기준: 고립 보조)"}}[var]
    return var, f"{var}={lv} {labs.get(lv, lv)} (기준 {ref})"


def _fit(d: pd.DataFrame, formula: str) -> tuple[object | None, str]:
    """Logit 적합. 예외·완전분리·미수렴·특이행렬은 status에 기록하고 값은 만들지 않는다(임의 보정 금지)."""
    import statsmodels.formula.api as smf
    from statsmodels.tools.sm_exceptions import PerfectSeparationError
    if d["y"].nunique() < 2:
        return None, "no_variation_in_y"
    error = None
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        try:
            res = smf.logit(formula, data=d).fit(disp=0, maxiter=200)
        except PerfectSeparationError:
            return None, "separation:PerfectSeparationError"
        except Exception as e:  # noqa: BLE001  (LinAlgError 등)
            error = e
    msgs = sorted({f"{x.category.__name__}" for x in w})
    if any("Separation" in x.category.__name__ or "separation" in str(x.message).lower() for x in w):
        return None, "separation:" + ",".join(msgs)
    if error is not None:
        return None, f"error:{type(error).__name__}:{str(error)[:80]}"
    if not res.mle_retvals.get("converged", False):
        return None, "not_converged:" + ",".join(msgs)
    if not np.isfinite(res.bse).all():
        return None, "singular_or_nonfinite_se"
    return res, ("ok" if not msgs else "ok_with_warning:" + ",".join(msgs))


def fit_logit_appendix(hiki: pd.DataFrame, iso: pd.DataFrame) -> pd.DataFrame:
    """부록(탐색적): '매우 필요(4)' 여부의 로지스틱 회귀. 본문 결론·공급 순위에 쓰지 않고 인과로 해석하지 않는다.
    모형 A(within_hikikomori): sq3(기준 4)·q4(기준 1)·연령·성별, 은둔만. 모형 B(group_compare): 집단(기준 고립)·연령·성별.
    영역별·모형별 listwise 제외. BH는 초점 항 전체(A 90개, B 10개) 범위."""
    h, i = hiki.copy(), iso.copy()
    h["group"], i["group"] = "hikikomori", "iso"
    pool = pd.concat([h[["group", "sq1a1", "hq1", *DEMAND_AREAS]], i[["group", "sq1a1", "hq1", *DEMAND_AREAS]]], ignore_index=True)
    models = {
        "within_hikikomori": (h, 'y ~ C(sq3, Treatment("4")) + C(q4, Treatment("1")) + C(sq1a1, Treatment("1")) + C(hq1, Treatment("1"))',
                              ["sq3", "q4", "sq1a1", "hq1"],
                              [f"C(sq3, Treatment(\"4\"))[T.{k}]" for k in "123"] + [f"C(q4, Treatment(\"1\"))[T.{k}]" for k in "234567"],
                              {"sq3", "q4"}, "within_hikikomori_sq3_q4_90"),
        "group_compare": (pool, 'y ~ C(group, Treatment("iso")) + C(sq1a1, Treatment("1")) + C(hq1, Treatment("1"))',
                          ["group", "sq1a1", "hq1"], ['C(group, Treatment("iso"))[T.hikikomori]'], {"group"}, "group_compare_10"),
    }
    rows = []
    for model, (df, formula, cols, focal_terms, focal_vars, family) in models.items():
        for q in DEMAND_AREAS:
            d = df[[*cols, q]].copy()
            d["y"] = d[q].eq("4").astype(float).where(d[q].notna())
            d = d.drop(columns=q).dropna()
            res, status = _fit(d, formula)
            n_used, n_events = len(d), int(d["y"].sum())
            if res is not None:
                ci = res.conf_int()
                est = pd.DataFrame({"coef": res.params, "se": res.bse, "p_value": res.pvalues, "lo": ci[0], "hi": ci[1]})
                terms = list(est.index)
            else:
                est, terms = pd.DataFrame(), []
            for t in focal_terms + [t for t in terms if t not in focal_terms]:
                var, lab = _term_info(t)
                has = t in est.index
                rows.append({"model": model, "area": q, "area_label": Q21_LABELS[q], "term": t, "term_label": lab,
                             "is_focal": t in focal_terms,
                             "odds_ratio": math.exp(est.at[t, "coef"]) if has else math.nan,
                             "or_ci95_lo": math.exp(est.at[t, "lo"]) if has else math.nan,
                             "or_ci95_hi": math.exp(est.at[t, "hi"]) if has else math.nan,
                             "coef": est.at[t, "coef"] if has else math.nan, "se": est.at[t, "se"] if has else math.nan,
                             "p_value": est.at[t, "p_value"] if has else math.nan, "p_bh": math.nan,
                             "fdr_family": family if t in focal_terms else "", "n_used": n_used, "n_events": n_events,
                             "converged": res is not None, "status": status if (has or res is None) else "term_absent"})
    t = pd.DataFrame(rows)
    for model in models:
        idx = t.index[t["model"].eq(model) & t["is_focal"]]
        t.loc[idx, "p_bh"] = _bh(t.loc[idx, "p_value"])
    return t
