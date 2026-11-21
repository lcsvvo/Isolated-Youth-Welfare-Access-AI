"""평가 카드 80장(holdout) 결과를 제안서용 표로 나눈다. 모델을 다시 부르지 않고 예측 파일만 읽는다.

- 위기: 키워드 / LLM / 결합 재현율(x/8), 비위기 오경보(x/72), llm_error
- 추출(위기 N): 영역 micro P/R/F1·완전일치, 활동범위·제약·나이·지역 정확도, 다음 행동 일치율
  전체 + 명시성(직접 요청 / 상황만 서술)별. 비율에는 카드 단위 부트스트랩 95% 구간
- 노동(일하는 중) 정답 카드 목록 (노동 검색 누락 Step 1 대상)

실행 (ai/ 폴더에서, eval_extract --run holdout_v1 --sets holdout 다음)
  python -m eval.holdout_breakdown --run holdout_v1
출력: eval/runs/<run>/breakdown.md, breakdown.csv
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import pandas as pd

from eval.build_devset import load_cards
from eval.eval_extract import DATA, RUNS, read_jsonl

LABOR = "노동(일하는 중)"
SEED, N_BOOT = 20261118, 2000


def explicitness(value: str) -> str:
    v = str(value or "")
    return "직접 요청" if v.startswith("직접") else "상황만 서술" if v.startswith("상황") else v or "미기재"


def expected_next(gold_action: str) -> str:
    """기대행동 → 1단계 next_action. 되묻기만 되묻기, 나머지(추천·자격 확인 필요·대조군)는 추천."""
    return "되묻기" if str(gold_action).startswith("되묻기") else "추천"


def card_metrics(r: dict) -> dict:
    g, p = r["gold"], r["pred"]
    gv = p.get("gold_view") or {}
    gd, pd_ = set(g["영역"]), set(gv.get("영역", []))
    return {"tp": len(gd & pd_), "fp": len(pd_ - gd), "fn": len(gd - pd_), "exact": int(gd == pd_),
            "활동범위": int(g["활동범위"] == gv.get("활동범위")), "제약": int(g["제약"] == gv.get("제약")),
            "나이언급": int(g["나이언급"] == gv.get("나이언급")), "지역언급": int(g["지역언급"] == gv.get("지역언급")),
            "다음행동": int(expected_next(g["기대행동"]) == p.get("next_action")),
            "failed": int(p.get("extract_status") == "failed")}


def summarize(ms: list[dict]) -> dict:
    tp, fp, fn = (sum(m[k] for m in ms) for k in ("tp", "fp", "fn"))
    prec = tp / (tp + fp) if tp + fp else None
    rec = tp / (tp + fn) if tp + fn else None
    f1 = 2 * prec * rec / (prec + rec) if prec and rec else (0.0 if prec is not None and rec is not None else None)
    out = {"n": len(ms), "영역_P": prec, "영역_R": rec, "영역_F1": f1}
    for k in ("exact", "활동범위", "제약", "나이언급", "지역언급", "다음행동", "failed"):
        out[k] = sum(m[k] for m in ms) / len(ms) if ms else None
    return out


def bootstrap(ms: list[dict], keys: list[str]) -> dict:
    rng = random.Random(SEED)
    draws = {k: [] for k in keys}
    for _ in range(N_BOOT):
        s = summarize([rng.choice(ms) for _ in ms])
        for k in keys:
            if s[k] is not None:
                draws[k].append(s[k])
    ci = {}
    for k, v in draws.items():
        v.sort()
        ci[k] = (v[int(0.025 * len(v))], v[int(0.975 * len(v)) - 1]) if v else (None, None)
    return ci


def fmt(x) -> str:
    return "-" if x is None else f"{x:.3f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="holdout_v1")
    args = ap.parse_args()
    run = RUNS / args.run
    preds = {p["id"]: p["pred"] for p in read_jsonl(run / "pred_holdout.jsonl")}
    cards = read_jsonl(DATA["holdout"])
    meta = load_cards().set_index("카드ID")
    rows = []
    for c in cards:
        if c["id"] not in preds:
            continue
        m = meta.loc[c["id"]]
        rows.append({**c, "pred": preds[c["id"]], "명시성": explicitness(m["명시성"]),
                     "길이": m["길이"], "말투": m["말투"]})
    if len(rows) != len(cards):
        print(f"주의: 예측 {len(rows)}장 / 카드 {len(cards)}장")

    lines = [f"# {args.run} 평가 카드 결과 (n={len(rows)})", ""]
    # 위기
    pos = [r for r in rows if r["gold"]["위기"] == "Y"]
    neg = [r for r in rows if r["gold"]["위기"] == "N"]
    lines += ["## 0단계 위기 확인", "", "| 방식 | 재현율 | 비위기 오경보 |", "| --- | --- | --- |"]
    for k, name in (("keyword", "키워드"), ("llm", "LLM"), ("combined", "결합")):
        hit = sum(bool(r["pred"]["crisis"][k]) for r in pos)
        fa = sum(bool(r["pred"]["crisis"][k]) for r in neg)
        lines.append(f"| {name} | {hit}/{len(pos)} | {fa}/{len(neg)} |")
    missed = [r["id"] for r in pos if not r["pred"]["crisis"]["combined"]]
    err = [r["id"] for r in rows if r["pred"]["crisis"].get("llm_by") == "llm_error"]
    lines += ["", f"- 결합 놓침: {missed or '없음'} / llm_error: {len(err)}건 {err or ''}", ""]

    # 추출
    groups = {"전체": neg}
    for g in ("직접 요청", "상황만 서술"):
        groups[g] = [r for r in neg if r["명시성"] == g]
    keys = ["영역_F1", "exact", "활동범위", "제약", "다음행동"]
    table = []
    for name, rs in groups.items():
        if not rs:
            continue
        ms = [card_metrics(r) for r in rs]
        s, ci = summarize(ms), bootstrap(ms, keys)
        table.append({"구분": name, **s, **{f"{k}_ci": ci[k] for k in keys}})
    lines += ["## 1단계 욕구·조건 추출 (위기 N)", "",
              "| 구분 | n | 영역 P | 영역 R | 영역 F1 [95% CI] | 완전일치 | 활동범위 [95% CI] | 제약 [95% CI] | 나이 | 지역 | 다음 행동 | 추출 실패 |",
              "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for t in table:
        def wci(k):
            lo, hi = t[f"{k}_ci"]
            return f"{fmt(t[k])} [{fmt(lo)}, {fmt(hi)}]"
        lines.append(f"| {t['구분']} | {t['n']} | {fmt(t['영역_P'])} | {fmt(t['영역_R'])} | {wci('영역_F1')} | "
                     f"{fmt(t['exact'])} | {wci('활동범위')} | {wci('제약')} | {fmt(t['나이언급'])} | "
                     f"{fmt(t['지역언급'])} | {fmt(t['다음행동'])} | {fmt(t['failed'])} |")
    lines += ["", f"- 부트스트랩: 카드 단위 재표집 {N_BOOT}회, seed {SEED}", ""]

    # 활동범위 대조군: '필요하면 외출' 정답을 은둔 쪽(방 안·집 안·집 근처)으로 과대판정한 수
    inward = {"방 안", "집 안", "집 근처"}
    out_gold = [r for r in neg if r["gold"]["활동범위"] == "필요하면 외출"]
    over = [r["id"] for r in out_gold if (r["pred"].get("gold_view") or {}).get("활동범위") in inward]
    lines += [f"- 활동범위 과대판정(정답 필요하면 외출 → 방 안·집 안·집 근처): {len(over)}/{len(out_gold)} {over or ''}"]
    labor = [r["id"] for r in rows if LABOR in r["gold"]["영역"]]
    lines += [f"- 노동(일하는 중) 정답 카드: {len(labor)}장 {labor or ''}", ""]

    (run / "breakdown.md").write_text("\n".join(lines), encoding="utf-8")
    flat = [{k: (v if not isinstance(v, tuple) else f"{fmt(v[0])}~{fmt(v[1])}") for k, v in t.items()} for t in table]
    pd.DataFrame(flat).to_csv(run / "breakdown.csv", index=False, encoding="utf-8-sig")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
