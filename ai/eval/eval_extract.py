"""1단계 평가: 0단계 위기 확인 + 1단계 욕구·조건 추출을 개발 세트에 돌리고 채점한다.

- 위기: 키워드 / LLM / 키워드 OR LLM 각각의 재현율·정밀도. 유형 10(위기 신호형) 중 사람 정답 N인 문장
  (합의안 221·251번 등 별도 감지 검토 대상)은 따로 센다.
- 추출: 사람 정답 위기 N인 문장만. 영역(집합 일치, micro P/R/F1, 영역별), 활동범위, 제약(가장 강한 하나),
  나이·지역 언급(카드만), 되묻기 여부. few-shot 예시는 같은 문장을 빼고 고른다(pick_examples).
- 예측은 runs/<이름>/predictions.jsonl에 한 줄씩 쌓고, 다시 실행하면 끝난 id는 건너뛴다(Colab 끊김 대비).

실행 (ai/ 폴더에서)
  python -m eval.eval_extract --mock                 # 모델 없이 흐름 확인
  python -m eval.eval_extract --run kanana_v1        # Colab GPU, config.yaml 모델
  python -m eval.eval_extract --run kanana_v1 --score-only
  python -m eval.eval_extract --run crisis_v2 --crisis-only   # 위기 프롬프트만 비교
  python -m eval.eval_extract --run kanana_v1_post --repost kanana_v1   # 기존 예측에 후처리만 다시 적용
  python -m eval.eval_extract --run retry_v1 --sets corpus,cards --ids U01-015,C07-11   # 특정 문장만
  python -m eval.eval_extract --run holdout_v1 --sets holdout   # 최종: 평가 카드 80장 (설정 동결 후 1회)
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import yaml

AI = Path(__file__).resolve().parents[1]
DATA = {"corpus": AI / "data" / "dev_corpus.jsonl", "cards": AI / "data" / "dev_cards.jsonl",
        "holdout": AI / "data" / "holdout_cards.jsonl"}
RUNS = AI / "eval" / "runs"


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]


def _prf(tp: int, fp: int, fn: int) -> dict:
    p = tp / (tp + fp) if tp + fp else None
    r = tp / (tp + fn) if tp + fn else None
    f = 2 * p * r / (p + r) if p and r else None
    return {"precision": p, "recall": r, "f1": f, "tp": tp, "fp": fp, "fn": fn}


def score_crisis(rows: list[dict]) -> dict:
    out = {}
    for key in ("keyword", "llm", "combined"):
        tp = fp = fn = 0
        for r in rows:
            g, p = r["gold"]["위기"] == "Y", r["pred"]["crisis"][key]
            if p is None:
                continue
            tp += g and p
            fp += (not g) and p
            fn += g and not p
        out[key] = _prf(tp, fp, fn)
    out["missed"] = [r["id"] for r in rows if r["gold"]["위기"] == "Y" and not r["pred"]["crisis"]["combined"]]
    t10_n = [r for r in rows if r["type_no"] == 10 and r["gold"]["위기"] == "N"]
    out["type10_gold_N"] = {"n": len(t10_n), "flagged": [r["id"] for r in t10_n if r["pred"]["crisis"]["combined"]]}
    out["false_alarm_other_types"] = [r["id"] for r in rows
                                      if r["type_no"] != 10 and r["gold"]["위기"] == "N" and r["pred"]["crisis"]["combined"]]
    # LLM 호출 실패를 안전한 쪽(위기)으로 둔 건수. 키워드에 안 걸렸는데 llm_error로 위기가 된 것은 오류 탓 판정
    err = [r for r in rows if r["pred"]["crisis"].get("llm_by") == "llm_error"]
    out["llm_error"] = {"n": len(err), "ids": [r["id"] for r in err],
                        "only_cause": [r["id"] for r in err if not r["pred"]["crisis"]["keyword"]],
                        "gold_N": [r["id"] for r in err if r["gold"]["위기"] == "N"]}
    return out


def score_extract(rows: list[dict], domains: list[str]) -> dict:
    rows = [r for r in rows if r["gold"]["위기"] == "N" and r["pred"].get("gold_view")]
    if not rows:
        return {"n": 0}
    exact = sum(set(r["gold"]["영역"]) == set(r["pred"]["gold_view"]["영역"]) for r in rows)
    tp = fp = fn = 0
    per = {d: [0, 0, 0] for d in domains}
    for r in rows:
        g, p = set(r["gold"]["영역"]), set(r["pred"]["gold_view"]["영역"])
        tp, fp, fn = tp + len(g & p), fp + len(p - g), fn + len(g - p)
        for d in domains:
            per[d][0] += d in g and d in p
            per[d][1] += d in p and d not in g
            per[d][2] += d in g and d not in p
    out = {"n": len(rows), "domain_exact": exact / len(rows), "domain_micro": _prf(tp, fp, fn),
           "domain_by_label": {d: _prf(*v) for d, v in per.items() if sum(v)}}

    for field in ("활동범위", "제약", "나이언급", "지역언급"):
        sub = [r for r in rows if field in r["gold"]]
        if not sub:
            continue
        ok = sum(r["gold"][field] == r["pred"]["gold_view"][field] for r in sub)
        conf = Counter((r["gold"][field], r["pred"]["gold_view"][field]) for r in sub)
        out[field] = {"n": len(sub), "accuracy": ok / len(sub),
                      "confusion": {f"{g} → {p}": c for (g, p), c in sorted(conf.items()) if g != p}}

    # 되묻기: 사람 정답 영역이 없는 문장에서 추천 대신 되묻는지
    ask = [r for r in rows if not r["gold"]["영역"]]
    out["되묻기"] = {"gold_none_n": len(ask),
                    "asked": sum(r["pred"]["next_action"] == "되묻기" for r in ask),
                    "asked_when_gold_has_domain": sum(r["pred"]["next_action"] == "되묻기"
                                                     for r in rows if r["gold"]["영역"])}
    out["evidence_not_found"] = sum(n["status"] == "확인 필요" for r in rows for n in r["pred"].get("needs", []))
    out["extract_failed"] = [r["id"] for r in rows if r["pred"].get("extract_status", "ok") == "failed"]
    return out


class Stage1:
    """0·1단계만 쓰는 최소 로더. 검색 색인(BM25·서비스 임베딩)은 만들지 않는다. pipeline.Pipeline과 같은 설정을 읽는다."""

    def __init__(self, config_path="configs/config.yaml", mock=False):
        from core.models import load_models
        self.cfg = cfg = yaml.safe_load(open(config_path, encoding="utf-8"))
        p = cfg["paths"]
        self.embedder, self.llm = load_models(cfg, mock)
        self.domains = yaml.safe_load(open(p["domains_file"], encoding="utf-8"))["domains"]
        self.rules_text = open(p["rules_file"], encoding="utf-8").read()
        self.examples = [r for r in read_jsonl(Path(p["dev_examples"])) if r.get("label")]
        self.example_emb = self.embedder.encode([e["text"] for e in self.examples]) if self.examples else None


def predict(rows: list[dict], pipe, done: set, out_path: Path, limit: int | None,
            crisis_only: bool = False) -> None:
    from extract.extract import extract
    from extract.schema import to_gold_labels
    from safety.crisis import keyword_hit, llm_check

    todo = [r for r in rows if r["id"] not in done][: limit or None]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "a", encoding="utf-8") as f:
        for i, r in enumerate(todo, 1):
            t0 = time.time()
            kw = keyword_hit(r["text"])
            llm = llm_check(r["text"], pipe.llm)
            pred = {"crisis": {"keyword": bool(kw), "keyword_term": kw, "llm": llm["crisis"],
                               "llm_by": llm.get("by"), "combined": bool(kw) or llm["crisis"]}}
            if r["gold"]["위기"] == "N" and not crisis_only:   # 추출 평가는 위기 아닌 문장만 (0단계 결과와 독립)
                ext = extract(r["text"], pipe.llm, pipe.embedder, pipe.rules_text, pipe.domains,
                              pipe.examples, pipe.example_emb, pipe.cfg["extract"])
                pred.update(ext)
                pred["gold_view"] = to_gold_labels(ext)
                if ext.get("extract_status") == "failed":   # 실패 원인 확인용 (JSON 잘림 등)
                    raw = getattr(pipe.llm, "last_raw", None)
                    pred["llm_raw_tail"] = raw[-300:] if isinstance(raw, str) else None
                    pred["llm_raw_len"] = len(raw) if isinstance(raw, str) else None
            pred["sec"] = round(time.time() - t0, 2)
            f.write(json.dumps({"id": r["id"], "pred": pred}, ensure_ascii=False) + "\n")
            f.flush()
            if i % 10 == 0 or i == len(todo):
                print(f"  {i}/{len(todo)} ({r['id']})", flush=True)


def repost(rows: list[dict], src: Path, out_path: Path) -> None:
    """다른 실행의 예측에 extract/postprocess.py만 다시 적용한다 (모델 호출 없음)."""
    from extract.postprocess import postprocess
    from extract.schema import to_gold_labels

    text = {r["id"]: r["text"] for r in rows}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for p in read_jsonl(src):
            pred = p["pred"]
            if pred.get("gold_view"):
                rec = postprocess({k: pred[k] for k in ("schema_version", "needs", "activity_range",
                                                         "activity_evidence", "constraints", "age",
                                                         "region_sido", "next_action")}, text[p["id"]])
                pred = {**pred, **rec, "gold_view": to_gold_labels(rec)}
            f.write(json.dumps({"id": p["id"], "pred": pred}, ensure_ascii=False) + "\n")


def fmt(x):
    return "-" if x is None else f"{x:.3f}" if isinstance(x, float) else str(x)


def report(scores: dict) -> str:
    lines = []
    for name, s in scores.items():
        c, e = s["crisis"], s["extract"]
        lines.append(f"## {name} (n={s['n']})")
        for k in ("keyword", "llm", "combined"):
            lines.append(f"- 위기 {k}: 재현율 {fmt(c[k]['recall'])}, 정밀도 {fmt(c[k]['precision'])} "
                         f"(tp {c[k]['tp']}, fp {c[k]['fp']}, fn {c[k]['fn']})")
        lines.append(f"- 위기 놓침: {c['missed'] or '없음'} / 유형10 정답 N {c['type10_gold_N']['n']}건 중 감지 "
                     f"{c['type10_gold_N']['flagged']} / 다른 유형 오경보 {c['false_alarm_other_types'] or '없음'}")
        if "llm_error" in c:
            le = c["llm_error"]
            lines.append(f"- 위기 llm_error: {le['n']}건 (키워드 없이 오류로만 위기 {le['only_cause'] or '없음'}, "
                         f"정답 N {le['gold_N'] or '없음'})")
        if e.get("n"):
            m = e["domain_micro"]
            lines.append(f"- 추출 n={e['n']}: 영역 완전일치 {fmt(e['domain_exact'])}, micro P {fmt(m['precision'])} "
                         f"R {fmt(m['recall'])} F1 {fmt(m['f1'])}")
            for field in ("활동범위", "제약", "나이언급", "지역언급"):
                if field in e:
                    lines.append(f"- {field} 정확도 {fmt(e[field]['accuracy'])} (n={e[field]['n']}) 오답 {e[field]['confusion']}")
            lines.append(f"- 되묻기: 정답 영역 없음 {e['되묻기']['gold_none_n']}건 중 {e['되묻기']['asked']}건, "
                         f"정답 영역 있는데 되묻기 {e['되묻기']['asked_when_gold_has_domain']}건 / 원문에 없는 근거 {e['evidence_not_found']}"
                         f" / 추출 실패 {len(e.get('extract_failed', []))}건 {e.get('extract_failed') or ''}")
            lines.append("- 영역별 F1: " + ", ".join(f"{d} {fmt(v['f1'])}" for d, v in e["domain_by_label"].items()))
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=None, help="결과 폴더 이름 (기본: mock 또는 날짜)")
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--sets", default="cards,corpus")
    ap.add_argument("--limit", type=int, default=None, help="세트마다 새로 돌릴 최대 건수")
    ap.add_argument("--score-only", action="store_true")
    ap.add_argument("--repost", default=None, help="이 실행의 예측에 후처리만 다시 적용 (모델 불필요)")
    ap.add_argument("--ids", default=None, help="쉼표로 구분한 id만 실행·채점 (실패 문장 재확인용)")
    ap.add_argument("--crisis-only", action="store_true", help="0단계 위기 확인만 실행 (추출 생략, 빠른 비교용)")
    args = ap.parse_args()

    run = RUNS / (args.run or ("mock" if args.mock else time.strftime("%Y%m%d_%H%M")))
    domains = yaml.safe_load(open(AI / "configs" / "domains.yaml", encoding="utf-8"))["domains"]
    sets = {}
    for s in args.sets.split(","):
        if not DATA[s].exists():
            raise SystemExit(f"{DATA[s].name} 없음 (평가 카드는 python -m eval.build_devset --holdout)")
        sets[s] = read_jsonl(DATA[s])
    if args.ids:
        keep = set(args.ids.split(","))
        sets = {k: [r for r in v if r["id"] in keep] for k, v in sets.items()}

    if args.repost:
        import sys
        sys.path.insert(0, str(AI))
        for name, rows in sets.items():
            repost(rows, RUNS / args.repost / f"pred_{name}.jsonl", run / f"pred_{name}.jsonl")
    elif not args.score_only:
        import os
        os.chdir(AI)                       # config.yaml 상대 경로 기준
        pipe = Stage1(mock=args.mock)
        for name, rows in sets.items():
            path = run / f"pred_{name}.jsonl"
            done = {p["id"] for p in read_jsonl(path)}
            print(f"[{name}] {len(rows)}건 중 {len(done)}건 완료, 이어서 실행")
            predict(rows, pipe, done, path, args.limit, args.crisis_only)

    scores = {}
    for name, rows in sets.items():
        preds = {p["id"]: p["pred"] for p in read_jsonl(run / f"pred_{name}.jsonl")}
        merged = [{**r, "pred": preds[r["id"]]} for r in rows if r["id"] in preds]
        scores[name] = {"n": len(merged), "crisis": score_crisis(merged), "extract": score_extract(merged, domains)}
    (run / "scores.json").write_text(json.dumps(scores, ensure_ascii=False, indent=2), encoding="utf-8")
    text = report(scores)
    (run / "report.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
