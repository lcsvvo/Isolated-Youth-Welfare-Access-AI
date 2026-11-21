"""Dense 문서 구성 비교. 개발 영역 일치는 정식 관련성 정확도가 아니다.

review: 인계 질의 Top-5 사람 관련성 판정(relevance_0_1_2)을 모델·문서 구성별로 집계한다.
  python ai/scripts/dense_quality_eval.py review --rankings <tables/handoff_rankings.csv>
      --review <handoff_relevance_review.csv 또는 dense_human_review_62.csv> --out <폴더>
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from time import perf_counter

import pandas as pd

AI = Path(__file__).resolve().parents[1]
ROOT = AI.parent
sys.path.insert(0, str(AI))
from core.models import Embedder
from retrieve.dense import DenseIndex
from retrieve.dense_texts import PROFILES, build_texts

MODELS = {"bge-m3": "BAAI/bge-m3", "kure-v1": "nlpai-lab/KURE-v1"}


def load_development(path):
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    selected = []
    for row in rows:
        gold = row.get("gold") or {}
        if gold.get("위기") == "N" and gold.get("영역"):
            selected.append({"id": "dev:" + row["id"], "dense": row["text"],
                             "gold": list(gold["영역"]), "kind": "development_raw"})
    if not selected or len({r["id"] for r in selected}) != len(selected):
        raise ValueError("개발 질의가 없거나 ID가 중복되었습니다")
    return selected


def domain_metrics(ranks, queries, domains, k=10):
    """분모 k 유지. 서비스 영역이 없는 질의도 전체 결과에 포함하고 별도 표시."""
    supported = set().union(*domains.values()) if domains else set()
    records = []
    for query, results in zip(queries, ranks):
        if query["kind"] != "development_raw":
            continue
        gold = set(query["gold"])
        matches = sum(bool(gold & domains.get(sid, set())) for sid, _ in results[:k])
        records.append({"query_id": query["id"], "gold_domains": "|".join(sorted(gold)),
                        "supported_gold": bool(gold & supported), "k": k,
                        "domain_matches": matches, "domain_precision": matches / k,
                        "domain_hit": int(matches > 0)})
    return records


def run(args):
    output = args.out / args.model
    if output.exists():
        raise FileExistsError(f"결과 덮어쓰기 방지: 새 --out 사용 필요: {output}")
    master = pd.read_csv(args.master, dtype=str, encoding="utf-8-sig", keep_default_na=False)
    if master.empty or master.servId.str.strip().eq("").any() or master.servId.duplicated().any():
        raise ValueError("서비스 또는 ID가 비었거나 중복되었습니다")
    # 모든 필드 사전 검사: 긴 GPU 작업 전에 입력 문제를 드러낸다.
    documents = {profile: build_texts(master, profile) for profile in PROFILES}
    queries = load_development(args.corpus)
    handoff = [json.loads(line) for line in args.handoff.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    queries += [{**q, "id": "handoff:" + q["id"], "kind": "handoff", "gold": []} for q in handoff]
    if len({q["id"] for q in queries}) != len(queries) or any(not q["dense"].strip() for q in queries):
        raise ValueError("질의 ID 중복 또는 빈 질의")
    domains = {row.servId: set(row.app_domain.split("|")) - {"", "N"} for row in master.itertuples()}
    started = perf_counter()
    model = Embedder(MODELS[args.model], cache_dir=str(AI / "data/models"), device=args.device,
                     batch_size=args.batch_size, max_length=args.max_length, revision=args.revision)
    model_seconds = perf_counter() - started
    timings, metrics, all_ranks = [], [], []
    names = dict(zip(master.servId, master.servNm))
    for profile, texts in documents.items():
        print(f"[{args.model}/{profile}] 문서 {len(texts)}건 임베딩 시작 (최초 계산은 수분 소요 가능)", flush=True)
        before = perf_counter()
        suffix = "" if profile == "baseline" else "_" + profile
        index = DenseIndex(model, master.servId, texts, AI / "data" / f"dense_{args.model}{suffix}.npz")
        index_seconds = perf_counter() - before
        print(f"[{args.model}/{profile}] 색인 준비 완료, 질의 {len(queries)}개 검색 시작", flush=True)
        before = perf_counter()
        rankings = index.search_batch_with_scores([q["dense"] for q in queries], 100)
        query_seconds = perf_counter() - before
        for query, results in zip(queries, rankings):
            for rank, (sid, score) in enumerate(results, 1):
                all_ranks.append({"model": args.model, "profile": profile, "query_id": query["id"],
                                  "kind": query["kind"], "rank": rank, "servId": sid,
                                  "servNm": names[sid], "score": score})
        for rec in domain_metrics(rankings, queries, domains):
            metrics.append({"model": args.model, "profile": profile, **rec})
        timings.append({"model": args.model, "profile": profile, "cache_hit": index.cache_hit,
                        "documents": len(texts), "fingerprint": index.fingerprint,
                        "index_seconds": index_seconds, "query_seconds": query_seconds})
        print(f"[{args.model}/{profile}] 완료: 색인 {index_seconds:.1f}초, 검색 {query_seconds:.1f}초", flush=True)
    output.mkdir(parents=True)
    pd.DataFrame(queries).to_json(output / "queries.json", orient="records", force_ascii=False, indent=2)
    pd.DataFrame(all_ranks).to_csv(output / "rankings.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(metrics).to_csv(output / "per_query_proxy.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(timings).to_csv(output / "times.csv", index=False, encoding="utf-8-sig")
    meta = {"model": args.model, "revision": args.revision, "model_identity": model.cache_key,
            "profiles": PROFILES, "max_length": args.max_length, "batch_size": args.batch_size,
            "model_load_seconds": model_seconds,
            "input_sha256": {str(p.name): hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in [args.master, args.corpus, args.handoff]},
            "note": "개발 원문 질의의 영역 일치 참고 평가. 실제 Step 2 확정 세션·독립 테스트·관련성 정확도 평가가 아님."}
    (output / "metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def summarize(args):
    frames, proxies, times = [], [], []
    reference_meta, reference_queries = None, None
    for alias in MODELS:
        folder = args.out / alias
        meta = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
        query_records = json.loads((folder / "queries.json").read_text(encoding="utf-8"))
        settings = {key: meta[key] for key in ["input_sha256", "profiles", "max_length", "batch_size"]}
        if reference_meta is not None and (settings != reference_meta or query_records != reference_queries):
            raise ValueError("모델 간 입력·설정·질의가 다릅니다. 같은 실험끼리만 집계할 수 있습니다")
        reference_meta, reference_queries = settings, query_records
        frames.append(pd.read_csv(folder / "rankings.csv", dtype={"servId": str}))
        proxies.append(pd.read_csv(folder / "per_query_proxy.csv"))
        times.append(pd.read_csv(folder / "times.csv"))
    ranks, proxy = pd.concat(frames, ignore_index=True), pd.concat(proxies, ignore_index=True)
    summaries = []
    for scope in ["all_development", "supported_domains"]:
        subset = proxy if scope == "all_development" else proxy[proxy.supported_gold]
        for (model, profile), group in subset.groupby(["model", "profile"]):
            summaries.append({"scope": scope, "model": model, "profile": profile,
                              "queries": len(group), "k": 10,
                              "domain_precision_at_10": group.domain_precision.mean(),
                              "domain_hit_at_10": group.domain_hit.mean()})
    summary = pd.DataFrame(summaries)
    baseline = summary[summary.profile == "baseline"][["scope", "model", "domain_precision_at_10", "domain_hit_at_10"]]
    summary = summary.merge(baseline, on=["scope", "model"], suffixes=("", "_baseline"))
    for metric in ["domain_precision_at_10", "domain_hit_at_10"]:
        summary[metric + "_delta"] = summary[metric] - summary[metric + "_baseline"]
    tables = args.out / "tables"
    tables.mkdir(exist_ok=True)
    summary.to_csv(tables / "development_proxy_comparison.csv", index=False, encoding="utf-8-sig")
    by_domain = proxy.assign(domain=proxy.gold_domains.str.split("|")).explode("domain")
    by_domain.groupby(["model", "profile", "domain"]).agg(
        queries=("query_id", "count"), domain_precision_at_10=("domain_precision", "mean"),
        domain_hit_at_10=("domain_hit", "mean")).reset_index().to_csv(
            tables / "development_proxy_by_domain.csv", index=False, encoding="utf-8-sig")
    pd.concat(times).to_csv(tables / "run_times.csv", index=False, encoding="utf-8-sig")
    ranks[ranks.kind == "handoff"].to_csv(tables / "handoff_rankings.csv", index=False, encoding="utf-8-sig")
    master = pd.read_csv(args.master, dtype=str, encoding="utf-8-sig", keep_default_na=False).set_index("servId")
    if hashlib.sha256(args.master.read_bytes()).hexdigest() != reference_meta["input_sha256"][args.master.name]:
        raise ValueError("실행에 사용한 서비스마스터와 집계 입력이 다릅니다")
    query_text = {q["id"]: q["dense"] for q in reference_queries}
    handoff = ranks[(ranks.kind == "handoff") & (ranks["rank"] <= 5)]
    pool = []
    for (qid, sid), group in handoff.groupby(["query_id", "servId"]):
        row = master.loc[sid]
        pool.append({"query_id": qid, "query": query_text[qid], "servId": sid, "servNm": row.servNm,
                     "systems": "|".join(sorted(set(group.model + "/" + group.profile))),
                     "summary_text": row.summary_text, "benefit_text": row.benefit_text,
                     "target_text": row.target_text, "criteria_text": row.criteria_text,
                     "detail_url": row.detail_url, "relevance_0_1_2": "", "reason": "", "reviewer": ""})
    pd.DataFrame(pool).to_csv(tables / "handoff_relevance_review.csv", index=False, encoding="utf-8-sig")
    print(summary.to_string(index=False))
    print("참고 지표만으로 모델·기본 구성을 자동 선정하지 않음. 관련성 검토와 정답표 평가 필요.")


# ── 사람 관련성 검수 집계 (review) ──────────────────────────────────
REVIEW_K = 5
RELEVANCE_COL = "relevance_0_1_2"   # 사람 판정만 사용. ai_* 등 AI 초안 열은 읽지 않음
RELEVANCE_VALUES = {0: "무관", 1: "간접 관련", 2: "직접 관련"}
DIRECT = 2
# nDCG gain. discount는 모두 1/log2(rank+1). exp2(팀 평가 기준) = 2^판정-1 → 0·1·3, linear = 판정값 0·1·2
GAINS = {"exp2": lambda rel: 2 ** rel - 1, "linear": lambda rel: rel}
GAIN_TEXT = {"exp2": "2^relevance - 1 (0·1·3)", "linear": "판정값 그대로(0·1·2)"}
DEFAULT_GAIN = "exp2"
RANKING_COLS = ["model", "profile", "query_id", "rank", "servId"]
REVIEW_COLS = ["query_id", "servId", RELEVANCE_COL]
# 관련성과 합산하지 않고 따로 세는 검수 열 (있을 때만)
SIDE_COLS = ["eligibility_status", "access_status"]


class ReviewInputError(ValueError):
    """검수 입력 오류. errors: [{"check", "query_id", "servId", "detail"}]"""

    def __init__(self, errors):
        self.errors = errors
        lines = [f"- {e['check']}: query_id={e['query_id']} servId={e['servId']} {e['detail']}"
                 for e in errors[:50]]
        more = f"\n... 외 {len(errors) - 50}건" if len(errors) > 50 else ""
        super().__init__(f"검수 입력 오류 {len(errors)}건\n" + "\n".join(lines) + more)


def _err(check, query_id="", serv_id="", detail=""):
    return {"check": check, "query_id": query_id, "servId": serv_id, "detail": str(detail)}


def _read_str_csv(path):
    return pd.read_csv(path, dtype=str, encoding="utf-8-sig", keep_default_na=False)


def parse_relevance(value):
    """'0'·'1'·'2'(앞뒤 공백, 엑셀 저장 '2.0' 허용) → int. 공란·그 밖의 값은 None."""
    text = str(value).strip()
    try:
        number = float(text)
    except ValueError:
        return None
    if not math.isfinite(number) or not number.is_integer() or int(number) not in RELEVANCE_VALUES:
        return None
    return int(number)


def load_top_rankings(rankings, k=REVIEW_K, expected_models=None, expected_profiles=None):
    """순위표에서 모델·문서 구성·질의별 Top-k. 형식 오류는 ReviewInputError."""
    missing = [c for c in RANKING_COLS if c not in rankings]
    if missing:
        raise ReviewInputError([_err("순위표 필수 열 없음", detail=missing)])
    df = rankings.copy()
    if "kind" in df:
        df = df[df["kind"] == "handoff"]
    for col in ["model", "profile", "query_id", "servId"]:
        df[col] = df[col].str.strip()
    errors = []
    df["rank_num"] = pd.to_numeric(df["rank"], errors="coerce")
    bad_rank = df["rank_num"].isna() | (df["rank_num"] < 1) | (df["rank_num"] % 1 != 0)
    for row in df[bad_rank].itertuples():
        errors.append(_err("순위표 rank 오류", row.query_id, row.servId, f"{row.model}/{row.profile} rank={row.rank!r}"))
    df = df[~bad_rank]
    for col in ["model", "profile", "query_id", "servId"]:
        for row in df[df[col] == ""].itertuples():
            errors.append(_err(f"순위표 {col} 공란", row.query_id, row.servId, f"{row.model}/{row.profile} rank={row.rank}"))
    sys_q = ["model", "profile", "query_id"]
    for (m, p, q, r), _ in df[df.duplicated(sys_q + ["rank_num"], keep=False)].groupby(sys_q + ["rank_num"]):
        errors.append(_err("순위표 같은 순위 중복", q, "", f"{m}/{p} rank={int(r)}"))
    for (m, p, q, sid), _ in df[df.duplicated(sys_q + ["servId"], keep=False)].groupby(sys_q + ["servId"]):
        errors.append(_err("순위표 같은 서비스 중복", q, sid, f"{m}/{p}"))
    expected = set(zip(df["model"], df["profile"]))
    if expected_models and expected_profiles:
        expected |= {(m, p) for m in expected_models for p in expected_profiles}
    queries = sorted(set(df["query_id"]) - {""})
    if not queries:
        errors.append(_err("순위표 인계 질의 없음"))
    have = set(zip(df["model"], df["profile"], df["query_id"]))
    for m, p in sorted(expected):
        for q in queries:
            if (m, p, q) not in have:
                errors.append(_err("순위표 조합·질의 누락", q, "", f"{m}/{p}"))
    if errors:
        raise ReviewInputError(errors)
    top = df.sort_values(sys_q + ["rank_num"])
    top = top[top["rank_num"] <= k].copy()
    top["rank"] = top["rank_num"].astype(int)
    return top[sys_q + ["rank", "servId"]].reset_index(drop=True)


def load_review(review, top):
    """사람 판정표 검사 → {(query_id, servId): 판정}.

    키 중복·Top-k 판정 누락·판정 공란·범위 오류를 모두 모아 ReviewInputError로 보고한다.
    relevance_0_1_2 외 열(ai_relevance_draft 등 AI 초안, reason, reviewer)은 집계에 쓰지 않는다.
    """
    missing = [c for c in REVIEW_COLS if c not in review]
    if missing:
        raise ReviewInputError([_err("검수표 필수 열 없음", detail=missing)])
    errors = []
    df = review.assign(query_id=review["query_id"].str.strip(), servId=review["servId"].str.strip())
    line = lambda i: f"행 {i + 2}"   # noqa: E731  CSV 머리글 다음 줄부터 2행
    for row in df[(df["query_id"] == "") | (df["servId"] == "")].itertuples():
        errors.append(_err("검수표 키 공란", row.query_id, row.servId, line(row.Index)))
    keyed = df[(df["query_id"] != "") & (df["servId"] != "")]
    for (q, sid), g in keyed[keyed.duplicated(["query_id", "servId"], keep=False)].groupby(["query_id", "servId"]):
        errors.append(_err("검수 키 중복", q, sid, "행 " + ",".join(str(i + 2) for i in g.index)))
    labels = {}
    for i, q, sid, raw in zip(keyed.index, keyed["query_id"], keyed["servId"], keyed[RELEVANCE_COL]):
        value = parse_relevance(raw)
        if str(raw).strip() == "":
            errors.append(_err("판정 공란", q, sid, line(i)))
        elif value is None:
            errors.append(_err("판정값 범위 오류(0·1·2만 허용)", q, sid, f"{line(i)} 값={raw!r}"))
        else:
            labels[(q, sid)] = value
    keys = set(zip(keyed["query_id"], keyed["servId"]))
    for q, sid in sorted(set(zip(top["query_id"], top["servId"])) - keys):
        hit = top[(top["query_id"] == q) & (top["servId"] == sid)]
        errors.append(_err(f"Top-{REVIEW_K} 판정 누락", q, sid,
                           "|".join(sorted(set(hit["model"] + "/" + hit["profile"])))))
    if errors:
        raise ReviewInputError(errors)
    return labels


def dcg(gains):
    return math.fsum(g / math.log2(i + 1) for i, g in enumerate(gains, 1))


def review_metrics(top, labels, k=REVIEW_K, gain=DEFAULT_GAIN):
    """질의·조합별 Top-k 판정 분포, Precision@k(판정 2 비율), 평균 판정, 0점 비율, nDCG@k.

    이상적 순위: 질의별로 사람이 판정한 후보 풀 전체(모든 조합 Top-k 합집합 + 검수표의 같은 질의 행)를
    판정값 내림차순으로 놓은 Top-k. 이상적 DCG가 0(풀 전체 무관)이면 nDCG는 정의하지 않음(NaN, ideal_zero=True).
    """
    to_gain = GAINS[gain]
    pool = {}
    for (q, _), v in labels.items():
        pool.setdefault(q, []).append(v)
    ideal = {q: dcg([to_gain(r) for r in sorted(v, reverse=True)[:k]]) for q, v in pool.items()}
    rows = []
    for (m, p, q), g in top.groupby(["model", "profile", "query_id"], sort=True):
        rels = [labels[(q, sid)] for sid in g.sort_values("rank")["servId"]]
        idcg, got = ideal[q], dcg([to_gain(r) for r in rels])
        n_direct, n_zero = sum(v == DIRECT for v in rels), sum(v == 0 for v in rels)
        rows.append({"model": m, "profile": p, "query_id": q, "k": k, "n_ranked": len(rels),
                     "n_direct": n_direct, "n_indirect": sum(v == 1 for v in rels), "n_irrelevant": n_zero,
                     "relevance": "|".join(map(str, rels)),
                     "precision_at_k": n_direct / k, "mean_relevance": sum(rels) / len(rels),
                     "zero_ratio": n_zero / len(rels),
                     "dcg": got, "ideal_dcg": idcg, "ideal_zero": idcg == 0,
                     "ndcg": got / idcg if idcg > 0 else float("nan"), "gain": gain,
                     "pool_size": len(pool[q]), "pool_direct": sum(v == DIRECT for v in pool[q])})
    return pd.DataFrame(rows)


def review_summary(per_query):
    """조합별 평균. 질의 단위 지표 분모 = 질의 수, ndcg_at5_mean 분모 = 이상적 DCG > 0인 질의 수."""
    rows = []
    for (m, p), g in per_query.groupby(["model", "profile"], sort=True):
        defined = g[~g["ideal_zero"]]
        rows.append({"model": m, "profile": p, "k": REVIEW_K, "queries": len(g),
                     "ndcg_at5_mean": defined["ndcg"].mean() if len(defined) else float("nan"),
                     "ndcg_queries": len(defined), "ndcg_excluded_ideal_zero": int(g["ideal_zero"].sum()),
                     "precision_at5_mean": g["precision_at_k"].mean(),
                     "mean_relevance": g["mean_relevance"].mean(), "zero_ratio": g["zero_ratio"].mean(),
                     "direct_at5_mean": g["n_direct"].mean(), "direct_at5_sum": int(g["n_direct"].sum()),
                     "indirect_sum": int(g["n_indirect"].sum()), "irrelevant_sum": int(g["n_irrelevant"].sum()),
                     "ranked_items": int(g["n_ranked"].sum()),
                     "gain": g["gain"].iloc[0], "relevance_column": RELEVANCE_COL})
    return pd.DataFrame(rows).sort_values(["ndcg_at5_mean", "model", "profile"],
                                          ascending=[False, True, True]).reset_index(drop=True)


def side_counts(top, review_df):
    """신청 자격·접근 제약 상태를 조합별 Top-k 건수로 따로 센다. 관련성 점수와 합산하지 않음."""
    cols = [c for c in SIDE_COLS if c in review_df]
    if not cols:
        return pd.DataFrame()
    keyed = review_df.assign(query_id=review_df["query_id"].str.strip(), servId=review_df["servId"].str.strip())
    merged = top.merge(keyed[["query_id", "servId"] + cols], on=["query_id", "servId"], how="left")
    rows = []
    for col in cols:
        merged[col] = merged[col].fillna("").str.strip().replace("", "(공란)")
        for (m, p, v), g in merged.groupby(["model", "profile", col], sort=True):
            rows.append({"item": col, "model": m, "profile": p, "status": v, "count": len(g),
                         "ranked_items": int(((merged.model == m) & (merged.profile == p)).sum())})
    return pd.DataFrame(rows)


def review(args):
    """사람 검수 집계. 입력 CSV는 읽기만 하고, 결과는 --out/tables/human_review_*에 쓴다."""
    for path in (args.rankings, args.review):
        if not Path(path).is_file():
            raise FileNotFoundError(f"입력 파일 없음: {path}")
    out = args.out / "tables"
    per_query_path, summary_path = out / "human_review_per_query.csv", out / "human_review_summary.csv"
    errors_path, meta_path = out / "human_review_errors.csv", out / "human_review_metadata.json"
    side_path = out / "human_review_eligibility_access.csv"
    gain = getattr(args, "gain", None) or DEFAULT_GAIN
    inputs = {Path(args.rankings).resolve(), Path(args.review).resolve()}
    if any(t.resolve() in inputs for t in (per_query_path, summary_path, errors_path, meta_path, side_path)):
        raise ValueError("입력 파일을 덮어쓸 수 없습니다. 다른 --out을 지정하세요")
    try:
        top = load_top_rankings(_read_str_csv(args.rankings), expected_models=list(MODELS),
                                expected_profiles=list(PROFILES))
        review_df = _read_str_csv(args.review)
        labels = load_review(review_df, top)
    except ReviewInputError as e:
        out.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(e.errors).to_csv(errors_path, index=False, encoding="utf-8-sig")
        print(str(e))
        print(f"오류 목록: {errors_path}")
        return 1
    per_query = review_metrics(top, labels, gain=gain)
    summary = review_summary(per_query)
    side = side_counts(top, review_df)
    out.mkdir(parents=True, exist_ok=True)
    per_query.to_csv(per_query_path, index=False, encoding="utf-8-sig")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    if len(side):
        side.to_csv(side_path, index=False, encoding="utf-8-sig")
    elif side_path.exists():
        side_path.unlink()   # 이번 검수표에 자격·접근 열이 없으면 이전 결과를 남기지 않음
    if errors_path.exists():
        errors_path.unlink()   # 이전 실패 기록 정리 (이번 입력은 오류 없음)
    top_keys = set(zip(top["query_id"], top["servId"]))
    meta = {"rankings": Path(args.rankings).name, "review": Path(args.review).name,
            "input_sha256": {Path(p).name: hashlib.sha256(Path(p).read_bytes()).hexdigest()
                             for p in (args.rankings, args.review)},
            "k": REVIEW_K, "relevance_column": RELEVANCE_COL, "relevance_values": RELEVANCE_VALUES,
            "ndcg": {"gain": GAIN_TEXT[gain], "discount": "1/log2(rank+1)",
                     "ideal": "질의별 사람 판정 후보 풀 전체를 판정값 내림차순으로 놓은 Top-5",
                     "ideal_zero": "이상적 DCG 0인 질의는 nDCG 미정의(NaN), 평균에서 제외, ndcg_excluded_ideal_zero에 개수 기록"},
            "systems": sorted({f"{m}/{p}" for m, p in zip(top["model"], top["profile"])}),
            "queries": sorted(set(top["query_id"])), "judged_pairs": len(labels),
            "top5_pairs": len(top_keys), "judged_not_in_top5": len(set(labels) - top_keys),
            "note": "인계 질의 사람 판정 기술 집계. 질의 수가 적어 모델·구성 우열의 통계적 근거로 쓰지 않음"}
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))
    print("인계 질의 기준 기술 집계. 모델 선정·운영 설정 변경은 팀 검토 후 결정.")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("command", choices=["run", "summarize", "review"])
    ap.add_argument("--model", choices=MODELS)
    ap.add_argument("--revision")
    ap.add_argument("--device", default="cuda", choices=["cpu", "cuda", "mps"])
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--max-length", type=int, default=8192)
    ap.add_argument("--master", type=Path, default=ROOT / "data/processed/services_master.csv")
    ap.add_argument("--corpus", type=Path, default=AI / "data/dev_corpus.jsonl")
    ap.add_argument("--handoff", type=Path, default=AI / "data/dense_handoff_queries.jsonl")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--rankings", type=Path, help="review: 실험 ZIP의 tables/handoff_rankings.csv")
    ap.add_argument("--review", type=Path, help="review: 사람 판정 CSV (query_id, servId, relevance_0_1_2)")
    ap.add_argument("--gain", choices=sorted(GAINS), default=DEFAULT_GAIN,
                    help="review: nDCG gain. exp2 = 2^판정-1 (기본), linear = 판정값")
    args = ap.parse_args()
    if args.batch_size < 1 or not 1 <= args.max_length <= 8192:
        ap.error("batch-size 양수, max-length 1~8192 필요")
    if args.command == "run" and (not args.model or not args.revision):
        ap.error("run에는 --model, --revision 필요")
    if args.command == "review":
        if not args.rankings or not args.review:
            ap.error("review에는 --rankings, --review 필요")
        sys.exit(review(args))
    (run if args.command == "run" else summarize)(args)


if __name__ == "__main__":
    main()
