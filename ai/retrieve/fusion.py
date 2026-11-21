"""3단계 검색 실행 + RRF 결합. pipeline.search_queries와 비교 스크립트가 같은 함수를 쓴다.

- 검색 모듈은 search(query, k) → 서비스 ID 순위 목록만 지키면 된다(BM25·Dense 모델·문서 구성 무관).
- 한 검색이 예외를 내거나 잘못된 목록을 주면 그 검색만 실패로 기록하고 나머지로 결합한다.
- 설정 오류(k·top_n·top_k_each)는 검색 실패로 숨기지 않고 ValueError.
- 후보 생성만 한다. 자격·접근성 판단은 규칙 단계(rules.filter.apply_rules) 몫.
"""
import logging
from time import perf_counter

from retrieve.rrf import DEFAULT_TOP_N, rrf_detail, validate_rank_list

log = logging.getLogger("retrieve.fusion")

# status 값
OK, EMPTY, PARTIAL, ERROR = "ok", "empty", "partial", "error"


def check_params(top_k_each, k, top_n=DEFAULT_TOP_N, weights=None):
    if isinstance(top_k_each, bool) or not isinstance(top_k_each, int) or top_k_each < 1:
        raise ValueError(f"top_k_each는 1 이상의 정수여야 합니다: {top_k_each!r}")
    rrf_detail([], k=k, top_n=top_n, weights=weights)   # k·top_n·weights 형식 검사
    params = {"top_k_each": top_k_each, "rrf_k": k, "rrf_top_n": top_n}
    if weights is not None:   # 가중치를 쓸 때만 키 추가(기존 params 동일)
        params["rrf_weights"] = dict(weights)
    return params


def run_one(name, retriever, query, top_k, known_ids=None):
    """검색 하나. (ids, info, error|None). Exception만 잡는다(KeyboardInterrupt 등은 그대로 전파)."""
    started = perf_counter()
    info = {"ok": False, "count": 0, "duplicates": 0,
            "query_empty": not (isinstance(query, str) and query.strip())}
    error = None
    try:
        ids = retriever.search(query, top_k)
        cleaned, dup = validate_rank_list(ids, name)
        if known_ids is not None:
            unknown = [sid for sid, _ in cleaned if sid not in known_ids]
            if unknown:
                raise ValueError(f"서비스 마스터에 없는 ID {len(unknown)}건: {unknown[:3]}")
        ids = list(ids)
        info.update(ok=True, count=len(cleaned), duplicates=dup)
    except Exception as e:  # noqa: BLE001  검색 모듈 실패를 기록하고 나머지 검색으로 진행
        ids = []
        error = {"module": name, "type": type(e).__name__, "message": str(e)}
        log.warning("%s 검색 실패: %s: %s", name, error["type"], error["message"], exc_info=True)
    info["ms"] = round((perf_counter() - started) * 1000, 3)
    return ids, info, error


def search_and_fuse(retrievers, queries, params, known_ids=None):
    """retrievers·queries: {이름: 검색 모듈}, {이름: 질의}. 이름 순서가 결과 키 순서.

    반환: {이름: ids..., fused, status, errors, rrf, modules, params}
      status ok(모두 성공·후보 있음) | empty(모두 성공·후보 없음)
             | partial(일부 실패·남은 검색으로 후보 있음) | error(후보 없음 + 실패 있음)
    """
    names = list(retrievers)
    lists, modules, errors = {}, {}, []
    for name in names:
        lists[name], modules[name], err = run_one(name, retrievers[name], queries.get(name),
                                                  params["top_k_each"], known_ids)
        if err:
            errors.append(err)
    started = perf_counter()
    ok_lists = {n: lists[n] for n in names if modules[n]["ok"]}
    detail = rrf_detail(ok_lists, k=params["rrf_k"], top_n=params["rrf_top_n"],
                       weights=params.get("rrf_weights")) if ok_lists else []
    for row in detail:   # 실패한 검색도 ranks·contrib에 None·0으로 보이게
        row["ranks"] = {n: row["ranks"].get(n) for n in names}
        row["contrib"] = {n: row["contrib"].get(n, 0.0) for n in names}
    modules["rrf"] = {"ms": round((perf_counter() - started) * 1000, 3), "count": len(detail)}
    fused = [row["servId"] for row in detail]
    if not errors:
        status = OK if fused else EMPTY
    else:
        status = PARTIAL if fused else ERROR
    return {**lists, "fused": fused, "status": status, "errors": errors,
            "rrf": detail, "modules": modules, "params": dict(params)}
