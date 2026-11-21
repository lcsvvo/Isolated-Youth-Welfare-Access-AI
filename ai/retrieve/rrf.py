"""3단계 RRF: 여러 순위 목록을 순위만 보고 합친다.

점수: score(d) = Σ_i 1 / (k + rank_i(d)),  rank는 1부터. 목록에 없는 검색은 0점 기여.
- 기본 가중치 없음(1:1). 선택 weights={검색 이름: 배수}면 기여가 w/(k+rank)가 됨(없는 이름은 1.0).
  검색 모델·문서 구성이 바뀌어도 서비스 ID 순위 목록만 받으므로 이 파일은 바뀌지 않는다.
- 같은 목록 안 중복 ID: 가장 앞(최고) 순위 한 번만 반영. 뒤쪽 중복은 버리고 그 자리 순위는 비우지 않는다
  (원래 순위 번호를 그대로 쓴다. 중복 하나 때문에 뒤 서비스 순위가 당겨지지 않게).
- 동점: ① RRF 점수 높은 순 ② 어느 한 목록에서의 최고 순위가 앞선 순 ③ 서비스 ID 문자열 순.
  목록 순서(BM25 먼저/Dense 먼저)를 바꿔도 결과가 같다.
- 점수 합은 math.fsum으로 계산해 더하는 순서에 따른 부동소수 차이가 동점 판정을 흔들지 않게 한다.
- 잘못된 입력(k·top_n·ID·목록 형식)은 조용히 넘기지 않고 ValueError.
"""
import math
import numbers
from collections.abc import Mapping

__all__ = ["rrf", "rrf_detail", "validate_rank_list", "DEFAULT_K", "DEFAULT_TOP_N"]

DEFAULT_K = 60
DEFAULT_TOP_N = 100


def _check_k(k):
    if isinstance(k, bool) or not isinstance(k, (int, float)) or not math.isfinite(k) or k < 0:
        raise ValueError(f"RRF k는 0 이상의 유한한 숫자여야 합니다: {k!r}")


def _check_top_n(top_n):
    if top_n is None:
        return
    if isinstance(top_n, bool) or not isinstance(top_n, int) or top_n < 1:
        raise ValueError(f"top_n은 1 이상의 정수 또는 None(전체)이어야 합니다: {top_n!r}")


def _check_id(sid, name, pos):
    """서비스 ID는 빈 값이 아닌 문자열 또는 정수(numpy 정수 포함, bool 제외)."""
    ok = (isinstance(sid, str) and sid.strip() != "") or (isinstance(sid, numbers.Integral) and not isinstance(sid, bool))
    if not ok:
        raise ValueError(f"잘못된 서비스 ID: 목록 {name!r}의 {pos}번째 값 {sid!r}")


def validate_rank_list(ranks, name="?"):
    """한 검색 결과 목록 검사 후 (중복 제거된 [(id, 원래 순위)], 버린 중복 수) 반환."""
    if isinstance(ranks, (str, bytes)) or isinstance(ranks, Mapping) or not hasattr(ranks, "__iter__"):
        raise ValueError(f"검색 결과 목록 {name!r}은 서비스 ID의 리스트여야 합니다: {type(ranks).__name__}")
    seen, out, dup = set(), [], 0
    for r, sid in enumerate(ranks, start=1):
        _check_id(sid, name, r)
        if sid in seen:
            dup += 1
            continue
        seen.add(sid)
        out.append((sid, r))
    return out, dup


def _named(rank_lists):
    """Mapping이면 {이름: 목록}, 시퀀스면 위치 번호를 이름으로."""
    if isinstance(rank_lists, Mapping):
        items = list(rank_lists.items())
    elif isinstance(rank_lists, (str, bytes)) or not hasattr(rank_lists, "__iter__"):
        raise ValueError("rank_lists는 순위 목록들의 리스트 또는 {이름: 목록} 딕셔너리여야 합니다")
    else:
        items = [(str(i), v) for i, v in enumerate(rank_lists)]
    names = [n for n, _ in items]
    if len(set(names)) != len(names):
        raise ValueError(f"검색 결과 이름이 중복되었습니다: {names}")
    return items


def _check_weights(weights):
    """None 또는 {이름: 0보다 큰 유한 실수}. 아니면 ValueError."""
    if weights is None:
        return {}
    if not isinstance(weights, Mapping):
        raise ValueError(f"weights는 {{이름: 배수}} 딕셔너리 또는 None이어야 합니다: {type(weights).__name__}")
    for n, w in weights.items():
        if isinstance(w, bool) or not isinstance(w, numbers.Real) or not math.isfinite(w) or w <= 0:
            raise ValueError(f"weights 값은 0보다 큰 유한한 숫자여야 합니다: {n!r}={w!r}")
    return dict(weights)


def rrf_detail(rank_lists, k=DEFAULT_K, top_n=DEFAULT_TOP_N, weights=None):
    """서비스별 RRF 점수·통합 순위·검색별 원래 순위(없으면 None).

    weights: None 또는 {이름: 배수}. 없는 이름은 1.0, 결합 대상이 아닌 이름은 무시.

    반환: [{"servId", "rrf_rank", "rrf_score", "ranks": {이름: 순위|None}, "contrib": {이름: 점수}}]
    rank_lists가 {"bm25": [...], "dense": [...]}이면 ranks 키도 bm25·dense. 리스트면 "0","1",...
    """
    _check_k(k)
    _check_top_n(top_n)
    w_map = _check_weights(weights)
    items = _named(rank_lists)
    per = {}       # sid -> {name: rank}
    order = []     # 처음 본 순서(정렬 기준 아님, 결과 키 순서 보존용)
    for name, ranks in items:
        cleaned, _ = validate_rank_list(ranks, name)
        for sid, r in cleaned:
            if sid not in per:
                per[sid] = {}
                order.append(sid)
            per[sid][name] = r
    names = [n for n, _ in items]
    rows = []
    for sid in order:
        contrib = {n: (1.0 / (k + r) if n not in w_map or w_map[n] == 1 else w_map[n] / (k + r))
                   for n, r in per[sid].items()}
        rows.append({"servId": sid, "rrf_score": math.fsum(contrib.values()),
                     "ranks": {n: per[sid].get(n) for n in names},
                     "contrib": {n: contrib.get(n, 0.0) for n in names},
                     "_best": min(per[sid].values())})
    rows.sort(key=lambda x: (-x["rrf_score"], x["_best"], str(x["servId"])))
    rows = rows if top_n is None else rows[:top_n]
    for i, x in enumerate(rows, start=1):
        del x["_best"]
        x["rrf_rank"] = i
    return [{"servId": x["servId"], "rrf_rank": x["rrf_rank"], "rrf_score": x["rrf_score"],
             "ranks": x["ranks"], "contrib": x["contrib"]} for x in rows]


def rrf(rank_lists, k=DEFAULT_K, top_n=DEFAULT_TOP_N, weights=None):
    """기존 인터페이스: 통합 순위대로 서비스 ID 리스트."""
    return [x["servId"] for x in rrf_detail(rank_lists, k=k, top_n=top_n, weights=weights)]
