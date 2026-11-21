"""전체 흐름을 순서대로 이어주는 곳. 각 단계는 자기 파일에 있고, 여기서는 '부르기'만 한다.

대화형 흐름 (2단계 사용자 확인 포함)
  session = p.prepare(text, profile)     # 0 위기 → 1 추출 → 확인 세션 (검색하지 않음)
  session = p.resume(session, action)    # 확인 화면 조작, 질문 답 (status가 ready가 될 때까지 반복)
  result  = p.search(session)            # ready일 때만 쿼리 → BM25·Dense → RRF → 규칙 → 혜택 매칭 → 설명
검색만 확인: p.search_queries(bm25_query, dense_query)   (3·4번 담당용)
  반환 bm25·dense·fused(기존 키) + status(ok|empty|partial|error)·errors·rrf(서비스별 순위·점수)·modules
  검색 결과 없음(empty)과 검색 오류(error)를 구분한다. 오류면 규칙·설명을 부르지 않고 카드 없이 돌려준다.
한 번에 실행: p.run(text)  (사용자 입력 없이 진행. 사용자 확인으로 처리하지 않음)

실행:  python pipeline.py --mock "취업은 하고 싶은데 밖에 나가는 게 부담스러워요."
       (--mock을 빼면 config.yaml의 진짜 모델을 불러온다. Colab GPU에서 실행. 레포 루트·ai/ 어디서 실행해도 같다)
"""
import argparse
import copy
import json
import os
import time
from pathlib import Path

import yaml

from core.models import load_models
from core.review_data import RecommendationReview, configured_addendum_path
from safety.crisis import check_crisis
from safety.guidance import crisis_guidance
from extract.extract import extract
from retrieve.benefit import rank_by_benefit_type
from retrieve.bm25 import BM25Index, build_texts as build_bm25_texts
from retrieve.dense import DenseIndex, cache_path_for as dense_cache_path
from retrieve.dense_texts import build_texts as build_dense_texts
from retrieve.fusion import check_params, search_and_fuse
from retrieve.relevance import select_relevant
from retrieve.rrf import DEFAULT_TOP_N
from rules.applicant import filter_applicant_conflicts
from rules.filter import apply_rules
from rules.normalize import build_sgg_map, use_sgg_map, valid_sgg_for
from rules.period import korea_today, parse_reference_date
from explain.explain import explain
from confirm import questions as Q
from confirm.session import new_session, rules_view
from query.build import build_queries, load_terms, strip_avoid
from extract.postprocess import sentences as split_sentences

AI_DIR = Path(__file__).resolve().parent   # config의 상대 경로는 실행 위치(cwd)가 아니라 이 폴더(ai/) 기준
DEFAULT_CONFIG = AI_DIR / "configs" / "config.yaml"


def _resolve(path):
    """상대 경로는 ai/ 기준 절대 경로로. 이미 절대 경로면 그대로."""
    p = Path(path)
    return str(p if p.is_absolute() else (AI_DIR / p).resolve())


class Pipeline:
    def __init__(self, config_path=None, mock=False, services_path=None, *, q1_review_path=None, q2_info_path=None,
                 as_of=None):
        """services_path로 마스터를 바꿀 수 있음. as_of는 평가용이며 기본은 검색 시점의 한국 날짜."""
        self.as_of = parse_reference_date(as_of) if as_of is not None else None
        config_path = Path(config_path) if config_path else DEFAULT_CONFIG
        if not config_path.is_absolute() and not config_path.exists():
            config_path = AI_DIR / config_path
        self.cfg = cfg = yaml.safe_load(open(config_path, encoding="utf-8"))
        # 경로는 ai/ 기준 절대 경로로 고정. 모델 캐시 위치는 환경변수 MODEL_CACHE_DIR가 config보다 우선
        p = cfg["paths"] = {k: (_resolve(v) if k != "cache_dir" and isinstance(v, str) else v)
                            for k, v in cfg["paths"].items()}
        if services_path:
            p["services"] = str(Path(services_path).resolve())
        p["cache_dir"] = os.environ.get("MODEL_CACHE_DIR") or p.get("cache_dir") or None

        # The bundled config points to locally generated files. Custom config paths are explicit.
        review_options = {"custom_master": services_path is not None, "optional": config_path.resolve() == DEFAULT_CONFIG}
        q1_path = configured_addendum_path(p.get("q1_review"), q1_review_path, **review_options)
        q2_path = configured_addendum_path(p.get("q2_info"), q2_info_path, **review_options)
        if os.environ.get("BANGEOL_REQUIRE_REVIEW_DATA") == "1" and (q1_path is None or q2_path is None):
            raise FileNotFoundError("최종 Q1/Q2 검수 데이터가 필요합니다. 저장소 루트에서 python ai/scripts/prepare_runtime.py를 실행해 주세요.")
        self.review = RecommendationReview(
            p["services"],
            q1_path=q1_path,
            q2_path=q2_path,
        )

        # ① 모델 불러오기 (여기서 한 번만)
        self.embedder, self.llm = load_models(cfg, mock)

        # ② 팀 규칙·영역 읽기
        self.domains = yaml.safe_load(open(p["domains_file"], encoding="utf-8"))["domains"]
        self.rules_text = open(p["rules_file"], encoding="utf-8").read()

        # ③ 서비스 목록 + 검색 색인 (임베딩은 캐시 파일로 재사용)
        self.services = self.review.services
        self.sgg_map = build_sgg_map(self.services, p["services"])
        # BM25와 Dense는 같은 id 목록(services.index, 마스터 servId 순서)을 받는다. 각 텍스트 구성은 따로다.
        ids = list(self.services.index)
        texts = build_dense_texts(self.services, cfg["retrieval"].get("dense_profile", "baseline"))
        self.bm25 = BM25Index(ids, build_bm25_texts(self.services),  # 검색 칸은 bm25.py BM25_FIELDS
                              cache_path=None if mock else p.get("bm25_cache"))
        # 문서 구성별 캐시 파일(config `{profile}`)로 baseline·core 전환 시 서로 덮어쓰지 않음
        cache = None if mock else dense_cache_path(p.get("dense_cache"),
                                                   cfg["retrieval"].get("dense_profile", "baseline"))
        self.dense = DenseIndex(self.embedder, ids, texts, cache)
        if list(self.bm25.ids) != list(self.dense.ids) or list(self.bm25.ids) != ids:
            raise ValueError("BM25와 Dense의 서비스 ID 순서가 다릅니다")

        # ④ few-shot 예시 (개발 세트)
        self.examples = []
        if os.path.exists(p["dev_examples"]):
            rows = [json.loads(l) for l in open(p["dev_examples"], encoding="utf-8") if l.strip()]
            self.examples = [r for r in rows if r.get("label")]  # 위기 문장 등 예시 라벨이 없는 행은 뺀다
        self.example_emb = self.embedder.encode([e["text"] for e in self.examples]) if self.examples else None

        # ⑤ 2단계 쿼리 사전
        self.terms = load_terms(p.get("query_terms") or _resolve("configs/query_terms.yaml"))

    # ── 1단계 호출 (2단계 질문의 자유 입력에도 같은 함수를 씀) ──────────────
    def _crisis(self, text):
        return check_crisis(text, self.llm)

    def _extract(self, text, cfg_extract=None):
        return extract(text, self.llm, self.embedder, self.rules_text, self.domains,
                       self.examples, self.example_emb, cfg_extract or self.cfg["extract"])

    # ── 2단계 대화형 흐름 ──────────────────────────────────────────
    def prepare(self, text, profile=None):
        """0 위기 확인 → 1 추출 → 확인 세션. 검색하지 않는다. 위기면 추출도 하지 않는다."""
        crisis = self._crisis(text)
        if crisis["crisis"]:
            return new_session(text, crisis, None, profile)
        return new_session(text, crisis, self._extract(text), profile)

    def resume(self, session, action):
        """확인 화면 조작·질문 답 하나를 반영한 새 세션. 검색하지 않는다."""
        with use_sgg_map(self.sgg_map):
            return Q.step(session, action, cfg=self.cfg, domains=self.domains,
                          crisis_fn=self._crisis, extract_fn=self._extract)

    def search(self, session, *, explain_mode="model", use_benefit_rerank=True, use_eligibility_gate=True,
               use_relevance_gate=True):
        """status가 ready일 때만 검색한다. 그 밖의 상태는 검색 없이 그대로 돌려준다."""
        if session.get("status") != "ready":
            out = {"status": session.get("status"), "searched": False, "crisis": session.get("crisis"),
                   "session": session, "cards": [], "retrieval_status": "not_searched", "retrieval_errors": []}
            if out["status"] == "crisis_stop":   # 추천 중단 + 공식 도움 경로(정적 설정, LLM 생성 아님)
                out["guidance"] = crisis_guidance()
            return out
        if explain_mode not in ("model", "source_first"):
            raise ValueError("알 수 없는 설명 모드")
        search_started = time.perf_counter()
        timings = {}
        stage_started = search_started
        q = build_queries(session, self.terms, self.cfg)
        timings["query_ms"] = round((time.perf_counter() - stage_started) * 1000, 1)
        s = copy.deepcopy(session)
        s["queries"] = {k: q[k] for k in ("bm25", "bm25_tokens", "bm25_search_tokens", "dense",
                                         "query_mode", "access_mode")}
        s["query_expansion"] = q["query_expansion"]
        s["access"] = q["access"]
        s["access_inferred"] = q["access_inferred"]

        # 규칙·설명에는 사용자 최종 확정값을 쓴다. 나이·지역은 rules_pass 설정에 따라 차단된 사본.
        view = rules_view(s["confirmed_profile"], self.cfg, s["access"])
        with use_sgg_map(self.sgg_map):
            view["region_sgg"] = valid_sgg_for(view.get("region_sido"), view.get("region_sgg"))
        view["today"] = (self.as_of or korea_today()).isoformat()
        stage_started = time.perf_counter()
        found = self.search_queries(q["bm25"], q["dense"])
        timings["retrieval_ms"] = round((time.perf_counter() - stage_started) * 1000, 1)
        retrieval = {"retrieval_status": found["status"], "retrieval_errors": found["errors"]}
        if found["status"] == "error":
            # 검색 실패는 '맞는 서비스 없음'으로 보이지 않게 따로 표시. 규칙·설명은 부르지 않는다(후보 없음)
            return {"status": "ready", "searched": True, "crisis": s["crisis"], "session": s,
                    "queries": s["queries"], "ranks": found, "rules_view": view, "cards": [],
                    "excluded": [], "timings_ms": {**timings, "total_ms": round((time.perf_counter() - search_started) * 1000, 1)}, **retrieval}
        stage_started = time.perf_counter()
        excluded = []   # 제외 서비스와 이유 기록(카드에는 넣지 않고 결과에 따로 둠)
        eligible = apply_rules(found["fused"], self.services, view,
                               {"source_limit_checks": True, "prioritize_verified_use": True}
                               if use_eligibility_gate else None, excluded=excluded)
        text = s.get("original_text") or ""
        answers = [q["answer"] for q in s.get("questions", [])
                   if q.get("slot") == "need" and q.get("answer_type") == "자유" and q.get("answer")]
        if use_eligibility_gate:
            eligible = filter_applicant_conflicts(eligible, self.services, text, excluded=excluded)
        limit = self.cfg["retrieval"]["final_top_n"]
        if use_benefit_rerank:
            eligible, _ = rank_by_benefit_type(eligible, text, self.services, confirmed_needs=view["needs"],
                                               need_answers=answers, top_n=limit)
        relevance_decisions = []
        judged = (select_relevant(eligible, text, self.services, view["needs"], found, limit, need_answers=answers,
                                  decisions=relevance_decisions)
                  if use_relevance_gate else eligible[:limit])
        recommendation_status = ("recommended" if judged else "insufficient_relevance" if eligible
                                 else "no_eligible_candidates")
        timings["rules_ms"] = round((time.perf_counter() - stage_started) * 1000, 1)
        stage_started = time.perf_counter()
        explain_view = {"needs": copy.deepcopy(s["confirmed_profile"]["needs"]),
                        "activity_range": s["confirmed_profile"]["activity_range"]}
        cards = []
        for j in judged:
            svc = self.services.loc[j["servId"]].to_dict()
            if explain_mode == "source_first":
                card = {"reason": svc.get("summary_text") or svc.get("benefit_text") or "상세 화면에서 서비스 안내 원문을 확인해 주세요.",
                        "how_to_apply": svc.get("apply_text", ""),
                        "check_needed": [k for k, v in j["checks"].items() if v == "확인 필요"],
                        "evidence": svc.get("target_text", ""), "fallback": False,
                        "explanation_pending": True, "explanation_kind": "source"}
            else:
                card = explain(svc, explain_view, j["checks"], self.llm)
            cards.append({"servId": j["servId"], "servNm": svc.get("servNm"), **card, "checks": j["checks"],
                          "check_reasons": j.get("check_reasons", {}),
                          **self.review.card_fields(j["servId"])})
        timings["explanation_ms"] = round((time.perf_counter() - stage_started) * 1000, 1)
        timings["total_ms"] = round((time.perf_counter() - search_started) * 1000, 1)
        return {"status": "ready", "searched": True, "crisis": s["crisis"], "session": s,
                "queries": s["queries"], "ranks": found, "rules_view": view, "cards": cards,
                "excluded": excluded, "relevance_decisions": relevance_decisions,
                "timings_ms": timings, "recommendation_status": recommendation_status,
                **retrieval}

    def explain_service(self, session, result, serv_id):
        """Explain only a service already returned for this confirmed session."""
        if session.get("status") != "ready" or result.get("status") != "ready" or not result.get("searched"):
            raise ValueError("검색을 마친 세션에서만 설명을 요청할 수 있습니다")
        searched = result.get("session") or {}
        if (session.get("confirmed_profile") != searched.get("confirmed_profile")
                or session.get("original_text") != searched.get("original_text")):
            raise ValueError("검색 조건이 바뀌었습니다. 다시 검색해 주세요")
        card = next((c for c in result.get("cards", []) if c["servId"] == serv_id), None)
        if card is None:
            raise ValueError("현재 검색 결과에 없는 서비스입니다")
        if not card.get("explanation_pending"):
            return copy.deepcopy(card)
        profile = session["confirmed_profile"]
        view = {"needs": copy.deepcopy(profile["needs"]), "activity_range": profile["activity_range"]}
        generated = explain(self.services.loc[serv_id].to_dict(), view, card["checks"], self.llm)
        return {**card, **generated, "explanation_pending": False,
                "explanation_kind": "source" if generated.get("fallback") else "model"}

    def _retrieval_params(self):
        """config 값 검사. 설정 오류는 검색 실패로 숨기지 않고 바로 ValueError."""
        r = self.cfg["retrieval"]
        return check_params(r["top_k_each"], r["rrf_k"], r.get("rrf_top_n", DEFAULT_TOP_N),
                            weights=r.get("rrf_weights"))

    def explore(self, session, limit=None):
        """욕구가 확정되지 않은 세션(explore_only 등)의 탐색용 후보 servId 목록. 맞춤 추천 순위가 아니다.

        원문에서 회피 절(밖에 못 나가요 등)을 뺀 문장으로 BM25·Dense 검색 후 자격·접근 규칙만 적용한다.
        욕구가 없으므로 관련성 게이트·혜택 재순위는 쓰지 않는다. 위기 세션·검색 실패면 빈 목록.
        """
        if session.get("status") == "crisis_stop":
            return []
        text = session.get("original_text") or ""
        query = " ".join(t for t in (strip_avoid(s)[0] for s in split_sentences(text, split_commas=False)) if t)
        if not query.strip():
            return []
        found = self.search_queries(query, query)
        if found["status"] == "error":
            return []
        view = rules_view({**(session.get("confirmed_profile") or {}), "needs": []}, self.cfg, [])
        with use_sgg_map(self.sgg_map):
            view["region_sgg"] = valid_sgg_for(view.get("region_sido"), view.get("region_sgg"))
        view["today"] = (self.as_of or korea_today()).isoformat()
        eligible = apply_rules(found["fused"], self.services, view,
                               {"source_limit_checks": True, "prioritize_verified_use": True})
        eligible = filter_applicant_conflicts(eligible, self.services, text)
        return [r["servId"] for r in eligible[:limit or self.cfg["retrieval"]["final_top_n"]]]

    def search_queries(self, bm25_query, dense_query):
        """검색만: BM25에는 bm25_query, Dense에는 dense_query를 각각 넣고 RRF로 합친다.

        반환(기존 키 bm25·dense·fused 유지, 나머지는 추가)
          bm25, dense : 각 검색의 서비스 ID 순위 목록. 실패한 검색은 [] (errors에 원인)
          fused       : RRF 통합 순위 서비스 ID 목록 (rrf_top_n개까지)
          status      : ok(모두 성공·후보 있음) | empty(모두 성공·후보 없음)
                        | partial(일부 실패·남은 검색으로 후보 있음) | error(후보 없음 + 실패 있음)
          errors      : [{module, type, message}]
          rrf         : [{servId, rrf_rank, rrf_score, ranks{bm25,dense}, contrib{bm25,dense}}]
          modules     : {bm25|dense: {ok, count, duplicates, query_empty, ms}, rrf: {ms, count}}
          params      : top_k_each, rrf_k, rrf_top_n
        검색 후보 생성만 한다. 자격·접근성 판단은 규칙 단계(apply_rules)가 한다.
        """
        retrievers = {"bm25": self.bm25, "dense": self.dense}   # RRF에 넣는 이름·순서. 가중치 없음
        return search_and_fuse(retrievers, {"bm25": bm25_query, "dense": dense_query},
                               self._retrieval_params(), known_ids=self.services.index)

    def run(self, text, profile=None):
        """사용자 입력 없이 진행. confirmation='none'(사용자 확인으로 처리하지 않음), '확인 필요' 영역 미사용.
        검색 가능 영역이 없으면 explore_only로 검색하지 않는다."""
        s = Q.auto_finish(self.prepare(text, profile))
        out = self.search(s)
        out["extracted"] = s.get("extracted")
        return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("text")
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--services", help="서비스 표 경로(기본: config paths.services)")
    ap.add_argument("--q1-review", help="선택한 서비스 마스터에 연결할 Q1 검수 JSON")
    ap.add_argument("--q2-info", help="선택한 서비스 마스터에 연결할 Q2 출처 비교 JSON")
    ap.add_argument("--as-of", help="평가용 고정 기준일 (YYYY-MM-DD)")
    args = ap.parse_args()
    out = Pipeline(mock=args.mock, services_path=args.services,
                   q1_review_path=args.q1_review, q2_info_path=args.q2_info, as_of=args.as_of).run(args.text)
    print(json.dumps(out, ensure_ascii=False, indent=2))
