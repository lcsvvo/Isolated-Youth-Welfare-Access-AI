# AI 파이프라인 기본 틀

## 핵심 개념: 모델은 "합치는" 게 아니라 "불러서 쓰는" 것
모델(BGE-M3, Kanana)은 코드 안에 들어가는 게 아니라, `core/models.py`에서 한 번 불러온 뒤
각 단계가 `embedder.encode(...)`, `llm.chat_json(...)`만 호출합니다.
그래서 모델을 바꾸려면 `configs/config.yaml`의 이름만 바꾸면 됩니다.

## 흐름과 파일
| 단계 | 파일 | 쓰는 모델 |
|---|---|---|
| 0 위기 확인 | safety/crisis.py | 키워드 + Kanana |
| 1 욕구·조건 추출 | extract/extract.py | Kanana (+ BGE-M3로 예시 고르기) |
| 2·3 검색 + RRF | retrieve/bm25.py, dense.py, rrf.py | BGE-M3 |
| 4 규칙 검증 | rules/filter.py | 없음 (코드) |
| 5 설명 생성 | explain/explain.py | Kanana |
| 전체 연결 | pipeline.py | — |

## 실행
서비스 표는 `configs/config.yaml`의 `paths.services`(기본 `../data/processed/services_master.csv`, 5,288건)를 `core/data.py`의 `load_services`로 읽음.
상대 경로는 실행 위치가 아니라 `ai/` 기준이라 레포 루트·`ai/` 어디서 실행해도 같음. 가짜 sample은 `Pipeline(services_path="data/sample_services.csv")` 또는 `--services`로 지정.
모델 캐시 폴더는 `paths.cache_dir`(기본 null) 또는 환경변수 `MODEL_CACHE_DIR`. Dense 문서 구성은 `retrieval.dense_profile`(기본 baseline).
1) 모델 없이 연결만 확인 (노트북·로컬 어디서든):
   python pipeline.py --mock "취업은 하고 싶은데 밖에 나가는 게 부담스러워요."
2) 진짜 모델 (Colab, 런타임 → T4 GPU):
   !git clone <레포주소> && %cd <레포>/ai
   !pip install -r requirements.txt
   from google.colab import drive; drive.mount('/content/drive')
   !python pipeline.py "취업은 하고 싶은데 밖에 나가는 게 부담스러워요."

## 1단계(위기 확인 + 욕구·조건 추출) 개발 세트와 평가
- 개발 세트 만들기: `python -m eval.build_devset` → `data/dev_corpus.jsonl`(말뭉치 300건, 주 검수자 판정 + 불일치 합의안), `data/dev_cards.jsonl`(검수 끝난 개발 카드, `작성자메모` 빈 칸만). 평가 카드 80장(홀드아웃)은 읽지 않음
- few-shot 예시 풀은 `dev_corpus.jsonl` 중 `label`이 있는 문장(위기 문장·근거를 원문에서 못 찾은 문장 제외). 평가 때 같은 문장은 예시에서 빠짐
- 위기 키워드: `configs/crisis_terms.yaml` (공백 무시, `exclude`로 "혼자 살" → "자살" 오탐 방지). 키워드에 안 걸리면 LLM이 다시 봄
- 평가: `python -m eval.eval_extract --mock` (흐름 확인) / Colab GPU에서 `python -m eval.eval_extract --run kanana_v1`
  - 결과: `eval/runs/<이름>/pred_*.jsonl`, `scores.json`, `report.md` (커밋하지 않음). 끊기면 같은 명령으로 이어서 실행
  - 위기 재현율(키워드·LLM·합), 영역 완전일치·micro F1, 활동범위·제약·나이·지역 정확도, 되묻기
- 키워드는 이 개발 세트를 보고 만들었으므로 개발 세트 재현율은 낙관적임. 최종 수치는 평가 카드로 냄

## 2단계(사용자 확인·수정 + 검색 쿼리)
- 흐름: `prepare(text, profile)` → `resume(session, action)` 반복 → `search(session)`
  - `prepare`: 위기 확인 → 추출 → 확인 세션(`confirm/session.py`, 형식 stage2-v1). 검색 없음
  - `resume`: 확인 화면 조작(영역 확인·수정·삭제·추가, 활동범위·제약·나이·지역 입력, 메모)과 질문 답
  - `search`: `status`가 `ready`일 때만 실행. 쿼리(`query/build.py`) → BM25·Dense 각각 → RRF → 규칙 → 설명
  - `search_queries(bm25_query, dense_query)`: 검색만 확인하는 진입점. 기존 `bm25`·`dense`·`fused` 키 + `status`(ok·empty·partial·error)·`errors`·`rrf`(서비스별 BM25·Dense 순위, RRF 점수)
  - 한쪽 검색 실패 시 남은 검색으로 결합(`partial`), 후보 없음+실패는 `error`로 규칙·설명 없이 반환. `search()` 결과의 `retrieval_status`로 확인
  - `run(text)`: 사용자 입력 없이 진행. 사용자 확인으로 처리하지 않음(`confirmation: none`), '확인 필요' 영역 미사용
- 질문: 욕구 → 지역 → 활동범위 → 나이, 세션 최대 2개(`stage2.questions`). 욕구를 건너뛰면 검색 없이 종료(explore_only)
- 확인 화면 메모는 위기 확인에만 쓰고 검색에 반영하지 않음
- 규칙·설명에는 사용자 최종 확정값(`confirmed_profile`) 사용
- `stage2.rules_pass`: 나이·지역을 규칙 모듈로 넘길지. 정규화(`rules/normalize.py`)와 판정 이유(`check_reasons`) 반영 후 true. 제외 기준은 나이, 지역(시군구 포함), 사업 종료, 방문 필수 확인 + 방문 불가. 활동범위·부담은 순위 감점만(`query/STAGE2_IO.md` 3절)
- 쿼리 사전: `configs/query_terms.yaml` v0.2(2번 담당 검수 반영, 2026-11-14). 핵심어 대부분을 조건부 확장어로 전환, 핵심어가 빈 영역은 대체 쿼리 사용. 후보 추출은 `python -m eval.build_query_terms_candidates`
- 테스트: `tests/test_stage2_confirm.py`, `test_stage2_query.py`, `test_stage2_pipeline.py`(kiwipiepy 필요, mock), `test_stage2_review.py`
- 3·4번 연결 명세와 실제 쿼리 예시: `query/STAGE2_IO.md`
- 검수·검증(저장소 루트에서, kiwipiepy 필요)
  - `python -m ai.eval.stage2_query_review sheets`: `outputs/06_stage2_query/review/`에 Q3 확정 욕구, 사전 검수 시트 생성(있으면 덮어쓰지 않음)
  - `python -m ai.eval.stage2_query_review check`: 검수 결과로 쿼리 생성 비교표(`tables/`) 생성. 검색 순위·임베딩·RRF 평가는 하지 않음
  - Q3는 평가용. 사전 수정 근거는 개발 말뭉치와 서비스마스터만

## 팀 규칙 파일 연결
`configs/config.yaml`의 `paths.rules_file`을 레포에 이미 있는 라벨링 규칙 파일 경로로 바꾸면
그 내용이 Kanana 추출 프롬프트에 그대로 들어갑니다.

## 남은 TODO (담당)
- rules/filter.py: 외출 대응표, 소득·기간 규칙 (A)
- query_terms.yaml: 보류 후보(사회참여, 교류, 이용권, 재직)와 상담 hold 정책 재검토 (2번)
- eval/eval_query.py: 확장·접근 처리 효과 실험, 관련성 후보 풀 출력 (2번)
- retrieve: 토큰 목록을 직접 받는 BM25 검색 여부 협의 (3번·2번)
- core/models.py: vLLM guided decoding으로 교체 (B)
- data/: 가짜 샘플을 실제 service_index, 개발 세트로 교체 (공통)
