# 2-b Dense 의미 검색 실행

Step 2의 `queries.dense` 문자열을 받아 서비스 ID 순위를 반환한다.
BGE-M3와 KURE-v1은 별도 모델이며 각 모델로 문서와 질의를 모두 임베딩한다.
추가 학습은 하지 않는다. BM25·RRF·자격 규칙은 변경하지 않는다.

## 설치 및 실행

Python 환경에서 기존 `ai/requirements.txt`를 설치한다.
GPU 사용 가능 여부는 `python -c "import torch; print(torch.cuda.is_available())"`로 확인한다.
아래 명령은 저장소 루트에서 실행한다. 최초 실행은 모델 다운로드와 문서 임베딩이 필요하다.

```bash
python -m pip install -r ai/requirements.txt
python ai/scripts/dense_demo.py "집과 집세 문제에 대한 도움을 찾고 있습니다." --model bge-m3 --k 100
python ai/scripts/dense_demo.py "집과 집세 문제에 대한 도움을 찾고 있습니다." --model kure-v1 --k 100
```

단독 스크립트는 Kanana를 불러오지 않는다. 기본 입력은 `data/processed/services_master.csv`이다.
`servId`를 문자열로 읽어 ID를 보존하며, 중복·빈 ID와 필수 칼럼 누락은 오류로 처리한다.
검색 필드는 기존 pipeline과 같은 서비스명·대상·혜택·신청방법이다.
BM25에서 고른 필드 조합을 Dense에 자동 적용하지 않는다.

## 인계 질의 4개 비교

`ai/data/dense_handoff_queries.jsonl`은 `ai/query/STAGE2_IO.md`의 실제 예시 4개이다.
수정 없음·경제 삭제·구직·건강 대체 쿼리를 같은 문서와 설정으로 비교한다.

```bash
python ai/scripts/dense_demo.py --queries ai/data/dense_handoff_queries.jsonl --model bge-m3 --k 100 --output private/dense_bge_m3.json
python ai/scripts/dense_demo.py --queries ai/data/dense_handoff_queries.jsonl --model kure-v1 --k 100 --output private/dense_kure_v1.json
```

기록: 모델 식별값, 문서 지문, 검색 필드, 문서 수, 최대 토큰 수, 배치 크기,
캐시 적중 여부, 모델 로드·색인·질의 시간, 질의별 서비스 ID·이름·점수·순위.
시간 비교 시 첫 실행과 캐시 적중 실행을 구분한다. GPU 시간은 환경과 워밍업의 영향을 받는다.
확정 비교에는 `--revision`으로 모델 커밋을 고정하고 실행 환경·라이브러리 버전도 보존한다.

## 메모리·오프라인 실행

- 기본 배치 8, 최대 길이 8192. 메모리 부족 시 먼저 `--batch-size 1`로 줄인다.
- 필요하면 `--max-length 1024`를 양 모델에 동일하게 적용한다. 문서 잘림에 따른 별도 실험으로 기록한다.
- `--device cpu` 또는 `--device cuda`로 실행 장치를 지정할 수 있다.
- 이미 받은 모델은 `--model-path <모델폴더>`로 로드한다. 이 경우 로컬 파일만 사용한다.
- 두 모델의 벡터를 서로 비교하거나 한 모델의 문서 캐시를 다른 모델의 질의에 사용하지 않는다.

## 캐시와 팀 연동

단독 실행 캐시는 `ai/data/dense_bge-m3.npz`, `ai/data/dense_kure-v1.npz`이다.
팀 파이프라인 캐시는 `ai/data/service_emb_<구성>.npz`이다(아래 '운영 구성').
모델 ID·리비전·토큰 길이·라이브러리 버전·로컬 파일 상태와 서비스 ID·문서·순서가 바뀌면 재계산한다.
이전 메타데이터 없는 `.npy` 캐시와 읽기 불가능한 캐시는 재계산한다.
캐시는 임시 파일 완성 후 원자적으로 교체하며 Git에 올리지 않는다.

팀 파이프라인은 `models.embedder`에 `BAAI/bge-m3`, `nlpai-lab/KURE-v1` 또는 로컬 모델 경로를 지정한다.
기존 `DenseIndex(embedder, ids, texts, cache_path)`와 `search(query, k=100) -> list[servId]`를 유지한다.
`search_with_scores()`는 단독 검토용 추가 인터페이스이다.
모델 식별 정보가 없는 외부 embedder는 안전을 위해 디스크 캐시를 사용하지 않는다.

## 검증 범위

```bash
python -m unittest discover -s tests -p test_dense.py -v
python -m pytest tests/test_dense.py tests/test_stage2_pipeline.py tests/test_bm25.py
```

가짜 임베딩 테스트는 연결·캐시·순위 연산 검증이며 실제 의미 검색 정확도 검증이 아니다.
실제 모델로 4개 예시를 실행해도 관련성 정확도나 최종 추천 품질을 확정하지 않는다.
Q3-② 관련성 정답표가 준비되면 기존 검색 평가 규격에 맞춰 두 모델과 BM25·RRF를 비교한다.
유사도 점수는 자격 충족 확률이 아니며, 자격·접근 조건 판정은 후속 규칙 단계에서 처리한다.

모델 공식 문서:
- https://huggingface.co/BAAI/bge-m3
- https://huggingface.co/nlpai-lab/KURE-v1

## 검색 품질 개선 실험

`08_dense_quality.ipynb`와 `dense_quality_colab_bundle.zip`으로 Colab에서 별도 실행한다.
이전 `08_dense.ipynb`와 초기 실행 ZIP은 기준선 재현을 위해 보존한다.

문서 구성은 `ai/retrieve/dense_texts.py`에서 정의한다.

- baseline: 서비스명·대상·혜택·신청방법. 이전 Dense 문자열 그대로 유지.
- content: 서비스명·요약·혜택·대상·선정기준. 정보 없는 전체 필드 안내문과 동일 필드값 중복 정리.
- core: 서비스명·요약·혜택. 대상 조건은 원자료에 보존하며 검색 문서에서만 분리.

개별 검색은 `dense_demo.py --profile content` 또는 `--profile core`로 확인할 수 있다.
팀 파이프라인 운영 구성은 PR #71 사람 검수 후 core로 변경함(아래 '운영 구성' 참고).
파이프라인·비교 스크립트는 `build_texts()`를 공통으로 사용한다.

`dense_quality_eval.py run --model ... --revision ... --out ...`는 모델당 3구성을 실행한다.
`dense_quality_eval.py summarize --out ...`는 두 모델의 같은 입력·설정만 집계한다.

- 개발 질의: 기존 말뭉치 중 위기 N이고 욕구 영역이 있는 원문 131건. 실제 Step 2 세션 출력은 아님.
- gold 영역과 app_domain은 참고 지표 계산에만 사용하며 문서·질의·검색 점수에 추가하지 않음.
- 영역 일치율@10 분모는 질의마다 10. 영역 적중률@10은 일치 서비스가 한 개 이상인 질의 비율.
- 서비스 영역이 존재하는 질의만의 결과를 별도 병기. 복수 영역 질의는 영역별 표에서 중복 포함.
- 인계 4질의 × 6구성의 Top-5 합집합을 관련성 검토표로 생성. 정답은 자동 입력하지 않음.
- 세 구성은 여러 필드와 안내문 처리가 함께 달라지므로 개별 변화의 인과 효과로 해석하지 않음.
- 현재는 개발 참고 평가이며 독립 관련성 정확도·모델 우열·최종 추천 품질 검증이 아님.

## 인계 질의 사람 관련성 검수 집계

- `dense_quality_eval.py review`: 사람 판정과 실험 순위표로 6개 조합(2모델 × 3구성) 집계
- 순위표·검수표는 읽기 전용, 결과는 `--out/tables/human_review_*`에 저장

```bash
python ai/scripts/dense_quality_eval.py review \
  --rankings outputs/08_dense_quality/colab_20261115/tables/08_dense_quality/colab_20261115/tables/handoff_rankings.csv \
  --review outputs/08_dense_quality/colab_20261115/tables/dense_human_review_62_final.csv \
  --out outputs/08_dense_quality/colab_20261115
```

결과 해석: `outputs/08_dense_quality/colab_20261115/human_review_report.md`

- 입력 연결: `query_id + servId`. 순위표는 `kind=handoff` 행의 모델·구성·질의별 Top-5만 사용
- 사람 판정: `relevance_0_1_2` 열만 사용(0 무관, 1 간접 관련, 2 직접 관련). `ai_*` 등 초안 열은 읽지 않음
- 검수표 형식: 원본 검수표(23열)와 검수 HTML 출력(`번호, query_id, servId, relevance_0_1_2, reason, reviewer`) 모두 가능
- 오류(하나라도 있으면 집계 중단, `human_review_errors.csv`에 키와 행 번호 기록): 검수 키 중복, Top-5 판정 누락, 판정 공란, 0·1·2 외 값, 순위표 조합·질의 누락, 순위 중복
- 질의 단위 지표(조합 평균 분모는 질의 수): Precision@5(판정 2 비율), 평균 판정, 0점 비율, 직접 관련 수
- nDCG@5(주지표): 이득 `2^판정-1`(0·1·3, `--gain exp2` 기본), 감쇠 1/log2(순위+1). `--gain linear`는 판정값 그대로. 이상적 순위는 질의별 사람 판정 후보 풀 전체(검수표의 해당 질의 행, 이번 62건은 6조합 Top-5 합집합과 동일)를 판정값 내림차순으로 놓은 Top-5
- 신청 자격(`eligibility_status`)·접근 제약(`access_status`): 관련성과 합산하지 않고 `human_review_eligibility_access.csv`에 조합별 건수로 따로 기록
- 이상적 DCG가 0인 질의(풀 전체 무관): nDCG 미정의(빈 값). 조합 평균에서 제외하고 `ndcg_excluded_ideal_zero`에 개수 기록, 분모는 `ndcg_queries`
- 출력: `human_review_per_query.csv`(질의·조합별), `human_review_summary.csv`(조합별 평균·분모, nDCG 내림차순), `human_review_eligibility_access.csv`(자격·접근 별도 건수), `human_review_metadata.json`(입력 해시·계산 기준)
- 질의 4개 기술 집계라 모델·구성 우열의 통계적 근거로 쓰지 않음. 모델 선정·운영 설정 변경은 팀 검토 후 결정

## 운영 구성 (BGE-M3 core, 잠정)

- 근거: 인계 질의 4개 × 사람 검수 62쌍 nDCG@5 BGE-M3 core 0.8294 > baseline 0.7782 (개발 평가, 최종 성능 근거 아님)
- 설정: `ai/configs/config.yaml`
  - `models.embedder: BAAI/bge-m3` (변경 없음, 1단계 few-shot 임베딩과 공용)
  - `retrieval.dense_profile: core` (이전 baseline)
  - `paths.dense_cache: data/service_emb_{profile}.npz` (이전 `data/service_emb.npy` 단일 파일)
- core 문서: 서비스명·요약·혜택(`dense_texts.PROFILES["core"]`). PR #71 평가와 같은 함수·같은 마스터(SHA-256 `269005…6d83d6e`)
- BM25 필드·토크나이저, RRF k·후보 수, 자격·접근·지역·기간 규칙, Q1 우대는 변경 없음
- baseline으로 되돌리기: `dense_profile: baseline`만 변경. 캐시는 `service_emb_baseline.npz`로 따로 생성·재사용
- Q3 독립 평가에서 baseline·core를 다시 비교할 수 있게 두 구성을 유지함

### 캐시

- 구성별 파일: `{profile}` 자리에 구성 이름. baseline·core 전환 시 서로 덮어쓰지 않음
- 재사용 조건: 파일 안 지문 = 모델 식별키(이름·HF 커밋·최대 길이·라이브러리 버전) + 서비스 ID·문서·순서. 하나라도 다르면 재계산 후 교체
- 마스터 갱신·모델 갱신·구성 변경 시 별도 삭제 없이 다음 실행에서 자동 재생성
- 이전 단일 캐시 `ai/data/service_emb.npy`는 더 읽지 않음(baseline 지문이 같으면 `service_emb_baseline.npz`로 이름만 바꿔 재사용 가능)
- 캐시 파일은 Git에 올리지 않음(`ai/data/*.npz` 무시)

### 실제 모델 검증 (Colab GPU)

`ai/scripts/dense_ops_check.py`: baseline·core 캐시 생성 → 재사용 → 전환, 2단계 검색 연동 확인.

```bash
python ai/scripts/dense_ops_check.py --device cuda            # 실제 BGE-M3, 설명문은 MockLLM
python ai/scripts/dense_ops_check.py --device cuda --real-llm # 설명문까지 Kanana
python ai/scripts/dense_ops_check.py --mock                   # 연결만 확인, 실제 검증 아님
```

- 필요 파일: `data/processed/services_master.csv`, `outputs/03_q1_recommendation/q1_review.json`, `outputs/04_q2_recommendation/q2_info.json`
- 질의: 인계 4개(파일 문장 그대로) + 개발 말뭉치 사람 label 세션 110개의 현재 `build_queries` 결과(#75 반영). Q3 카드 미사용
- 출력: `outputs/10_dense_ops/<실행시각>/tables/`(`index_checks`, `queries`, `rankings`, `top5_cards`, `profile_compare`, `exceptions`), `runs/metadata.json`
- 종료 코드 0: 서비스 5,288건·ID·임베딩·지문·캐시 재사용·구성별 파일 분리·예외 처리 모두 통과
- baseline·core 공통 서비스 수는 순위 차이 기술이며 관련성 정확도가 아님
