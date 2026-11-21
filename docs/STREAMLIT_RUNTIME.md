# 최종 모델·데이터 Streamlit 연결

## 실행 소스

- 최신 기준 main은 `9e9fa20`이며 기존 UI 기준은 `4235700`
- `feb5e5e` 방문 부담 후처리와 `03e5be7` Q1 확정 분류가 최신 기준에 포함됨
- 로컬 작업 트리에 병합한 상태이며 커밋·푸시하지 않음
- GitHub 실행 노트북의 기본 브랜치는 `codex/streamlit-handoff-20261118`. 게시 전에는 ZIP 노트북으로 로컬 수정본을 확인할 수 있음

## 연결 구성

- Kanana: `kakaocorp/kanana-2-3b-instruct`, 기존 추출 프롬프트·나이 보정·JSON 재시도 사용
- 임베딩: `BAAI/bge-m3`, 문서 `core` 구성, BM25와 RRF 1:1 유지
- 운영 CSV: `data/processed/services_master.csv`, 5,288건
- Q1: `outputs/03_q1_recommendation/q1_review.json`, 원본 CSV를 바꾸지 않고 검수된 분야·순위 정보 연결
- Q2: `outputs/04_q2_recommendation/q2_info.json`, 동일 사업 38건·6개 항목의 출처 근거 연결
- Gemini: 일상 대화의 기존 별도 경로 유지. 키가 없으면 기본 응답 사용

## 로컬 검증

```powershell
python -m pip install -r ai/requirements-test.txt
python ai/scripts/prepare_runtime.py
$env:BANGEOL_PIPELINE_MODE = '실제 흐름(가짜 모델)'
$env:BANGEOL_REQUIRE_REVIEW_DATA = '1'
cd ai
python -m streamlit run app.py
```

위 모드는 최종 서비스·검수 데이터와 실제 파이프라인을 사용하지만 LLM·임베딩은 가짜이므로 모델 정확도·T4 속도를 입증하지 않는다. `?dev=1` → 개발자에서 서비스·Q1·Q2 연결 건수와 모델 설정을 확인한다.

코드·설정·데이터 변경 후 앱을 재시작한다. Streamlit의 Pipeline 캐시도 입력 파일 버전으로 구분하며 변경 감지 시 기존 검색 대화를 초기화한다. 모델·색인 캐시 자체는 기존 지문 검증을 유지한다.

## Colab

GitHub 실행 노트북 순서: GPU/설정 → GitHub 받기 → 설치·검수 데이터 준비 → 실제 모델 점검(선택) → 앱 실행. 실행 중인 앱은 실행 셀에서 종료 후 새 프로세스로 시작한다. 토큰은 원격 URL이나 노트북 출력에 넣지 않으며, 로컬 수정이 있는 checkout은 자동으로 덮어쓰지 않는다.

로컬 미커밋 코드를 T4에서 확인하려면 다음으로 ZIP을 만든 뒤 기존 `streamlit_t4.ipynb`에서 업로드한다.

```powershell
python ai/scripts/build_runtime_bundle.py --output C:/temp/bangeol-real-ai.zip
```

ZIP에는 생성한 Q1/Q2 연결 파일이 포함된다. manifest의 파일별 해시가 실제 실행 버전이며 `working_tree_dirty=true`는 기준 커밋 외 로컬 수정이 포함됐다는 뜻이다. 모델 가중치·토큰·개인 설정·검색 캐시는 포함하지 않는다.

실제 GPU 점검은 UI와 같은 목록 우선 경로를 사용한다. `python scripts/real_model_check.py --explain`을 실행하면 상세 1건의 생성 시간도 별도로 측정한다. 기본 점검만으로 Gemini의 실제 API 응답이나 공개 서비스 부하까지 검증한 것은 아니다.

## 화면 변경

- 둘러보기는 연결된 Q1 핵심 분야 사용: 노동 4건, 미래설계 8건
- Q2 동일 사업 출처 비교는 추천 카드뿐 아니라 둘러보기·이번 방문의 상세에서도 제공
- 나이·소득·신청 경로·신청 기간·서류·문의처에 대해 두 수집 원문과 출처 링크 표시
- 원문 수집 시점과 현재 모집 여부는 구분하며 Q2 비교값을 신청 조건으로 자동 덮어쓰지 않음
- 서비스 상세 대화상자와 목록 스크롤 유지 동작 보존
- 시작 화면의 예시 문장은 말풍선과 입력창 사이에 배치하고 개인정보·외부 AI 안내는 작은 보조 글씨로 표시
- 대화 갱신 시 최신 사용자 메시지부터 이어지는 묶음으로 이동하며 직접 스크롤하면 자동 이동 중단. 추천 결과 화면 진입 시 맨 위로 이동
- 둘러보기는 넓은 본문에서 4열, 좁은 본문에서 2열로 전환. 검색창은 두 칸 너비 유지
- 좁은 화면은 사이드바 대신 상단 메뉴 제공. 메뉴에서 새 이야기·개인정보·안전 안내에 접근 가능
- 기능 제목·본문·버튼의 크기와 줄 간격을 정리하고, 카드 소개는 세 줄까지 표시. 전체 내용은 상세에서 제공
- 이번 방문은 카드별 작은 선택 칸과 전체 선택, 삭제 버튼 하나만 제공. 하나·여러 개·전체를 체크한 뒤 같은 버튼으로 삭제하며 현재 세션의 담은 목록에만 적용
