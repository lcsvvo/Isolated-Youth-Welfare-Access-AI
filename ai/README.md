# AI 추천 파이프라인과 Streamlit

[전체 인수인계](../docs/STREAMLIT_HANDOFF.md) · [구조도](../docs/STREAMLIT_ARCHITECTURE.md) · [변경 기록](../docs/STREAMLIT_CHANGELOG_2026-11-17.md)

## 화면 실행

저장소 루트에서 설치하고 `ai/`에서 실행합니다.

```powershell
python -m pip install -r ai/requirements-ui.txt
python ai/scripts/prepare_runtime.py
cd ai
python -m streamlit run app.py
```

기본은 시연용 가짜입니다. 상단 모드 배지 또는 `?dev=1`에서 `실제 흐름(가짜 모델)`을 선택하면 운영 마스터와 실제 Pipeline을 GPU 없이 확인할 수 있습니다. 이 모드는 추천 품질 검증이 아닙니다.

실제 모델은 `requirements-streamlit.txt`를 설치한 GPU 환경에서 사용합니다. [GitHub 실행 노트북](notebooks/streamlit_colab_github.ipynb)은 코드 업데이트·검수 데이터 생성·앱 실행을 연결합니다. 로컬 미커밋 수정은 GitHub 받기에 포함되지 않으므로 해당 수정은 [T4 ZIP 노트북](notebooks/streamlit_t4.ipynb)과 로컬 런타임 ZIP으로 확인합니다. 실제 모델 실패 시 가짜 모드로 자동 대체하지 않습니다.

`prepare_runtime.py`는 Git에 있는 최종 검수 입력으로 Q1 5,288건과 Q2 동일 사업 38건의 연결 파일을 생성하고 마스터 SHA-256·서비스 ID를 검사합니다. 입력이 같으면 검증한 결과를 재사용합니다. Colab 실행은 `BANGEOL_REQUIRE_REVIEW_DATA=1`로 누락 시 중단합니다. [최종 연결 실행 안내](../docs/STREAMLIT_RUNTIME.md)에 확인 범위를 정리했습니다.

## 모듈 API

```python
from pipeline import Pipeline
p = Pipeline(mock=True)  # 실제 모델은 mock=False
s = p.prepare("월세와 생활비가 걱정돼요")
# p.resume(s, action)으로 확인·답변한 뒤 status=ready에서만 검색
out = p.search(s, explain_mode="source_first")
# 검색 결과 상세에서 사용자 요청 시만 호출
# card = p.explain_service(s, out, out["cards"][0]["servId"])
```

- `prepare`: 위기 확인 → 욕구·조건 추출 → 확인 세션
- `resume`: 확인 조작·추가 질문·자유 답 위기 재확인
- `search`: BM25 + bge-m3 Dense → RRF → 규칙 → 카드
- `search(s)` 기본값은 기존 AI 설명 포함. Streamlit은 `source_first`로 목록을 먼저 표시
- `explain_service`: 현재 결과·확정 조건을 검사하고 요청한 1건만 설명
- `timings_ms`: 쿼리·검색·규칙·설명·전체 소요 시간

## 주요 파일

| 역할 | 경로 |
|---|---|
| 화면·상태 전환 | `app.py` |
| 파이프라인 | `pipeline.py` |
| 모델 로딩 | `core/models.py` |
| 욕구 추출 | `extract/` |
| 위기 확인·정적 안내 | `safety/` |
| 사용자 확인·추가 질문 | `confirm/` |
| 쿼리 | `query/` |
| BM25·Dense·RRF | `retrieve/` |
| 자격·접근·기간 규칙 | `rules/` |
| 원문 근거 설명 | `explain/` |
| 소개 문장 표시 | `ui_text.py` |
| 동의 기반 작은 활동 | `small_activities.py`, `data/small_activities.json` |
| 런타임 ZIP | `scripts/build_runtime_bundle.py` |
| Colab 실행·복구 | `scripts/colab_launch.py` |
| 실제 모델 확인 | `scripts/real_model_check.py` |

## 실행 ZIP

```powershell
python ai/scripts/build_runtime_bundle.py --output C:/temp/bangeol-real-ai.zip
```

위 명령은 저장소 루트에서 실행합니다. 모델·개인 설정·인증정보를 포함하지 않고 현재 소스와 서비스 데이터의 해시를 검증합니다. 실행 경로·임시 URL을 소스에 고정하지 마세요.

## 테스트

```powershell
python -m pip install -r ai/requirements-test.txt
python -m pytest ai/eval/test_service_first.py ai/eval/test_region_residence_scope.py ai/eval/test_small_activities.py ai/eval/test_real_model_adapter.py ai/eval/test_streamlit_handoff.py ai/eval/test_runtime_handoff.py -q
```

GPU 실측은 별도 T4 노트북의 사전 검증 셀입니다. 자동 정보 갱신·OpenAI API·기관 공유·영구 기록은 현재 기능이 아닙니다. 필터 범위·문구·데이터 라벨·후속 작업은 전체 인수인계를 확인하세요.
