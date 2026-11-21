# Streamlit·실제 AI 인수인계

기준일: 2026-11-17 / 작업 브랜치: `feat/streamlit-ui` / 시작 커밋: `483b5d8`.

최신 모델·검수 데이터 연결과 실행 순서는 [최종 연결 실행 안내](STREAMLIT_RUNTIME.md)를 우선 참고한다. 아래는 최초 인수 시점 기록이며, Q1/Q2 연결 후 둘러보기는 노동 4건·미래설계 8건을 표시한다.

이 문서는 이번 화면·모델 연결 작업의 현재 실행 방법과 기술적 변경·검증·제약을 묶은 진입점이다. 설계는 [화면 규격](STREAMLIT_DESIGN.md)·[구조도](STREAMLIT_ARCHITECTURE.md), 변경 이유는 [작업 기록](STREAMLIT_CHANGELOG_2026-11-17.md), 디자인 원본은 [참조 자료](streamlit-reference/README.md)에 보존했다.

## 1. 현재 상태와 검증 수준

| 항목 | 현재 상태 | 확인 근거·한계 |
|---|---|---|
| Streamlit 화면 | HTML 기준 복원·연결 | 로컬 브라우저 확인, 사용자가 디자인·대화 화면 표시 확인 |
| Kanana + bge-m3 | 실제 T4에서 실행 확인 | 아래 단일 smoke 결과. 지속 서비스·부하 검증 아님 |
| 서비스 먼저 표시 | UI 기본 동작 | 로컬 회귀 테스트 통과. T4 개선 후 지연 시간은 미측정 |
| 상세 AI 설명 | 명시 요청 시 1건 생성 | 이번 세션 카드에 저장해 재사용. 최초 요청 지연은 남음 |
| 작은 활동 | 동의 → 조건 → 최대 3개 | 문장 규칙과 19개 팀 초안. 임상·회복 단계 모델 아님 |
| 정리표 상태 복구 | 현재 답변 유지 후 재편집 | ready/awaiting_answer/explore_only 복구 테스트 |
| 서울에 부안군 사업 노출 | 확인한 원인 수정 | `WLF00004762` 서울 제외, 부안군 통과 회귀 검증 |
| 말투 | 소개 표시를 해요체로 정리, 생성 지시 통일 | 기관 원문은 보존. 임의 문장 전체를 완벽히 재작성하는 모델 아님 |
| 최신 복지정보 자동 갱신 | 미구현 | 기존 API 수집기는 있으나 주기 실행→앱 반영 연결 없음 |
| OpenAI API | 미사용 | 수집한 공고 구조화 용도로 검토 가능. 채택 합의·키 연결 미확인 |
| 영구 저장·기관 전송 | 앱 기능으로 미구현 | 인프라·모델 처리·로그 정책은 운영 전 확인 필요 |

사용자의 Colab은 일회성 런타임이다. 최신 지역·문구 패치가 그 런타임에 반영되었는지는 별도로 확인해야 한다. Git 코드 완료와 사용자의 개별 런타임 배포를 같은 상태로 취급하지 않는다.

## 2. 이어받는 순서

1. `feat/streamlit-ui` 최신 코드 받기. `main` 직접 푸시·강제 푸시 금지.
2. `data/processed/services_master.csv`와 `.meta.json` 확인.
3. 아래 로컬 가짜 모델 모드로 UI·상태 전환 확인.
4. 번들 생성 후 Colab 노트북에서 T4 실제 모델 실행.
5. 목록 우선 표시의 T4 지연 시간과 상세 설명 시간 측정.
6. 자동 갱신 요구를 별도 작업으로 설계. 현재 CSV를 실시간 데이터로 설명하지 않기.

```powershell
git switch feat/streamlit-ui
git pull --ff-only origin feat/streamlit-ui
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r ai/requirements-test.txt
cd ai
..\.venv\Scripts\python.exe -m streamlit run app.py
```

로컬 검증 환경은 Windows / Python 3.13.9 / Streamlit 1.67.0이었다. `ai/`에서 실행해야 프로젝트의 `.streamlit/config.toml` 밝은 테마가 적용된다. 사용자의 PC는 MX450 2GB로 확인되어 실제 모델 두 개를 로컬 GPU에서 함께 실행하지 않았다.

### 실행 모드

상단 상태 배지를 누르거나 `http://127.0.0.1:8501/?dev=1`로 열면 개발자 설정을 사용할 수 있다.

| 선택 값 | 실제 동작 | 화면 배지 |
|---|---|---|
| 시연용 가짜 | MockPipeline + 예시 서비스, 화면 확인용 | 시연용 가짜 · 화면 확인용 |
| 실제 흐름(가짜 모델) | 실제 Pipeline·운영 CSV·BM25 + MockLLM/MockEmbedder | Pipeline 연결 · 테스트 모델 |
| 실제 모델 | 실제 Pipeline·운영 CSV + Kanana/bge-m3 | 실제 Pipeline 연결 |

환경변수 `BANGEOL_PIPELINE_MODE`로 새 세션의 기본 모드를 정한다. 값은 표의 선택 값과 정확히 같아야 한다. 모드 변경은 대화를 초기화한다. 실제 모델 초기화 실패 시 가짜 모드로 자동 전환하지 않고 오류와 재연결 버튼을 표시한다.

### 일상 대화(Gemini) 키 설정

인사·고마움 같은 일상 대화 문장만 외부 AI(Google Gemini)로 보낸다. 위기 확인(`prepare`)을 먼저 통과하고 필요한 도움이 하나도 없을 때만이며, 보내는 것은 그 한 문장뿐이다(이전 대화·세션·조건은 보내지 않음). 복지 찾기·위기 안내는 기존 Pipeline 그대로다. 키가 없거나 오류·10초 초과·빈 응답이면 정해진 기본 답으로 대체하고 화면은 멈추지 않는다. 코드는 `ai/chitchat.py`, 지시문은 `ai/prompts/chitchat_system.md`.

키는 **`st.secrets["GEMINI_API_KEY"]` → 환경변수 `GEMINI_API_KEY`** 순서로 읽는다. 키 값을 커밋·출력·로그로 남기지 않는다.

- **회사 PC**: `ai/.streamlit/secrets.toml` 파일을 만들고 아래 한 줄을 넣는다. 이 파일은 `.gitignore`(`**/.streamlit/secrets.toml`) 대상이라 커밋되지 않는다. 앱은 `ai/`에서 실행해야 이 파일을 읽는다. 환경변수(`$env:GEMINI_API_KEY="..."`)로도 된다.

  ```toml
  GEMINI_API_KEY = "여기에_키"
  ```

- **Colab**: 왼쪽 열쇠(보안 비밀)에 이름 `GEMINI_API_KEY`로 키를 저장하고 노트북 접근을 켠다. 화면 실행 셀 **앞**에서 환경변수로 옮긴다. `colab_launch.py`가 환경변수를 그대로 이어받는다.

  ```python
  import os
  from google.colab import userdata
  os.environ["GEMINI_API_KEY"] = userdata.get("GEMINI_API_KEY")
  ```

  Colab은 `ai/requirements.txt`로 설치하며 `google-genai`가 들어 있다. 실행 ZIP에는 `secrets` 이름의 파일이 들어가지 않는다.
- **모델**: 기본 `gemini-3.5-flash-lite`. `GEMINI_MODEL`(같은 방식으로 secrets 또는 환경변수)로 바꿀 수 있다. 모델 종료 일정은 공식 deprecations 페이지에서 확인한다.
- 개발자 화면에 마지막 응답이 `gemini`인지 `fallback:이유`(no_key·TimeoutError 등)인지 표시된다.
- 테스트는 가짜 Gemini를 쓴다(`ai/eval/test_chitchat.py`). 진짜 키로 확인할 때는 "안녕하세요"·"고마워요"가 외부 AI 답(말풍선 아래 "AI가 만든 답이에요."), "월세가 걱정돼요"가 기존 복지 흐름으로 가는지 본다.

## 3. Colab 실행·복구

정식 런처 정리본은 로컬 모의 테스트를 통과했으며 이 최종 revision의 T4 재실행은 아직 미확인이다. 앞서 사용자 T4에서 연결 성공을 확인한 복구 스크립트와 같은 현재 origin/압축 설정을 소스로 정리했다.

정식 실행 노트북: [`ai/notebooks/streamlit_t4.ipynb`](../ai/notebooks/streamlit_t4.ipynb).

### 최신 실행 ZIP 만들기

```powershell
python ai/scripts/build_runtime_bundle.py --output C:/temp/bangeol-real-ai.zip
```

생성기는 런타임 소스·설정·활동 초안·few-shot 예시·서비스 마스터만 허용 목록으로 묶고 SHA-256 manifest를 검증한다. `.git`, 인증정보 파일, 모델·임베딩 캐시, 개인 설정, 테스트 결과는 넣지 않는다. 지정 경로의 ZIP은 생성 결과물이므로 커밋하지 않는다.

### T4에서 실행

1. 노트북을 Colab에 업로드하고 `런타임 → 런타임 유형 변경 → T4 GPU` 선택.
2. GPU 확인 셀 실행: `torch.cuda.is_available()` 및 `nvidia-smi` 확인.
3. 파일 업로드 셀에서 **bangeol-real-ai.zip** 선택. manifest 검증.
4. `ai/requirements-streamlit.txt` 설치 셀 실행.
5. 최초 인수 시 실제 모델 사전 검증 권장. 기본 Pipeline의 전체 설명 생성까지 검증하므로 수 분 소요 가능.
6. 화면 실행 셀 실행. 목록 우선 표시 UI로 시작.
7. 출력의 **실제 AI 화면을 새 탭으로 열기** 링크 또는 노트북 내 미리보기 사용. 노트북·T4 연결 유지.

노트북은 `colab_launch.py`를 사용한다. 매번 `google.colab.kernel.proxyPort(8501)`에서 현재 주소를 얻고, 그 정확한 origin을 CORS 허용 목록에 넣으며 WebSocket 압축을 끈다. CORS/XSRF 보호 전체를 끄지 않는다. 이전에 전달한 `*.colab.dev` 주소를 문서·코드에 고정하지 않는다.

`127.0.0.1:8501`은 사용자 PC의 로컬 앱이다. Colab 실제 모델 앱 주소와 다르다. `Local/Network/External URL` 로그만 보고 로컬 주소나 VM IP를 열지 않는다.

### 화면만 다시 열기

앱 폴더가 유지된 T4 런타임에서 아래 셀만 실행한다. 기존 화면 앱은 재시작되며 대화는 초기화된다. 모델·임베딩 파일 캐시는 유지된다.

```python
from pathlib import Path
import runpy
ROOT = Path('/content/bangeol-real-ai')
launch = runpy.run_path(str(ROOT / 'ai/scripts/colab_launch.py'))
streamlit_process = launch['start_preview'](ROOT, globals().get('streamlit_process'))
```

- `health=200`은 서버 HTTP 응답 확인이며 실제 모델 적재·화면 렌더 완료를 보장하지 않는다.
- 모델 최초 적재와 첫 임베딩은 느릴 수 있다. 앱 제목·본문도 안 뜨는 지속적인 빈 화면을 모두 모델 대기로 설명하지 않는다.
- TensorFlow의 CUDA 경고만으로 CPU 실행이라고 결론내리지 않는다. 이 경로의 실제 추론은 PyTorch이며 `torch.cuda.is_available()`·`nvidia-smi`·메모리 실측으로 확인한다.
- 여전히 비면 `ROOT / 'streamlit.log'` 확인. 사용자 대화 원문이 들어갈 수 있는 로그를 무분별하게 공유하지 않는다.
- 런타임이 초기화되어 `bangeol-real-ai`가 없으면 GPU 확인 → ZIP 업로드 → 설치 → 화면 실행 순서로 복원한다. 실제 모델 사전 검증 전체는 매번 반복하지 않아도 된다. 모델 캐시도 사라졌다면 다운로드가 다시 필요하다.
- 다른 이름의 ZIP을 선택하면 업로드 셀의 파일명 검증에서 멈춘다. ZIP명은 위 이름 유지.
- 예전 복구 `.py` 파일 업로드 시 Colab이 `(1)`을 붙여 `KeyError`가 발생했다. 업로드 반환 dict의 실제 키를 사용하거나, **방금 하나의 신뢰하는 파일만 업로드한 경우** `exec(next(iter(uploaded.values())).decode('utf-8'))`로 실행했다.
- 과거 복구·속도·지역 패치 파일은 일회성 이행 수단이다. 새 팀원은 패치 순서를 재현하지 말고 최신 소스로 새 번들을 만든다.

### GitHub에서 바로 받는 노트북

[`ai/notebooks/streamlit_colab_github.ipynb`](../ai/notebooks/streamlit_colab_github.ipynb)는 ZIP 없이 GitHub 저장소를 바로 받아 실제 모델로 실행한다(위 ZIP 방식의 대안). 실행 결과는 지우고 커밋했으므로 열면 빈 상태다.

- **준비**
  1. 런타임 유형을 T4 GPU로 바꾼다.
  2. 왼쪽 열쇠(보안 비밀)에 `GH_TOKEN`(**필수**, 저장소를 읽을 수 있는 GitHub 토큰)을 저장한다.
  3. `GEMINI_API_KEY`(일상 대화용, 선택)를 저장한다. 키가 없으면 자동으로 넘어가요: 셀 1이 "Gemini 키: 없음"을 출력하고 앱은 일상 대화에 기본 답을 쓴다. 키가 있으면 "Gemini 키: 있음"이 나온다(키 값은 출력되지 않음).
  4. 각 비밀의 **노트북 액세스**를 켠다.
- **실행 순서**
  - 처음: 셀 1 → 2 → 3 → 5
  - 코드가 바뀌었을 때: 셀 2 → 5 (최신 코드를 받고 화면을 다시 띄움)
  - 문제가 생겼을 때: 셀 7 (앱 로그 끝부분과 GPU 메모리 출력)
- **병합 전 확인**: 셀 1의 `BRANCH = "main"`을 `feat/streamlit-ui`(확인할 브랜치 이름)로 바꾸면 main에 합치기 전 코드를 Colab에서 볼 수 있다. 확인 뒤에는 `main`으로 되돌린다.
- 셀 6은 선택이다. 외부에서 열 수 있는 임시 링크(cloudflared)를 만든다. 링크를 아는 누구나 접속할 수 있으므로 실제 개인정보를 입력하지 않는다.
- 토큰·키는 노트북에 쓰지 않고 보안 비밀에서만 읽는다. 셀 2는 출력에서 토큰을 가리고 clone 뒤 저장소 주소에서 토큰을 지운다. 노트북을 저장소에 다시 올릴 때는 실행 결과를 지우고 `ghp_`·`github_pat_`·`AIza`로 시작하는 문자열과 임시 링크 주소가 없는지 확인한다.

## 4. 실제 T4에서 확인한 수치

2026-11-17 Colab 셀 출력에서 확인한 **속도 개선 전 단일 사례**다. 평균·지속 부하·모든 문장 성능이 아니다.

| 기록 | 관측값 |
|---|---|
| GPU | Tesla T4 |
| 모델 | `kakaocorp/kanana-2-3b-instruct`, `BAAI/bge-m3` |
| 서비스 | 5,288건 |
| Pipeline 초기화 | 389.46초 |
| 일반 문장 검색(설명 5개 포함) | 223.1초 |
| 일반 결과 | ready / retrieval_status=ok / 카드 5개 / 추천 이유 존재 |
| ‘모르겠어요’ 추출 | awaiting_confirm / needs 0개 |
| 위기 문장 | crisis_stop / searched=false / 카드 0개 |
| GPU peak allocated | 8.67 GiB |
| GPU reserved | 8.76 GiB |

추출·위기 확인도 실제 모델 호출이므로 첫 입력 지연은 별도로 남는다. 빠른 결과 표시 뒤에도 상세 AI 설명 최초 요청은 시간이 걸릴 수 있다.

새 `search()` 결과의 `timings_ms`에는 `query_ms`, `retrieval_ms`, `rules_ms`, `explanation_ms`, `total_ms`가 기록된다. 타이머는 wall-clock 기준이며 GPU 커널 프로파일러 수치가 아니다. 새 경로의 실측 없이 ‘몇 초로 개선’이라고 보고하지 않는다.

## 5. 데이터·필터·문구

- 운영 서비스: `data/processed/services_master.csv` + 메타 JSON. 로컬 파일 기반이며 자동 실시간 갱신 아님.
- Q1/Q2 검수 부가 JSON은 기본 config 경로에서 없으면 선택적으로 생략. 존재하면 원문 마스터 해시·스키마 검증. 맞지 않는 파일을 억지 연결하지 않는다.
- 나이·시도·시군구·접근 조건은 확정된 `confirmed_profile → rules_view`를 통해 백엔드로 전달.
- 지역 불일치는 원문 거주 요건과 검수 overlay를 함께 고려. 제공 지역이 다른 것만으로 모든 경우를 무조건 자격 없음으로 판단하지 않는다.
- 만남 형태·반복 이용·면접 없음 세부 필터는 받은 TOP 5 안에서만 적용한다. 필터 후 5개 미만이 되어도 백엔드 추가 후보로 자동 보충하지 않는다.
- 화상/가정방문/온라인 등 이용 방식 5종 전달 필터는 이번 작업에서 구현하지 않았다.
- 활동 범위는 은둔 단계 자동 판정이 아니다. 고정 말투 선택과 명시한 활동 범위에만 사용한다.
- 서비스 둘러보기는 `app_domain` 토큰·검색어로 직접 탐색하며 추천 순위가 아니다. 정보 부족 화면은 분야별 예시 최대 6개이며 자격 확정이 아니다.
- 현재 마스터에서 노동(일하는 중)·미래설계 토큰은 0건. 버튼 값과 백엔드 분야명은 일치했다. 노동·미래설계 라벨 재검수는 미완료이고 버튼을 `label_plain`으로 통일할지 팀 결정 필요.

### 확인한 지역 오류

`WLF00004762 / 청년 주거비용 지원사업 / 전북특별자치도 부안군`:

- 대상: `19세 이상 45세 이하인 부안군 거주 무주택 청년 가구`
- 기준의 `전국 무주택자`를 기존 `_NON_RESIDENCE`가 ‘전국 모집’으로 잘못 해석하여 서울 사용자에게 확인 필요로 남김.
- `전국 무주택·주택·재산·소득·중복·조회·검색`은 지역 무관 표현에서 제외하도록 보완.
- 서울은 제외, 부안군은 통과, 명시적인 전국 청년 누구나 신청 가능 사례는 제외하지 않는 테스트 추가.
- 이 한 사례 수정이 모든 서비스의 지역 정확도를 보장하지는 않는다. 제공 지역·원문 자격 지역 불일치 및 다른 표현은 계속 검수 필요.

### 문구 정책

- `ui_text.friendly_summary()`는 소개 표시만 해요체로 정리. 숫자·조건·기관 원문 데이터는 변경하지 않는다.
- 알려진 종결형은 바꾸고, 해석하기 어려운 명사구는 인용 안내로 남겨 임의 자격·혜택을 만들지 않는다.
- 상세의 지원 대상·선정 기준·지원 내용·신청 방법은 원문 구분을 유지.
- `explain/explain.py`의 생성 지시는 reason/how_to_apply의 해요체를 요구하나 모든 출력 문체의 완벽한 보장을 뜻하지 않는다.
- 근거 원문 검증 실패 시 설명 템플릿으로 대체. 생성 설명과 원문 소개를 같은 것으로 표시하지 않는다.

## 6. 작은 활동

`small_activities.should_offer()`는 첫 입력 및 적절한 자유 답변에서 ‘뭘 해야’, ‘의욕이 없’, ‘무기력’, ‘아무것도 하기 싫’, ‘모르겠어요’ 등 정해진 표현을 감지한다. 나이·지역 질문에 ‘모르겠어요’라고 답한 것만으로 자동 제안하지 않는다.

1. 먼저 `prepare()` 또는 `resume(memo)`로 위기 확인.
2. 위기가 아니면 활동 동의 화면. 서비스 질문을 먼저 강제하지 않음.
3. 작은 활동 / 서비스 먼저 / 쉬어가기 선택.
4. 동의 후 활동 범위·오늘의 여유(1~3)·상호작용(0/1)을 직접 선택.
5. 초안 19개에서 조건 내 최대 3개. 기본은 가벼운·상호작용 없는 선택. 허용 범위 밖으로 임의 확대하지 않음.
6. 방문 불가이면 집 밖 범위를 제한. 서비스 이야기로 돌아갈 때 기존 대화·질문을 이어감.
7. 위기 세션은 제안·활동·쉬어가기 페이지도 안전 화면으로 차단하고 이전 활동 결과를 비움.

원본 HTML의 결과 하단 선택 기능을, 이번 사용자 요구에 따라 초기 막막함에도 먼저 묻는 흐름으로 확장했다. 목록의 `q21`, `domains`는 출처·설계 메타이며 현재 선택 함수는 범위·effort·contact 조건으로 고른다. 자동 우울/은둔/회복 단계 판단이나 임상 효과 검증으로 설명하지 않는다. 이 선택은 사용자 서비스 자격·추천 조건을 변경하지 않는다.

## 7. 상태·추천 API 유지 사항

```python
session = pipe.prepare(text)
session = pipe.resume(session, action)
# 검색은 ready에서만 가능
result = pipe.search(session)  # 기존 API: AI 설명까지 생성
result = pipe.search(session, explain_mode='source_first')  # Streamlit 기본
card = pipe.explain_service(session, result, result['cards'][0]['servId'])
```

- `source_first`도 동일 BM25/Dense/RRF/규칙을 사용. 순위·서비스 수를 설명 생략 때문에 바꾸지 않음.
- source_first 카드: `explanation_pending=True`, `explanation_kind='source'`.
- 상세 설명 성공: pending=False, kind=model. 근거 검증 실패 템플릿은 kind=source.
- 설명 요청은 ready이고 현재 결과에 있는 ID이며 검색 당시 원문 입력·확정 프로필이 일치할 때만 허용.
- UI는 결과 카드에 설명을 저장해 다시 열어도 추가 생성하지 않음. 새 검색·대화에는 자동 공유하지 않음.
- `reopen_confirm()`은 과거 snapshot으로 답을 되돌리기보다 현재 세션을 복사해 확인 단계로 돌리고 questions/queries를 새 라운드로 초기화. accepted actions·프로필은 유지. 위기는 우회 불가.
- Streamlit 세션만 있어 ‘이번 방문’과 대화는 영구 기록이 아님. 새로운 이야기에서는 방문 목록 유지, 전체 초기화/세션 종료 등에서는 소실 가능.

## 8. 파일별 담당 지점

| 경로 | 책임 |
|---|---|
| `ai/app.py` | CSS·화면·채팅 슬롯·상태 복구·모드 선택·활동 동의·source_first UI |
| `ai/.streamlit/config.toml` | 밝은 테마 |
| `ai/core/models.py` | 실제 모델 로딩, attention_mask 포함 생성, 임베딩 batch 환경변수 |
| `ai/pipeline.py` | prepare/resume/search, 설명 지연·개별 생성·단계 시간 |
| `ai/explain/explain.py` | 원문 근거 설명·문체 지시·fallback |
| `ai/rules/region.py` | 거주지 자격 판정·전국 무주택 오해 수정 |
| `ai/ui_text.py` | 원문 저장값과 분리된 소개 표시 종결형 정리 |
| `ai/chitchat.py` · `ai/prompts/chitchat_system.md` | 일상 대화 판별 규칙, Gemini 호출·기본 답 대체, 지시문 |
| `ai/small_activities.py` | 동의 제안 문장 규칙·활동 조건 선택 |
| `ai/data/small_activities.json` | 원본 HTML에서 가져온 19개 팀 초안 |
| `ai/scripts/real_model_check.py` | 고정 예문의 실제 GPU·모델 smoke |
| `ai/scripts/colab_launch.py` | 현재 proxy origin·프로세스 재시작·링크·미리보기 |
| `ai/scripts/build_runtime_bundle.py` | 인증정보·캐시 제외 번들 생성·해시 검증 |
| `ai/notebooks/streamlit_t4.ipynb` | 설치·GPU·업로드·검증·화면 실행 순서 |
| `ai/requirements-ui.txt` | GPU 없이 화면·가짜 모델 연결 실행 |
| `ai/requirements-streamlit.txt` | 실제 모델 + Streamlit 설치 |
| `ai/requirements-test.txt` | UI·회귀 테스트 추가 의존성 |
| `ai/eval/test_*.py` | 이번 추가 회귀·런타임 테스트 |
| `src/collect/collect_welfare.py`, `gov24.py` | 기존 공식 API 수집. 자동 갱신 파이프라인 아님 |

검증 상세: [회귀·실측 기록](STREAMLIT_VERIFICATION.md).

## 9. 검증 재현

```powershell
python -m pip install -r ai/requirements-test.txt
python -m pytest tests/test_rules_region.py tests/test_retrieval_fusion.py tests/test_stage2_pipeline.py tests/test_stage2_confirm.py tests/test_stage1_devset.py ai/eval/test_service_first.py ai/eval/test_region_residence_scope.py ai/eval/test_small_activities.py ai/eval/test_real_model_adapter.py -q
python -m pytest ai/eval/test_streamlit_handoff.py ai/eval/test_runtime_handoff.py -q
```

기존 관련 묶음은 최초 164 통과·3 실패였고, 세 실패는 로컬 `openpyxl` 미설치였다. 설치 후 해당 3개 모두 통과. 실행 환경을 재현하도록 테스트 requirements에 openpyxl을 명시했다. 최종 추가 회귀 묶음 22개는 모두 통과했다(앞선 묶음과 일부 중복). 이 중 Streamlit 수명주기·번들·런처 테스트 10개도 통과했다. 로컬 테스트는 GPU 실제 추론 품질·최종 T4 화면 속도·모바일 전수 검증을 대체하지 않는다.

## 10. 남은 작업·주의할 회귀

| 우선 | 작업 | 완료 판단 |
|---|---|---|
| 1 | 최신 커밋으로 Colab 재배포 | 상단 실제 배지·현재 지역 표시·부안 사업 서울 제외 확인 |
| 1 | source_first T4 시간 측정 | 검색/규칙/설명/상세 요청 시간을 구분 기록 |
| 1 | 실제 모델 오류·빈 결과·부분 오류 UI 재확인 | 서로 다른 안내, 입력 보존·재시도 확인 |
| 1 | 초기 이해·질문 지연 | prepare/resume 시간 측정. 목록 우선 표시로는 해결되지 않음 |
| 2 | 활동 제안 표현 확장 검수 | 부정문·서비스 필요와 막막함 혼합·활동 거절·재제안 확인 |
| 2 | 지역 문구 추가 검수 | 원문 지역 제한/전국 자산 검사/전국 모집 구별 |
| 2 | 노동·미래설계 0건 | 라벨 기준·서비스 실제 대상 대조 후 데이터 수정 |
| 2 | 둘러보기 label_plain 통일 여부 | 팀 문구 결정 후 일괄 적용 |
| 2 | 모바일·다양한 창 크기 | 카드·사이드바·배지·입력·안전 카드 시각 비교 |
| 3 | 최신 정보 자동 갱신 | 공식 API 수집→변경 감지→검증→마스터·색인 교체 연결 |
| 3 | OpenAI 활용 여부 | 공고 구조화 모델·비용·근거·키 관리·검수 정책 결정 |

원격 CSV를 바꿀 때는 임베딩/BM25 캐시의 데이터·모델 지문 재검증 및 Q1/Q2 addendum 해시 일치도 함께 확인한다. 단순 git pull만으로 실행 중인 Colab의 데이터가 바뀌지는 않는다.

## 11. 기존 명세·남은 백엔드 연구 보존

- [2단계 입출력 계약](../ai/query/STAGE2_IO.md), `configs/query_terms.yaml`의 검수 버전·hold 정책·개발/평가 세트 분리를 그대로 따른다.
- UI의 사전 채팅 질문 최대 3개와 backend `stage2.questions.max_total=2`는 서로 다른 라운드·용도다. 숫자를 하나로 합치거나 사용자가 준 값을 다시 묻는 회귀 주의.
- `memo`는 위기 확인·기록용이며 검색 쿼리 재료에 넣지 않는다.
- 원래 README의 TODO는 [시작 시점 보존본](streamlit-reference/ai_README_before_2026-11-17.md)에 남겼다. 외출·소득·기간 규칙 등 일부는 다른 팀 작업으로 이미 구현되어 있을 수 있어 이 목록을 현재 미구현 사실로 일괄 복사하지 않는다.
- vLLM guided decoding, 보류 쿼리 후보·상담 hold 재검토, 확장·접근 실험, BM25 토큰 목록 직접 입력 등의 원래 연구 항목은 이번 UI 작업에서 완료하지 않았다.

실행 중인 Pipeline은 서비스 DataFrame과 모델을 `st.cache_resource`에서 유지한다. CSV 파일만 교체한 뒤 화면 새로고침으로 최신 DB가 반영된다고 가정하지 않는다. 새 마스터 적용 시 Pipeline 캐시 초기화/앱 재시작까지 배포 절차에 포함해야 한다.
