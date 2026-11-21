# Streamlit 기술 구조도

기준일: 2026-11-17. 실선은 현재 연결, 별도 ‘미구현’ 도식은 향후 요구다. 실행 방법은 [인수인계](STREAMLIT_HANDOFF.md) 참조.

## 화면·대화 흐름

```mermaid
flowchart TD
    H[첫 화면 / 자유 입력] --> P[prepare: 위기 확인 + 욕구·조건 추출]
    P --> C{위기인가?}
    C -- 예 --> S[안전 안내 / 서비스·활동 중단]
    C -- 아니오 --> O{막막함·무기력 표현인가?}
    O -- 예 --> A[작은 활동부터 볼지 먼저 질문]
    A --> Y[활동 보기 동의]
    A --> N[서비스 먼저 찾기]
    A --> R[쉬어가기]
    Y --> F[범위·오늘의 여유·상호작용 직접 선택]
    F --> AC[규칙으로 활동 최대 3개]
    AC --> N
    AC --> R
    O -- 아니오 --> Q[채팅: 지역·외출·나이·전화 등]
    N --> Q
    Q --> M[resume: 자유 답은 memo로 위기 재확인]
    M --> C2{위기인가?}
    C2 -- 예 --> S
    C2 -- 아니오 --> K[정리표 / 확인·수정·건너뛰기]
    K --> FIN[finish_confirm]
    FIN --> B{백엔드 상태}
    B -- awaiting_answer --> BQ[추가 질문 / resume answer]
    BQ --> B
    B -- explore_only --> U[서비스 먼저 탐색 / 활동 선택 진입]
    B -- ready --> SEARCH[search source_first]
    SEARCH --> RS[검색 결과 / 원문 소개 먼저]
    RS --> D[서비스 상세 / 원문과 신청 경로]
    D --> EX[사용자가 AI 설명 요청]
    EX --> E[explain_service: 현재 결과·프로필 검증]
    E --> CACHE[세션 카드에 설명 저장·재사용]
    D --> V[이번 방문에 담기]
    K --> BACK[조건 재편집: 현재 답 보존]
    RS --> BACK
    BACK --> K
```

안전 화면에는 사용자가 누르는 연락 링크만 있다. 자동 통화·신고·출동 요청·위치/대화 전송 기능은 없다. 위기 세션에서 메뉴를 눌러도 결과·상세·방문·탐색·활동 화면으로 우회하지 못한다.

## 모델·검색 흐름

```mermaid
flowchart LR
    T[사용자 입력] --> KA[Kanana: 위기 + 욕구 추출]
    KA --> CP[confirmed_profile]
    CP --> QB[쿼리 생성]
    QB --> BM[Kiwi + BM25]
    QB --> DE[bge-m3 + Dense]
    BM --> RR[RRF]
    DE --> RR
    RR --> RULE[나이·시도/시군구·접근·기간 등 규칙]
    MASTER[(서비스 마스터 CSV)] --> BM
    MASTER --> DE
    MASTER --> RULE
    RULE --> TOP[최종 TOP 5]
    TOP --> RAW[원문 소개 / 즉시 결과 경로]
    TOP -. 명시 요청 시 .-> REASON[Kanana 설명 / 원문 근거 검증]
    REASON --> EVID{근거가 원문에 있는가?}
    EVID -- 예 --> GEN[생성 설명]
    EVID -- 아니오 --> FALL[원문 템플릿]
```

UI는 `search(..., explain_mode='source_first')`를 호출한다. 다른 연구·스크립트의 기존 `search(session)`은 AI 설명 5개를 생성하는 기본값을 유지한다. 표시는 빨라져도 초기 추출·위기 확인은 여전히 모델 작업이다.

## 현재 배포 구조

```mermaid
flowchart LR
    CL[팀원 로컬 Git checkout] --> PACK[build_runtime_bundle.py]
    PACK --> ZIP[allow-list ZIP + SHA256 manifest]
    ZIP --> NB[Colab notebook upload]
    NB --> GPU[T4 runtime]
    GPU --> ST[Streamlit server :8501]
    GPU --> K[Kanana]
    GPU --> BG[bge-m3]
    ST --> K
    ST --> BG
    ST --> DB[(services_master.csv + caches)]
    BROWSER[사용자 Chrome] <-->|현재 Colab proxy origin / WebSocket| PROXY[Colab notebook proxy]
    PROXY <--> ST
```

- PC의 localhost 앱과 T4의 실제 모델 앱은 서로 다른 서버다.
- 최종 연결은 cloudflared·OpenAI API·외부 추론 서버를 사용하지 않는다.
- 런타임 종료 시 `/content` 파일과 모델 캐시가 사라질 수 있다.
- `st.cache_resource`로 한 앱 프로세스에서 모델을 재사용한다. 사전 검증은 별도 프로세스 종료 후 화면을 띄워 모델 이중 적재를 피한다.
- Colab proxy는 매번 다시 얻는다. health 응답과 실제 앱 렌더는 별도로 확인한다.

## 상태 계약

| 상태 | 허용 작업 | 검색 |
|---|---|---|
| awaiting_confirm | 확인 조작·memo·finish_confirm | 하지 않음 |
| awaiting_answer | pending 질문에 answer·memo | 하지 않음 |
| ready | 검색·현재 결과 상세 설명 | 허용 |
| explore_only | 직접 탐색, 확인 라운드 다시 열기 | 맞춤 추천 하지 않음 |
| crisis_stop | 정적 안전 안내·새 이야기 | 서비스·활동 차단 |

기존 오류는 정리표 페이지와 backend 상태가 어긋나 ‘새로운 이야기’를 요구하던 것이다. 이제 현재 session의 accepted actions/confirmed_profile을 유지하여 확인 라운드를 다시 연다. 과거 snapshot으로 뒤늦은 답변을 덮어쓰지 않는다.

## 상세 설명 계약

```python
result = pipe.search(session, explain_mode='source_first')
# pending 카드만 사용자가 요청할 때 아래 호출
card = pipe.explain_service(session, result, serv_id)
# UI가 result['cards']의 해당 카드를 교체
```

조건 변경·위기·검색 결과에 없는 ID의 설명 요청은 거부한다. 소개와 생성 설명의 출처 구분은 `explanation_kind`, 생성을 기다리는지는 `explanation_pending`으로 관리한다.

## 자동 갱신 요구 — 아래 흐름은 미구현

```mermaid
flowchart LR
    TR[주기 실행 / 미연결] -.-> API[공식 복지·공공서비스 API 수집기]
    API -.-> DIFF[서비스 ID·원문 변경 감지 / 미구현]
    DIFF -.-> STRUCT[공고 구조화 / OpenAI 등 후보, 미연결]
    STRUCT -.-> VALID[원문 근거·날짜·지역·중복 검증 / 통합 미구현]
    VALID -.-> REVIEW[불확실 항목 검수]
    REVIEW -.-> RELEASE[새 마스터 버전·색인·부가자료 검증 후 교체]
    RELEASE -.-> APP[앱에 반영]
```

현재 `src/collect/collect_welfare.py`, `src/collect/gov24.py`는 공식 API 수집의 시작점이다. 이미 받은 원문 캐시를 재사용하는 동작도 있어 단순 반복 실행만으로 변경 감지가 되리라 가정하지 않는다. ‘AI가 최신 서비스를 알아서 가져온다’고 설명하지 않는다. OpenAI는 공고 구조화에 사용할 수 있는 후보이며, 주기 실행·수집·변경 감지·반영을 대신하는 기능으로 연결되지 않았다.
