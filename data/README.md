# Data

이 저장소에는 원자료와 전처리 데이터를 포함하지 않음. 아래 출처에서 자료를 받아 지정 경로에 두고 전처리 코드를 실행함.

## Sources

| 자료 | 출처, 데이터셋 ID | 배치 경로 | 공개 범위 |
|---|---|---|---|
| 중앙부처 복지서비스 목록, 상세 | 한국사회보장정보원, 공공데이터포털 `15090532` | `data/raw/ssis/central_*.csv` | 공개 API, 수집 코드 `src/collect/` |
| 지자체 복지서비스 목록, 상세 | 한국사회보장정보원, 공공데이터포털 `15108347` | `data/raw/ssis/local_*.csv` | 공개 API, 수집 코드 `src/collect/` |
| 보조금24 공공서비스 | 행정안전부, 공공데이터포털 `15113968` | `data/raw/gov24/` | 공개 API, 수집 코드 `src/collect/gov24.py` |
| 2023 고립, 은둔 청년 실태조사 원자료 | 보건복지부, 한국보건사회연구원, 공공데이터포털 `15156074` | `data/raw/isolated_youth_survey_2023/` | 재배포하지 않음 |
| 2024 청년의 삶 실태조사 | 국무조정실, MDIS | `data/external/` | 마이크로데이터 재배포 금지 |

- API 키는 `.env`에 두며 저장소에 커밋하지 않음. 예시는 [`.env.example`](../.env.example)
- `data/processed/`는 아래 명령으로 다시 생성함

## Processing

| 출력 | 내용 | 생성 |
|---|---|---|
| `services_master.csv` | SSIS 서비스 통합 마스터 5,288건과 접근성 라벨, 근거 문장 | `python -m src.preprocess.ssis_master` |
| `demand_2023_*.csv` | 은둔 표본 4,448명, 고립 보조집단 3,032명의 지원 필요도(응답자 단위) | `python -m src.preprocess.survey_2023` |
| `gov24_*.csv` | 보조금24 목록, 상세, 지원조건 병합과 청년 대상 필터 | `python -m src.preprocess.gov24_youth` |
| `q2_*.csv` | SSIS와 보조금24 동일사업 후보 매칭, 항목 명시 판정 | `python -m src.preprocess.q2_match` |

- 서비스 마스터 컬럼 규격은 [docs/services_master_spec.md](../docs/services_master_spec.md)
- 응답자 단위 자료는 비가중, 비확률 표본이라 모집단 비율로 해석하지 않음
