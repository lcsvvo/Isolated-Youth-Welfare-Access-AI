# 저장소 구조

```text
data/
├── raw/<출처>/                  # 원자료 (커밋 제외, README·수집 로그·_metadata만 공유)
└── processed/                   # 정제·결합 데이터 (커밋 제외)
src/
├── collect/                     # 공공데이터 수집
├── preprocess/                  # 정제·청년 대상 필터링
├── labeling/                    # 서비스 접근성(외출 필요도·신청 방식) 분류와 평가
├── recommend/                   # 서술형 입력 이해·서비스 검색·추천
├── gap_index/                   # 지역별 수요·비대면 공급 집계와 공백 지수
└── app/                         # 청년용 화면·정책 담당자용 대시보드
notebooks/                       # STEP·FINAL 노트북 (.ipynb 전용)
outputs/<stage_id>/              # 단계 수행 시 생성
├── tables/                      # 결과표 원본 (CSV)
├── figures/                     # 결과그림 원본
└── summary/<stage_id>_summary_final.html
logs/
└── validation/                  # 일치도·민감도·재현성·QA
docs/                            # 산출물 작성 표준
private/                         # 개인 작업 공간 (커밋 제외)
```

## 단계 ID

노트북, `outputs/`, HTML 요약에서 같은 ID를 쓴다.

| stage_id | 노트북 | 내용 |
| --- | --- | --- |
| `00_data_preparation` | `notebooks/00_data_preparation.ipynb` | SSIS 수집 결과 확인·정제 |
| `00_gov24_preparation` | `notebooks/00_gov24_preparation.ipynb` | 보조금24 원자료 무결성·자격조건·신청경로 |
| `01_eda` | `notebooks/01_eda.ipynb` | 탐색적 분석 |
| `02_labeling` | `notebooks/02_labeling.ipynb` | 서비스 라벨·앱 9개 영역 대응 |
| `03_q1` | `notebooks/03_q1.ipynb` | Q1 지원 수요와 복지서비스 후보 비교 |
| `04_q2` | `notebooks/04_q2.ipynb` | Q2 플랫폼 간 동일사업 매칭과 정보 명시성 |
| `05_q3` | `notebooks/05_q3.ipynb` | Q3 일상어·행정어 검색 한계 |
| `06_gap_index` | `notebooks/06_gap_index.ipynb` (예정) | 지역별 공백 지수 (기존 04에서 이동) |
| `07_recommend` | `notebooks/07_recommend.ipynb` (예정) | 추천·위기 감지 평가 (기존 05에서 이동) |
| - | `notebooks/FINAL_<주제>.ipynb` | 통합 |

연구질문과 단계 번호의 대응: Q1 = 03_q1, Q2 = 04_q2, Q3 = 05_q3.

Q3 검색 결과·검수 시트·노트북 산출물의 공식 경로는 `outputs/05_q3/`이다.

`outputs/<stage_id>/`와 하위 폴더는 해당 단계를 실제로 수행할 때 만든다. 빈 폴더를 미리 만들지 않는다.

작성 규칙은 [docs/RESEARCH_OUTPUT_STANDARD.md](docs/RESEARCH_OUTPUT_STANDARD.md)를 따른다. 이 저장소는 표준의 `results/` 대신 `outputs/`를 쓴다.
