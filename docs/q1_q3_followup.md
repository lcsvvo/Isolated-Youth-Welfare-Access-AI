# Q1~Q3 추천 연결 및 독립 평가 준비

기준: 최초 [main ed28066](https://github.com/lcsvvo/Isolated-Youth-Welfare-Access-AI/commit/ed28066cd8ef0442ef701d3c6b7e574075079b3a), 작업 중 [R4 제출 PR #66](https://github.com/lcsvvo/Isolated-Youth-Welfare-Access-AI/pull/66)이 병합된 `dff0048`로 갱신. 변경된 R4 시트는 Q1~Q3 입력으로 사용하지 않음. [전체 파이프라인 검토](https://app.notion.com/p/419dad625c5e80068adbf2faf772bc62)의 Q1~Q3 및 미해결 항목 참조. 서비스 마스터는 SSIS 5,288건·155열, 생성 코드 `556847e`. 원본·기존 노트북·기존 분석표는 읽기 전용. 아래 산출물은 별도 단계에 생성하며 Git에는 추가하지 않음.

RRF 통합 기준은 `feature/rrf-retrieval-hardening`의 `af17e24`.

## Q1: 추천용 분류 레이어

- [Q1 최종 판정 #42](https://github.com/lcsvvo/Isolated-Youth-Welfare-Access-AI/pull/42)의 사람 검수 236건과 불일치 라벨 87개 합의, 원문 접근성 검토 330건을 `servId`로 연결. #42의 마스터와 현재 마스터는 동일한 Git blob임.
- 조사 Q21 라벨에서 [앱 9개 영역 대응표](https://app.notion.com/p/40fdad625c5e80e88a1de81ccd7e5138)의 핵심 대응만 다시 파생. 일반 돌봄은 직접 회복으로 합치지 않음. 교육은 실제 학습·훈련 및 교육 목적 비용, 경제 main은 생계·필수 반복비용 근거를 사용. Q1 결론이나 자동/수동 검수 원본은 갱신하지 않음.
- 노동과 미래설계에는 핵심 Q21 매핑이 없으므로 두 영역에 대한 미판정을 5점 감점 사유로 사용하지 않음. 별도 노동·미래설계 핵심 태그 확정에는 운영 기준과 서비스별 원문 검수가 필요함. ⑧ 구직을 재직 노동으로, 부분 대응을 핵심으로 자동 승격하지 않음.
- 경제 candidate 1,284건 모두 기존 경제 영역 비교에서 감점됨. 이 중 1,082건은 앱 핵심 분류가 아예 없고 202건에는 다른 앱 영역이 있음. candidate는 경제 핵심과 분리하고, 확정 불일치와 같은 감점을 주지 않음. 원문의 사용처가 명확히 다른 일부 사례는 경제 제외를 유지함.
- 기준 규칙의 주거(q21_11) 추가(명시적 주거 지원·원룸 지원)도 Q1 라벨에 이월함(2026-11-16). 사람 판정이 우선이며 q21_01·q21_03 차이는 이월하지 않음. 변경 4건: WLF00001322·WLF00002298·WLF00002826·WLF00005990 주거 핵심 추가. 주거 핵심 474→478, 주거 감점 4,813→4,809. 경제 수치(감점 3,494, candidate 3, 미분류 1)와 36개 순위 비교 Top-5는 변화 없음. WLF00005990은 경제 제외 유지
- 재검수 대상(라벨 미변경): WLF00006298 자동차사고 피해자 지원사업의 경제(h_q21_10) '아니오'. 급여에 재활보조금·생활자금 대출이 있으나 단독 검수 근거가 방문돌봄·생활지원 인용임
- RRF 결합 뒤 주 욕구 확정 일치에 순위 점수 1점 우대 적용. 최종 점수 동점에서도 확정 일치를 먼저 배치함. 주 욕구 미판정은 영역 점수 0, 불일치는 기존 감점 5점 적용. 보조 욕구 미판정으로 인한 주 욕구 감점 해제 방지. 현행 1점은 민감도 분석 후에도 유지함.
- 신청·이용은 별도 사람 판정과 근거로 보존함. 물리적 접근성 검토는 별도 정보이며 연령·지역 자격, 모집 여부, 방문 필수 여부를 자동 확정하거나 제외하지 않음. 양육생활비의 일반화, 생계 목적이 모호한 수당, 현물·비용 보전의 돌봄 범위, 용도 미상 축하금 등 개별 합의 밖 경계는 검토 대기.
- 노동·미래설계 승인 핵심 후보 12건(`logs/validation/q1_labor_future_review/labor_future_final_review.csv`)만 핵심 영역에 추가해 주 욕구 일치 가점 1점 적용, 판정 출처 ai_assisted, 미판정 유지(감점 0점), 부분 후보 기록만, 신규 경제·구직 등 권고 미반영, 자동 규칙 확대 없음

```powershell
python -m src.gap_index.q1_recommendation
# 기존 비교표를 보존하며 현재 순위를 다시 계산
python -m src.gap_index.q1_recommendation --out outputs/03_q1_recommendation/rrf_followup
```

결과: `outputs/03_q1_recommendation/q1_review.json` 및 `tables/`의 영역별 건수, 대상 ID별 감점 차이, BM25 후보와 후보 재정렬 비교. 9영역 × 질의 2개 × 프로필 2개를 사용하고 질의·텍스트·후보 목록을 고정함. 검색과 재정렬 변화를 구분하며 관련성 우열은 판단하지 않음.

우대 점수 민감도 분석:

- 현행 우대 1점 유지. 0·0.5·1·2점을 같은 BM25 상위 100개 후보, 질의·프로필 36건에 적용해 비교함. 감점 5점과 동점 처리는 고정하고 `filter.py` 기본값은 바꾸지 않음.
- 구성 변화: Top-5 서비스 집합이 다름. 순서 변화: Top-5 목록(순서 포함)이 다름(구성 변화 포함). Top-1 변화: 1위 서비스가 다름.
- 자격 판정 통과 집합은 모든 우대 점수에서 동일함(변화 0건).

| 비교 | 구성 변화 | 순서 변화 | Top-1 변화 |
| --- | --- | --- | --- |
| 0점 대비 0.5점 | 1 | 4 | 0 |
| 0점 대비 1점 | 1 | 9 | 0 |
| 0점 대비 2점 | 3 | 10 | 5 |
| 1점 대비 0점 | 1 | 9 | 0 |
| 1점 대비 0.5점 | 0 | 7 | 0 |
| 1점 대비 2점 | 2 | 9 | 5 |

- 주거 라벨 이월과 노동·미래설계 승인 핵심 후보 12건 반영 후 기준. 반영 전 순서 변화는 0점 대비 1점 7건, 1점 대비 2점 8건

- 사람 관련성 판정이 없어 추천 정확도 개선을 주장하지 않음. 민감도 분석 결과임.

```powershell
python -m src.gap_index.q1_bonus_sensitivity --out outputs/03_q1_recommendation/bonus_sensitivity_labor_future
```

결과: `outputs/03_q1_recommendation/bonus_sensitivity_labor_future/tables/bonus_top5.csv`, `bonus_summary.csv`, `bonus_sensitivity.json`.

## Q2: 동일 사업에 근거를 연결

- [Q2 판정 #33](https://github.com/lcsvvo/Isolated-Youth-Welfare-Access-AI/pull/33)과 [최종 분석 #43](https://github.com/lcsvvo/Isolated-Youth-Welfare-Access-AI/pull/43)의 판정·배정·사후정정 키로 A/B/C 38쌍 추출. 사람 판정이 ‘동일’인 U 1쌍은 기존 최종 분석 범위 밖이고, 지역 범위가 달라 사후 ‘다름’으로 정정된 1쌍도 제외함.
- 항목별 양쪽 원문, 원문 URL, 수집·수정일, 명시성 규칙·근거를 보존함. 있는 경우 사람의 본문 명시성 판정 및 출처 차이 최종 판정을 별도 보존함. 서식 링크나 ‘상시신청’ 기본 표현만으로 제출서류·현재 신청 가능 여부를 단정하지 않음.
- `supplement_possible`은 같은 사업에서 확인한 보완 단서임. `source_conflict`는 기존 사람이 실제 충돌로 판단한 항목, `source_check_required`는 확인 불가와 미정리 차이, `freshness_check_required`는 신청기간 등 현재 확인이 필요한 항목임. 모든 레코드의 `auto_fill`은 false. 2026-10-30·31 수집본과 기존 공식 확인 이력을 연결했으며 현재 운영 여부는 별도 확인 필요.

```powershell
python -m src.preprocess.q2_recommendation
```

결과: `outputs/04_q2_recommendation/q2_info.json`, `tables/confirmed_ssis_gov24.csv`, `tables/field_evidence.csv`, `tables/field_status_counts.csv`. 보조금24 사업은 검색 인덱스에 추가하지 않음.

기본 파이프라인은 위의 생성 파일을 찾으면 같은 마스터 해시를 검증해 연결함. 생성 전에는 기본 마스터만 사용. `Pipeline(..., services_path=...)` 또는 CLI `--services`로 마스터를 지정할 때는 `--q1-review`와 `--q2-info`로 연결할 파일을 지정. 설정 파일을 따로 쓸 때는 `paths.q1_review`, `paths.q2_info`를 지정하거나 비워 둠. 지정한 JSON의 마스터 해시가 다르면 로딩 오류 표시.

추천 카드의 `q1_access_review`에서 신청·이용 검토를, `q2_source_comparison.items`에서 항목별 상태 문구·양쪽 근거·URL·확인일을 제공함. 비교 정보는 추천 이유·신청방법 생성에 전달하지 않음. 카드에 표시할 때 `status_label`과 `freshness_label`을 양쪽 출처와 같이 사용.

## Q3: 기술 기준선과 독립 평가

```powershell
python -m src.recommend.q3_rebaseline
python -m src.recommend.q3_independent
```

- 30쌍 의도 검수에 따라 주 24쌍(동일·직접 비교), 보조 27쌍(동일), 기존 비교용 30쌍으로 집계. 기존 단순 키워드 8조건과 동점 진단을 재사용. `outputs/05_q3_rebaseline/tables/`에 검색 건수·0건·Top-10 교집합·동점 교체 및 같은 쌍으로 재집계한 이전 값과의 차이를 저장.
- 기존 4인 × 20개 작성 슬롯을 재사용해 새 시트에 배정. 평가 본문과 정답은 사람이 작성. 작성자는 검색 결과와 개발 문구를 보지 않음. 다른 사람이 개발용 입력·Q3 v1/v2·2단계 및 Dense 인계 질의·새 카드끼리 유사성을 검토. 표면 유사성 0.70 이상은 수정 후 재확인. 수치 미만도 의미가 사실상 같으면 제외. 독립성 판정 후 입력·프로필과 실행 설정을 동결.
- `outputs/05_q3_independent/review/`의 작성·검수표는 80개 슬롯이며 관련성/자격 스키마는 빈 양식. CSV에는 헤더 1줄과 데이터만 저장. `card_review.csv`는 별도 작성자·검수자, 판정 근거, 기대 동작(search/clarify/stop), 앱 영역(`|` 구분)과 확인된 프로필을 받음. 모르는 프로필은 비워 둠.
- `revise` 또는 `exclude`된 카드는 `python -m src.recommend.q3_independent --cards 기존작성표.csv --reviews 기존검수표.csv --replace 카드ID --out private/새작성표`로 교체 작업을 시작함. 새 표에서 해당 슬롯의 작성문과 검수란을 비우며 유형·배정과 앞선 파일은 보존함. 새 본문 작성 뒤 다시 검수함.
- `ai.eval.independent_retrieval pool`은 `--cards`, `--reviews`, `--out private/<새 실행명>`으로 실행. 기존 구성대로 승인 80장 필요. 검색 카드는 동일 본문과 확인된 프로필을 사용함. `needs`의 첫 영역을 주 욕구로, 이후 영역을 보조 욕구로 사용. 운영 파이프라인과 동일한 RRF 및 후보 제한 설정 적용. 검색기 오류가 있으면 평가 풀 생성 중단. Dense는 지정 모델을 로컬 캐시에서 사용.
- 각 검색 방식의 상위 10(최종 추천은 설정상 최대 5)의 합집합을 블라인드 판정. 관련성은 0 무관, 1 간접, 2 직접. 자격 시트에 확인된 사용자 조건을 표시하고 compatible/incompatible/uncertain으로 판정함. `score --run ... --relevance ...`로 관련성 검수 완료분의 Top-5 지표 산출. 직접 관련 서비스의 자격 검수 완료 후 같은 명령에 `--eligibility ...`를 더해 자격 지표 산출. 관련성 파일 해시별 폴더에 결과 저장. Recall은 검수 풀 한정이며 분모 0은 결측.

- Q3 독립 평가 설정 미고정(2026-11-16 기준 평가 카드 80장 본문 미작성). Q1 최종 병합 후 `python -m src.gap_index.q1_recommendation`으로 재생성한 운영용 `q1_review.json`과 당시 설정을 기준으로 고정·평가함.

독립 평가의 카드 작성·사람 판정, Dense 최종 모델 선택, Q2 공식 원문 최신성 확인은 후속 작업임.
