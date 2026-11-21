# 1단계 추출 출력 형식 extract-v1

`extract.extract()`가 돌려주는 JSON. 2단계(검색 쿼리), 4단계(규칙 검증), 5단계(설명), 평가 코드가 이 형식을 읽는다.
정의와 검사 코드는 `ai/extract/schema.py`, 테스트는 `tests/test_extract_schema.py`.

```json
{
  "schema_version": "extract-v1",
  "needs": [
    {"domain": "미취업(구직)", "level": "main", "evidence": "취업이 계속 안 돼서", "status": "명시"},
    {"domain": "경제", "level": "sub", "evidence": "돈이 없어", "status": "명시"}
  ],
  "activity_range": "집 근처",
  "activity_evidence": "편의점 말고는 안 나가고",
  "constraints": [
    {"channel": "전화", "level": "부담", "evidence": "전화는 좀 무서워"},
    {"channel": "방문", "level": "불가", "evidence": "센터 방문은 못 해"}
  ],
  "age": 29,
  "region_sido": "대구",
  "next_action": "추천",
  "extract_status": "ok"
}
```

## 필드

| 필드 | 값 | 비어 있을 때 | 비고 |
|---|---|---|---|
| `needs[].domain` | `configs/domains.yaml` 9개 영역 | `[]` | 목록 밖 값은 버림. 같은 영역은 하나만(main 우선) |
| `needs[].level` | `main` / `sub` | — | 이상한 값은 `main` |
| `needs[].status` | `명시` / `확인 필요` | — | 코드가 붙임. 근거가 원문에 없으면 `확인 필요` |
| `activity_range` | 방 안 / 집 안 / 집 근처 / 필요하면 외출 / 판단 불가 | `판단 불가` | 조사 F7 기준. 근거가 원문에 없으면 `판단 불가` |
| `activity_evidence` | 원문 일부 | `null` | |
| `constraints[].channel` | `전화` / `방문` | `[]` | 채널당 하나(불가 우선). 근거가 원문에 없으면 버림 |
| `constraints[].level` | `부담`(선호) / `불가`(필수 제한) | — | |
| `age` | 정수(10–99) | `null` | 글에 나온 나이만 |
| `region_sido` | 시도 약칭 17개(서울, 경기 …) | `null` | 정식 명칭은 약칭으로 바꿈 |
| `next_action` | `추천` / `되묻기` | — | 코드가 붙임. 영역이 하나도 없으면 `되묻기` |
| `extract_status` | `ok` / `failed` | `ok` | 코드가 붙임. LLM 호출 오류·JSON 깨짐·JSON 객체 아님이면 `failed`. 키가 없는 예전 기록은 `ok`로 읽음 (`schema.is_failed`) |

- 위기 여부는 0단계(`safety/crisis.py`) 결과로 따로 둔다. 위기면 1단계를 실행하지 않는다(`추천 중단·상담 안내`).
- LLM 출력이 끝내 JSON이 아니면 빈 결과(`되묻기`)에 `extract_status: failed`를 붙여 돌려준다. 값은 '정보 없음'과 같으므로 구분은 이 필드로 한다. 활동 범위·제약은 원문 규칙(`postprocess`)으로 채워질 수 있다.
- 이전 키 `outing_range`, 제약의 `type`, `대면`/`외출`, `불가능`은 읽을 때만 새 이름으로 바꿔 준다. 새 코드는 쓰지 않는다.

## 정답 라벨과의 대응 (`to_gold_labels`)

| 카드 정답 열 | 추출 결과 |
|---|---|
| `정답_영역` | `needs[].domain` 집합 |
| `정답_활동범위` | `activity_range` |
| `정답_제약` | `constraints` 중 가장 강한 하나: 방문 불가(필수 제한) > 전화 불가 > 방문 부담(선호) > 전화 부담(선호), 없으면 `없음` |
| `정답_나이언급` | `age`가 있으면 Y |
| `정답_지역언급` | `region_sido`가 있으면 Y |
| `정답_위기` | 0단계 `crisis` |
| `기대행동` | 위기 → 추천 중단·상담 안내, 그 밖은 `next_action` |

## 이전 형식에서 바뀐 점

- `outing_range` → `activity_range` (11-06 결정 '활동 범위' 반영), 근거 `activity_evidence` 추가
- 제약: `type`(전화·대면·외출·비용) + `level`(부담·불가능) → `channel`(전화·방문) + `level`(부담·불가). 카드·말뭉치 정답에 비용 제약이 없음
- `next_action`, `schema_version` 추가
- 이 키를 읽는 `rules/filter.py`, `explain/explain.py`, `core/mock.py`도 함께 고침
