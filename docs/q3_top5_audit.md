# Q3 후보 검색·Top-5 진단 실행법

기준은 [2026-11-17 Q3 최종 평가](Q3_FINAL_EVALUATION_2026-11-17.md)임. 독립 평가 32장은 원인 분석에만 사용하고, 변경안 비교는 별도의 개발 질의 30개에서 수행함.

## 독립 평가 32장

저장소 루트에서 실행. `$base`는 Q3 입력과 판정 자료를 보관한 저장소 경로이며, Q3 기준 시점과 동일한 마스터·설정·Q1 판정을 사용함.

```powershell
$base = '<보관 저장소의 절대 경로>'
$run = Join-Path $base 'private/q3_run_v1'
$split = Join-Path $run 'review/split_core'
$review = @('소현', '지우', '지홍', '유진_done') | ForEach-Object {
    Join-Path $split "relevance_$_.csv"
}
python -m ai.eval.q3_top5_audit --run $run --subset "$split/subset_with_q1off.json" `
    --relevance $review --master "$base/data/processed/services_master.csv" `
    --q1-review "$base/outputs/03_q1_recommendation/q1_review_before_final_policy.json" `
    --out outputs/05_q3_top5_audit
```

- `candidate_coverage.csv`: 단위는 카드–서비스 쌍. 판정 범위 내 micro recall은 후보 수에 관계없이 기존 2점 쌍 전체를 분모로 사용함. 미판정은 별도 집계하며, 전체 서비스 DB의 Recall로 해석할 수 없음.
- `by_card.csv`: P@5 분모는 카드당 5개. 판정 범위 내 상한은 카드당 `min(5, 기존 2점 후보 수)`의 합을 전체 Top-5 슬롯 수로 나눈 값임.
- `missed_relevant.csv`: 2점인데 최종 Top-5에 없는 후보의 검색 순위, 규칙 재실행 후 순위와 제외 사유. 재실행 Top-5가 저장본과 다른 카드는 규칙 원인을 확정하지 않음. 빈 순위는 해당 순위표에 없음 또는 규칙 결과 미확정을 의미함.
- `review/relevance_top20_new.csv`: BM25·Dense·RRF 각 Top-20 합집합에서 기존 판정 쌍을 뺀 블라인드 양식. `assigned_reviewer`는 카드의 기존 검수자. 1차 검수는 `by_card.csv`에서 `known_relevant=0`인 카드부터 시작함. 관련성 0/2와 자격 확인은 별도 판정임.

## 개발 질의 30개

```powershell
python -m ai.eval.q3_top5_dev --data-root $base --out outputs/05_q3_top5_audit
```

- 개발 30개 보관 순위와 검수 CSV 사용. 후보 비교는 RRF Top-100과 각 검색기 Top-20을 추가한 경우를 포함함.
- 순위 비교는 같은 RRF 후보에 대해 영역 불일치 감점 5를 0으로만 변경함. 명백한 자격 불일치 제외, 나머지 감점, Q1 가점은 유지함.
- `dev_precision_summary.csv`의 P@5 범위는 미판정 후보를 모두 0점 또는 2점으로 둘 때의 범위임. 변화량은 두 방식이 공통으로 추천한 후보를 상쇄한 범위이며, 신뢰구간이 아님.
- 각 결과표는 `outputs/05_q3_top5_audit/tables/`에 저장함. 여기에는 가중치·유의성 검정을 적용하지 않음. 기존 결과를 보존하려면 다시 실행할 때 새 출력 경로를 사용함.

서비스 지원내용을 직접 비교하는 개발 실험은 같은 명령에 `--benefit-text`를 추가함.

- 후보 확보: 실제 사용자 요청으로 `benefit_text`만 BM25 검색하고 Top-20에서 RRF Top-100에 없는 후보를 추가함. 추가 후보의 미판정 쌍은 `review/dev_benefit_recall.csv`에 저장함.
- 순위 재정렬: 기존 RRF Top-100 후보와 자격 규칙을 그대로 적용한 뒤, 지원내용으로 다시 검색했을 때 상위 10개인 통과 후보에만 순위에 따라 최대 5점의 가점을 부여함. Top-5 구성 변화로 인해 추가 판정이 필요한 쌍은 `review/dev_benefit_precision.csv`에 저장함.
- 수치 범위와 새 후보 수는 `tables/dev_benefit_summary.csv`에서 확인함.

10쌍의 사람 판정을 바탕으로 비용 지원만 하는 서비스가 기존 Top-5를 밀어내는 경우를 검토함. 같은 명령에 `--reviewed-precision <검수한 CSV>`를 추가하면 교육훈련·미래설계 요청에서 비용 관련 표현이 없을 때 해당 교체를 보류하고, `tables/dev_benefit_cost_gate.csv`에 개발 질의별 P@5 비교를 저장함. 비교에 사용한 30개 질의에서 규칙을 도출했으므로 탐색 결과로 해석함.

혜택 유형 매칭과 Cross-Encoder는 규칙 통과 후 기존 상위 12개에 적용함. 혜택 유형은 요청과 지원내용에 같은 유형의 근거가 있을 때만 가점을 적용함. Cross-Encoder는 질의 원문과 실제 지원내용을 함께 입력하며, 상위 점수 후보에 순위별 가점을 적용함.

```powershell
python -m ai.eval.q3_benefit_experiment --data-root $base `
    --model-dir <로컬 KURE-Reranker-nano 폴더> --reviewed-precision <검수한 10쌍 CSV> `
    --out outputs/05_q3_top5_audit/benefit_experiment
```

[KURE-Reranker-nano 모델](https://huggingface.co/nlpai-lab/KURE-Reranker-nano) 리비전 `ad02a90a6bb00811bc9bfe9fdf2f0fdb7f89ad64`를 로컬에서 실행함. 결과표의 P@5 차이 범위는 순위가 교체된 미판정 후보에 0점 또는 2점을 줄 때의 최솟값과 최댓값임. 공통 추천의 미판정 후보는 차이 계산에서 상쇄됨. 순위 교체로 인해 추가 판정이 필요한 쌍은 방법별로 `review/`에 저장함.

## 독립 평가 재측정

혜택 유형 매칭을 고정한 뒤 독립 평가 명령의 모듈을 `ai.eval.q3_benefit_holdout`으로 바꾸고, `--out`에는 새 결과 폴더를 지정함. 규칙 재실행의 Top-5가 저장본과 다른 카드는 비교군과 실험군 모두 저장 순위를 사용함. 새 추천 중 미판정 건은 `review/independent_q3_new_top5.csv`로 출력함. 같은 명령에 `--additional-review <완료한 검수 CSV>`와 새 출력 경로를 지정하면 판정 반영 후 P@5를 집계함.
