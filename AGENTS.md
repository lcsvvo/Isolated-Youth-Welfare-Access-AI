# AI 에이전트 작업 규칙

Claude, Codex 등 AI 도구로 이 저장소에서 작업할 때 따른다.

## 커밋·푸시

- 커밋 author와 committer는 작업을 요청한 팀원 본인의 Git 설정(`user.name`, `user.email`)을 사용한다. AI 이름이나 AI 서비스 이메일로 커밋하지 않는다.
- 커밋 메시지에 `Co-Authored-By`, `Generated with`, 세션 링크 등 AI 관련 trailer나 문구를 넣지 않는다.
- Pull Request 제목·설명·코멘트에도 AI 생성 표시를 넣지 않는다.
- 커밋 전 `git log -1 --format='%an <%ae>%n%cn <%ce>%n%B'`로 작성자와 메시지를 확인한다.
- `main`에 직접 커밋·푸시하지 않고 [CONTRIBUTING.md](CONTRIBUTING.md)의 브랜치·PR 흐름을 따른다.

## 노트북·결과물

- STEP·FINAL 노트북과 결과표·그림은 [docs/RESEARCH_OUTPUT_STANDARD.md](docs/RESEARCH_OUTPUT_STANDARD.md)를 읽고 적용한다. 단계별 HTML 요약은 사용자가 요청한 경우에만 생성한다.
- 이 저장소는 표준의 `results/` 대신 `outputs/`를 쓴다. 결과표와 그림은 각각 `outputs/<stage_id>/tables/`와 `outputs/<stage_id>/figures/`에 저장한다.
- `outputs/<stage_id>/`와 하위 폴더는 해당 단계를 실제로 수행할 때 만든다. 빈 폴더를 미리 만들지 않는다.
- 노트북에 HTML 요약을 만드는 코드와 `# 최종 완료 확인` 섹션을 두지 않는다. 기존 HTML 파일은 지시 없이 다시 생성하지 않는다. 완료 확인과 검증 결과는 작업 보고나 PR 설명에 남긴다.
- 노트북 Markdown(셀과 코드가 출력하는 관찰·해석·한계)은 직관적으로 핵심만 담아 짧게 쓴다. `-` 목록으로 나열하고 명사형 또는 `~음`·`~함`으로 끝낸다. `~다`, `~합니다` 종결과 긴 문단은 쓰지 않는다.

## 저장소 범위

- 코드와 실행에 필요한 파일만 커밋한다. 회의록·일정·기획서·발표자료는 Notion에서 관리한다.
- API 키, `.env`, 개인정보, 생성 결과물은 커밋하지 않는다.
- 원자료(`data/raw/`)와 전처리 데이터(`data/processed/`)는 커밋하지 않는다. 출처와 배치 경로는 `data/README.md`에 적는다.
- 기술 스택이 확정되지 않은 상태에서 프레임워크·패키지를 임의로 추가하지 않는다.
