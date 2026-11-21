# Git 브랜치 및 협업 가이드

`main`에 직접 작업하지 않고, 작업별 브랜치를 만들어 Pull Request로 합친다.

## 기본 흐름

```text
main 최신화 → 작업 브랜치 생성 → 수정·검증 → commit → push → Pull Request → 검토 후 main에 merge
```

## 1. 처음 한 번만: 저장소 복제

GitHub Collaborator 초대를 수락한 뒤 실행한다.

```bash
git clone https://github.com/lcsvvo/Isolated-Youth-Welfare-Access-AI.git
cd Isolated-Youth-Welfare-Access-AI
```

커밋 작성자가 본인 GitHub 계정으로 표시되도록 이름과 이메일을 확인한다. 이메일은 GitHub 계정에 등록된 주소 또는 GitHub noreply 주소를 사용한다.

```bash
git config user.name
git config user.email
```

## 2. 작업 시작 전: main 최신화

```bash
git switch main
git pull origin main
```

수정 중인 파일이 있어 `git switch`가 실패하면 먼저 커밋하거나 임시 저장한다. 다른 사람의 변경을 덮어쓰는 강제 명령은 사용하지 않는다.

## 3. 작업 브랜치 만들기

```bash
git switch -c feature/작업명
```

| 접두사 | 용도 | 예시 |
| --- | --- | --- |
| `feature/` | 수집, 분석, 기능 추가 | `feature/collect-welfare-api` |
| `fix/` | 코드·계산 오류 수정 | `fix/labeling-parser` |
| `docs/` | README 등 저장소 문서 변경 | `docs/update-readme` |
| `chore/` | 환경 설정과 구조 정리 | `chore/add-dependencies` |

브랜치 이름은 영문 소문자와 하이픈을 사용한다. 서로 다른 작업은 하나의 브랜치에 섞지 않는다.

## 4. 확인 후 커밋

```bash
git status
git diff
```

API 키, `.env`, 개인정보, 가상환경, 생성 결과물이 포함되지 않았는지 확인하고 필요한 파일만 추가한다. 원자료와 전처리 데이터는 추가하지 않는다.

```bash
git add src/collect/파일명.py
git status
```

`git add .`을 사용했다면 커밋 전에 stage 목록과 내용을 다시 확인한다.

```bash
git diff --cached --name-status
git diff --cached
```

커밋 메시지는 변경 목적이 보이도록 작성한다.

- `feat: add welfare service API collector`
- `fix: correct region code mapping`
- `docs: update setup instructions`
- `chore: add analysis dependency`

AI 도구를 사용한 경우에도 커밋 메시지와 Pull Request에 AI 관련 문구를 넣지 않는다. 자세한 규칙은 [AGENTS.md](AGENTS.md)를 따른다.

## 5. 작업 브랜치 push

```bash
git push -u origin feature/작업명   # 처음
git push                             # 이후
```

## 6. Pull Request 만들기

GitHub에서 push한 브랜치의 **Compare & pull request**를 선택한다. base는 `main`, compare는 본인 작업 브랜치다.

설명에는 다음을 적는다.

1. 작업 목적
2. 변경한 파일과 핵심 내용
3. 사용한 데이터와 출처
4. 실행 또는 검증 방법
5. 확인이 필요한 부분

검토가 끝나면 `main`에 merge하고 작업 브랜치는 삭제해도 된다.

## 7. 작업 중 main 변경사항 반영

```bash
git switch main
git pull origin main
git switch feature/작업명
git merge main
```

충돌이 나면 충돌 표시가 있는 파일에서 필요한 내용을 남기고 저장한 뒤 마무리한다.

```bash
git add 충돌을_해결한_파일
git commit
git push
```

어느 내용을 남겨야 할지 불분명하면 임의로 삭제하지 말고 해당 파일을 수정한 팀원과 확인한다.

## 8. merge 후 정리

```bash
git switch main
git pull origin main
git branch -d feature/작업명
```

## 파일 위치 규칙

| 파일 종류 | 위치 | GitHub 공유 |
| --- | --- | --- |
| 재사용 Python 코드 | `src/<모듈>/` | 추적 |
| STEP·FINAL 노트북 | `notebooks/<stage_id>.ipynb`, `notebooks/FINAL_<주제>.ipynb` | 출력 크기와 민감정보 확인 후 추적 |
| 코드 검증 | `ai/eval/test_*.py` | 추적 |
| 원자료 | `data/raw/<출처>/` | Git 제외, 출처와 배치 경로는 `data/README.md` |
| 정제·결합 데이터 | `data/processed/` | Git 제외 |
| 결과표 원본 | `outputs/<stage_id>/tables/` | 검토 완료된 표만 추적 |
| 결과그림 원본 | `outputs/<stage_id>/figures/` | 검토 완료된 그림만 추적 |
| 단계별 HTML 요약 | `outputs/<stage_id>/summary/` | 추적 |
| 검증 기록 | `logs/validation/` | 필요한 기록만 추적, `*.log`는 제외 |
| 개인 작업 | `private/` | Git 제외 |

노트북·결과표·그림·HTML은 [docs/RESEARCH_OUTPUT_STANDARD.md](docs/RESEARCH_OUTPUT_STANDARD.md)를 따른다. 이 저장소는 표준의 `results/` 대신 `outputs/`를 쓴다.

데이터를 수집·변경하면 [data/README.md](data/README.md)의 카탈로그를 함께 갱신한다. 기획서·제안서·발표자료는 Notion에서 관리한다.

저장소를 공개하기 전에는 `data/raw/`의 원자료를 Git 이력에서도 제거하고, `.gitignore`의 `!data/raw/**` 공유 예외를 해제한다. 현재 브랜치에서 폴더를 삭제하는 것만으로는 과거 커밋의 원자료가 사라지지 않는다.

- 노트북 이름은 단계 ID와 같게 한다(`01_eda.ipynb`). `notebooks/`에는 `.ipynb`만 둔다. 여러 곳에서 쓰는 Python 코드는 `src/`에 둔다.
- 파일을 잘못된 위치에 만들었다면 복사하지 말고 `git mv`로 옮긴다.
- 파일·폴더 이름은 영문 소문자·숫자·밑줄만 쓴다. 한글 파일명은 macOS와 Windows에서 다르게 저장되어 같은 파일이 중복으로 보일 수 있다.
- 대소문자만 다른 파일명을 만들지 않는다. macOS와 Windows는 기본적으로 대소문자를 구분하지 않는다.
- 개인 PC의 절대경로 대신 저장소 기준 상대경로를 사용한다. 경로 구분자는 `pathlib`로 처리한다.
- Notebook은 Pull Request 전에 위에서 아래까지 다시 실행한다.
- 검증 전 수치나 해석을 확정된 결론처럼 작성하지 않는다.

## 사용하지 않는 명령

`git reset --hard`, `git push --force`, 추적 중인 파일의 강제 삭제는 다른 사람의 작업이나 이력을 잃게 할 수 있으므로 팀 합의 없이 사용하지 않는다.
