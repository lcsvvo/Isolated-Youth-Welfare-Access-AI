"""평가 카드 80장(홀드아웃) 실행 전 설정 동결 기록.

holdout_v1을 돌리기 직전에 같은 환경(Kaggle)에서 한 번 실행한다. 결과 해석에 쓰이는 코드·설정·환경을
eval/runs/<run>/freeze.json에 남기고, 커밋되지 않은 변경이 있으면 멈춘다.

실행 (ai/ 폴더에서)
  python -m eval.freeze_record --run holdout_v1
  python -m eval.freeze_record --run holdout_v1 --allow-dirty   # 변경 내용도 기록하고 진행
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import time
from importlib import metadata
from pathlib import Path

import yaml

AI = Path(__file__).resolve().parents[1]
RUNS = AI / "eval" / "runs"
# 1단계 결과를 바꿀 수 있는 파일
FROZEN = ["configs/config.yaml", "configs/crisis_terms.yaml", "configs/domains.yaml",
          "extract/postprocess.py", "extract/extract.py", "extract/schema.py",
          "safety/crisis.py", "core/models.py", "data/dev_corpus.jsonl", "data/holdout_cards.jsonl"]
PACKAGES = ["torch", "transformers", "accelerate", "sentence-transformers", "FlagEmbedding",
            "numpy", "pandas", "pyyaml"]


def sha256(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], cwd=AI, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def gpu() -> str | None:
    try:
        import torch
        return torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    except ImportError:
        return None


def version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def collect() -> dict:
    cfg = yaml.safe_load(open(AI / "configs" / "config.yaml", encoding="utf-8"))
    files = {f: sha256(AI / f) for f in FROZEN}
    files.update({p.relative_to(AI).as_posix(): sha256(p) for p in sorted((AI / "prompts").glob("*"))
                  if p.is_file()})
    status = git("status", "--porcelain", "--", ".")
    return {
        "recorded_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "commit": git("rev-parse", "HEAD"),
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": status.splitlines(),
        "models": cfg.get("models"),
        "extract": cfg.get("extract"),
        "files_sha256": files,
        "packages": {p: version(p) for p in PACKAGES},
        "python": platform.python_version(),
        "gpu": gpu(),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="holdout_v1")
    ap.add_argument("--allow-dirty", action="store_true", help="커밋 안 된 변경이 있어도 기록하고 진행")
    args = ap.parse_args()
    rec = collect()
    if rec["files_sha256"]["data/holdout_cards.jsonl"] is None:
        raise SystemExit("holdout_cards.jsonl 없음 → python -m eval.build_devset --holdout 먼저 실행")
    tracked = [s for s in rec["dirty"] if not s.startswith("??")]
    if tracked and not args.allow_dirty:
        raise SystemExit("커밋 안 된 변경 있음 (동결 불가):\n" + "\n".join(tracked))
    out = RUNS / args.run / "freeze.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        raise SystemExit(f"{out} 이미 있음. 동결 기록은 실행마다 한 번만 남김")
    out.write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"동결 기록 → {out}")
    print(f"- commit {rec['commit'][:7]} ({rec['branch']}), GPU {rec['gpu']}, python {rec['python']}")
    print(f"- LLM {rec['models']['llm']}, extract {rec['extract']}")
    print(f"- packages {', '.join(f'{k} {v}' for k, v in rec['packages'].items() if v)}")


if __name__ == "__main__":
    main()
