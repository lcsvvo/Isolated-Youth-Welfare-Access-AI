"""Build an allow-listed, hash-verified Colab runtime ZIP without credentials or caches.

python ai/scripts/build_runtime_bundle.py --output /path/to/bangeol-real-ai.zip
"""
import argparse
import ast
import hashlib
import json
import subprocess
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TEXT_EXTENSIONS = {".py", ".js", ".css", ".yaml", ".yml", ".md", ".txt", ".json", ".toml"}
DATA_NAMES = {"dev_corpus.jsonl", "small_activities.json", "sample_services.csv"}


def runtime_files(root=ROOT):
    root = Path(root).resolve()
    files = []
    for path in (root / "ai").rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(root)
        if any(part in {"__pycache__", ".ipynb_checkpoints", "eval", "tests", "notebooks"} for part in rel.parts):
            continue
        if "data" in rel.parts:
            if path.name not in DATA_NAMES:
                continue
        elif path.suffix.lower() not in TEXT_EXTENSIONS:
            continue
        if path.name.startswith(".env") or "secret" in path.name.lower():
            continue
        files.append(path)
    for name in ("data/processed/services_master.csv", "data/processed/services_master.meta.json",
                 "outputs/03_q1_recommendation/q1_review.json", "outputs/04_q2_recommendation/q2_info.json"):
        path = root / name
        if path.exists():
            files.append(path)
    if not (root / "data/processed/services_master.csv").exists():
        raise FileNotFoundError("data/processed/services_master.csv가 필요합니다")
    return sorted(set(files))


def build_bundle(output, root=ROOT):
    root = Path(root).resolve()
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    paths = runtime_files(root)
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=normal"],
                                            cwd=root, text=True, stderr=subprocess.DEVNULL).strip())
    except (OSError, subprocess.CalledProcessError):
        commit = None
        dirty = None
    manifest = {"schema": "bangeol-runtime-v1", "base_commit": commit, "working_tree_dirty": dirty,
                "files": {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in paths:
            data = path.read_bytes()
            if path.suffix == ".py":
                ast.parse(data.decode("utf-8"))
            archive.writestr(path.relative_to(root).as_posix(), data)
        archive.writestr("bundle-manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
    with zipfile.ZipFile(output) as archive:
        for name, digest in manifest["files"].items():
            if hashlib.sha256(archive.read(name)).hexdigest() != digest:
                raise ValueError(f"ZIP 검증 실패: {name}")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build_bundle(args.output)
    print(f"{args.output.resolve()}: {len(result['files'])} files, {args.output.stat().st_size} bytes")
