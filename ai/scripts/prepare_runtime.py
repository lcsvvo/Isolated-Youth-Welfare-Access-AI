"""Build and validate final Q1/Q2 runtime data from the checked-out repository.

Run after installing ai/requirements-ui.txt, before starting the actual models.
No model downloads or API requests are made here.
"""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
AI = ROOT / "ai"
sys.path.insert(0, str(AI))
MANIFEST = "outputs/runtime_data/manifest.json"
ADDENDA = {
    "q1": ("src.gap_index.q1_recommendation", "outputs/03_q1_recommendation/q1_review.json"),
    "q2": ("src.preprocess.q2_recommendation", "outputs/04_q2_recommendation/q2_info.json"),
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def input_fingerprint(root=ROOT):
    # Include reviewed inputs and their producer code. Never inspect credentials,
    # personal files, models, generated runtime output, or retrieval caches.
    patterns = ["src/**/*.py", "ai/core/data.py", "ai/core/review_data.py",
                "logs/validation/**/*.csv", "outputs/02_labeling/tables/*.csv",
                "outputs/04_q2/tables/final/*.csv", "data/raw/gov24/*.csv",
                "data/raw/gov24/_collection.json", "data/processed/services_master.*",
                "data/processed/q2_items_v3_1.csv", "data/processed/q2_match.*"]
    paths = sorted({p for pattern in patterns for p in root.glob(pattern) if p.is_file()})
    inputs = {p.relative_to(root).as_posix(): sha256(p) for p in paths}
    return hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()


def prepare(root=ROOT, *, rebuild=False):
    from core.data import load_services
    from core.review_data import load_addendum
    import yaml

    root = Path(root).resolve()
    master = root / "data/processed/services_master.csv"
    services = load_services(master)
    fingerprint = input_fingerprint(root)
    manifest_path = root / MANIFEST
    previous = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    exports = {}
    for kind, (module, relative) in ADDENDA.items():
        path = root / relative
        old_hash = (previous.get("exports") or {}).get(kind)
        reusable = (not rebuild and previous.get("inputs_sha256") == fingerprint
                    and path.exists() and old_hash == sha256(path))
        if not reusable:
            print(f"Building {kind.upper()} reviewed data...", flush=True)
            subprocess.run([sys.executable, "-m", module], cwd=root, check=True)
        rows = load_addendum(path, "q1-review-v1" if kind == "q1" else "q2-info-v1",
                             master, set(services.servId))
        exports[kind] = sha256(path)
        print(f"{kind.upper()}: {len(rows)} services validated", flush=True)
    config = yaml.safe_load((root / "ai/configs/config.yaml").read_text(encoding="utf-8"))
    report = {"schema": "bangeol-runtime-data-v1", "inputs_sha256": fingerprint,
              "master_sha256": sha256(master), "services": len(services),
              "models": config["models"], "dense_profile": config["retrieval"]["dense_profile"],
              "exports": exports}
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()
    print(json.dumps(prepare(rebuild=args.rebuild), ensure_ascii=False, indent=2))
