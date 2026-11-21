"""File versions used to invalidate Streamlit's shared Pipeline cache."""
from pathlib import Path
import yaml

AI = Path(__file__).resolve().parent


def pipeline_inputs(config_path=None, master_path=None):
    config = Path(config_path).resolve() if config_path else AI / "configs/config.yaml"
    cfg = yaml.safe_load(config.read_text(encoding="utf-8"))
    files = {config, AI / "pipeline.py", AI / "runtime_state.py"}
    for name, value in cfg["paths"].items():
        if not value or name in {"cache_dir", "dense_cache", "bm25_cache"}:
            continue
        path = Path(value)
        files.add(path if path.is_absolute() else AI / path)
    if master_path:
        files.add(Path(master_path).resolve())
    for folder in ("core", "extract", "safety", "query", "retrieve", "rules", "confirm", "explain", "configs", "prompts"):
        files.update(p for p in (AI / folder).rglob("*") if p.is_file() and p.suffix in {".py", ".yaml", ".md"})
    version = tuple((str(p.resolve()), p.stat().st_size, p.stat().st_mtime_ns) if p.exists()
                    else (str(p.resolve()), None, None) for p in sorted(files))
    return str(config), str(Path(master_path).resolve()) if master_path else None, version
