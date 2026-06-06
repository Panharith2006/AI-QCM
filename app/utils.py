from pathlib import Path
from config import ROOT, DEFAULT_MODEL_PATHS


def resolve_model_path(path_text: str) -> Path:
    path = Path(path_text).expanduser()
    if path.is_absolute():
        return path
    return (ROOT / path).resolve()


def pick_default_model_path() -> str:
    for candidate in DEFAULT_MODEL_PATHS:
        if candidate.exists():
            return str(candidate)
    return str(DEFAULT_MODEL_PATHS[0])
