import json
from pathlib import Path

from config import (
    CIRCLE_REFERENCE_FILE,
    BUBBLE_REFERENCE_FILE,
    LEGACY_REFERENCE_FILE,
    TABLE_REFERENCE_FILE,
)


def _resolve_reference_file(reference_type: str = "circle") -> Path:
    normalized = str(reference_type).strip().lower()

    if normalized in {"bubble", "normal", "default"}:
        return BUBBLE_REFERENCE_FILE
    if normalized == "circle":
        return CIRCLE_REFERENCE_FILE
    if normalized == "table":
        return TABLE_REFERENCE_FILE

    return BUBBLE_REFERENCE_FILE


def save_reference(answers_grid: list[list[str]], reference_type: str = "circle") -> bool:
    reference_file = _resolve_reference_file(reference_type)

    with open(reference_file, "w") as f:
        json.dump(answers_grid, f, indent=2)
    return True


def load_reference(reference_type: str = "circle") -> list[list[str]] | None:
    reference_file = _resolve_reference_file(reference_type)

    if reference_file.exists():
        with open(reference_file, "r") as f:
            return json.load(f)

    if reference_file == BUBBLE_REFERENCE_FILE and LEGACY_REFERENCE_FILE.exists():
        with open(LEGACY_REFERENCE_FILE, "r") as f:
            return json.load(f)

    return None
