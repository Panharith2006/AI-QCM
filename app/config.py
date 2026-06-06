
from pathlib import Path

# Root directory and file paths
ROOT = Path(__file__).resolve().parents[1]
LEGACY_REFERENCE_FILE = ROOT / "reference_answers.json"
CIRCLE_REFERENCE_FILE = ROOT / "circle_reference_answers.json"
TABLE_REFERENCE_FILE = ROOT / "table_reference_answers.json"
BUBBLE_REFERENCE_FILE = LEGACY_REFERENCE_FILE
REFERENCE_FILE = CIRCLE_REFERENCE_FILE

# YOLO model paths
DEFAULT_MODEL_PATHS = [
    ROOT / "artifacts" / "yolo" / "best.pt",
]

# Streamlit page configuration
PAGE_CONFIG = {
    "page_title": "AI OMR",
    "layout": "wide"
}

# UI constants
MIN_ROWS = 1
MAX_ROWS = 5
MIN_COLS = 1
MAX_COLS = 20
DEFAULT_ROWS = 2
DEFAULT_COLS = 10

# Answer display settings
MAX_DISPLAY_COLS = 6
INTERPRETATION_MAX_COLS = 3

# Scoring thresholds
SCORE_GOOD_THRESHOLD = 70
SCORE_OKAY_THRESHOLD = 50
