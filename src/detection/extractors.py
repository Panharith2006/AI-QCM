from __future__ import annotations

import cv2
import numpy as np
from dataclasses import dataclass


@dataclass
class ExtractionResult:
    """Result from region-level extraction."""
    answer: str
    confidence: float
    fill_scores: dict[str, float]
    debug_info: dict = None


def extract_mcq_block(roi_bgr: np.ndarray, option_labels: list[str] | None = None) -> ExtractionResult:
    """Extract MCQ answer from a detected MCQ block.
    
    Process:
    1. Convert to grayscale
    2. Threshold to binary
    3. Detect contours (circles/bubbles)
    4. Measure fill ratio per option
    5. Return highest filled bubble
    
    Args:
        roi_bgr: Cropped block image (BGR)
        option_labels: List of option labels (default: A, B, C, D)
    
    Returns:
        ExtractionResult with selected answer
    """
    if option_labels is None:
        option_labels = ["A", "B", "C", "D", "E"]
    
    # Convert and threshold
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Find contours (bubbles)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Sort contours by x-position (left to right)
    contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])
    
    # Extract fill scores for each option
    fill_scores = {}
    for idx, contour in enumerate(contours[:len(option_labels)]):
        x, y, w, h = cv2.boundingRect(contour)
        bubble_roi = binary[y:y+h, x:x+w]
        
        if bubble_roi.size == 0:
            fill_scores[option_labels[idx]] = 0.0
        else:
            fill_ratio = cv2.countNonZero(bubble_roi) / bubble_roi.size
            fill_scores[option_labels[idx]] = float(fill_ratio)
    
    # Choose option
    if not fill_scores:
        return ExtractionResult("UNANSWERED", 0.0, {}, {"reason": "no_contours"})
    
    sorted_items = sorted(fill_scores.items(), key=lambda kv: kv[1], reverse=True)
    top_label, top_score = sorted_items[0]
    
    min_threshold = 0.12
    margin = 0.03
    
    if top_score < min_threshold:
        return ExtractionResult("UNANSWERED", top_score, fill_scores, {"reason": "below_threshold"})
    
    if len(sorted_items) > 1 and (top_score - sorted_items[1][1]) < margin:
        return ExtractionResult("AMBIGUOUS", top_score, fill_scores, {"reason": "too_close_to_second"})
    
    return ExtractionResult(top_label, top_score, fill_scores, {"reason": "selected"})


def _generate_roman_numerals(count: int) -> list[str]:
    """Generate Roman numerals from I to count.
    
    Args:
        count: Number of Roman numerals to generate (1-100)
    
    Returns:
        List of Roman numeral strings
    """
    roman_map = [
        (1, 'I'), (4, 'IV'), (5, 'V'), (9, 'IX'), (10, 'X'),
        (40, 'XL'), (50, 'L'), (90, 'XC'), (100, 'C'),
    ]
    
    def int_to_roman(num: int) -> str:
        result = ""
        for value, numeral in reversed(roman_map):
            count = num // value
            if count:
                result += numeral * count
                num -= value * count
        return result
    
    return [int_to_roman(i) for i in range(1, count + 1)]


def extract_roman_block(roi_bgr: np.ndarray, num_options: int = 10) -> ExtractionResult:
    """Extract Roman numeral answer from Roman matching block.
    
    Process:
    1. Segment image into rows
    2. Detect marked areas per row
    3. Map row index to Roman numeral
    
    Args:
        roi_bgr: Cropped block image
        num_options: Number of Roman numeral options (1-100). Default 10 (I-X)
                    Examples: 5 for I-V, 15 for I-XV
    
    Returns:
        ExtractionResult with selected Roman numeral
    """
    roman_numerals = _generate_roman_numerals(min(num_options, 100))
    
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    h, w = binary.shape
    row_height = h // len(roman_numerals)
    
    fill_scores = {}
    for idx, roman in enumerate(roman_numerals):
        row_start = idx * row_height
        row_end = (idx + 1) * row_height if idx < len(roman_numerals) - 1 else h
        
        row_region = binary[row_start:row_end, :]
        fill_ratio = cv2.countNonZero(row_region) / row_region.size if row_region.size > 0 else 0.0
        fill_scores[roman] = float(fill_ratio)
    
    # Select highest filled row
    sorted_items = sorted(fill_scores.items(), key=lambda kv: kv[1], reverse=True)
    top_label, top_score = sorted_items[0]
    
    if top_score < 0.10:
        return ExtractionResult("UNANSWERED", top_score, fill_scores, {"reason": "no_marking"})
    
    return ExtractionResult(top_label, top_score, fill_scores, {"reason": "selected"})


def extract_tfng_block(roi_bgr: np.ndarray, options: list[str] | None = None) -> ExtractionResult:
    """Extract True/False/No Given answer from TFNG block.
    
    Process:
    1. Divide into N regions based on options
    2. Measure fill in each region
    3. Return highest filled region
    
    Args:
        roi_bgr: Cropped block image
        options: Custom option labels (default: ["T", "F", "NG"])
    
    Returns:
        ExtractionResult with selected answer
    """
    if options is None:
        options = ["T", "F", "NG"]
    
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    h, w = binary.shape
    col_width = w // len(options)
    
    fill_scores = {}
    for idx, option in enumerate(options):
        col_start = idx * col_width
        col_end = (idx + 1) * col_width if idx < len(options) - 1 else w
        
        col_region = binary[:, col_start:col_end]
        fill_ratio = cv2.countNonZero(col_region) / col_region.size if col_region.size > 0 else 0.0
        fill_scores[option] = float(fill_ratio)
    
    sorted_items = sorted(fill_scores.items(), key=lambda kv: kv[1], reverse=True)
    top_label, top_score = sorted_items[0]
    
    if top_score < 0.08:
        return ExtractionResult("UNANSWERED", top_score, fill_scores, {"reason": "no_marking"})
    
    return ExtractionResult(top_label, top_score, fill_scores, {"reason": "selected"})


def extract_completion_block(roi_bgr: np.ndarray) -> ExtractionResult:
    """Extract Completion block answer using position-based mapping.
    
    Process:
    1. Detect handwritten/filled regions
    2. Map position to answer options
    3. Return most confident match
    
    Args:
        roi_bgr: Cropped block image
    
    Returns:
        ExtractionResult with completion answer
    """
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Find contours of filled regions
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return ExtractionResult("UNANSWERED", 0.0, {}, {"reason": "no_markings"})
    
    # Get centroid of largest filled region
    largest_contour = max(contours, key=cv2.contourArea)
    m = cv2.moments(largest_contour)
    
    if m["m00"] == 0:
        return ExtractionResult("UNANSWERED", 0.0, {}, {"reason": "invalid_contour"})
    
    cx = int(m["m10"] / m["m00"])
    cy = int(m["m01"] / m["m00"])
    
    # Position-based mapping (example: divide into quadrants)
    h, w = roi_bgr.shape[:2]
    if cx < w // 2:
        answer = "LEFT"
        confidence = 0.7
    else:
        answer = "RIGHT"
        confidence = 0.7
    
    return ExtractionResult(answer, confidence, {"filled_area": float(cv2.contourArea(largest_contour))}, {"centroid": (cx, cy)})


def route_and_extract(roi_bgr: np.ndarray, block_label: str, option_config: dict | None = None) -> ExtractionResult:
    """Router function: dispatch to appropriate extractor based on block type.
    
    Args:
        roi_bgr: Cropped block image
        block_label: Block class from YOLO (e.g., "mcq_block", "roman_block")
        option_config: Optional configuration per block type
    
    Returns:
        ExtractionResult
    """
    if option_config is None:
        option_config = {}
    
    if block_label == "mcq_block":
        options = option_config.get("options", ["A", "B", "C", "D"])
        return extract_mcq_block(roi_bgr, option_labels=options)
    
    elif block_label == "roman_block":
        num_options = option_config.get("num_options", 10)
        return extract_roman_block(roi_bgr, num_options=num_options)
    
    elif block_label == "tfng_block":
        options = option_config.get("options", ["T", "F", "NG"])
        return extract_tfng_block(roi_bgr, options=options)
    
    elif block_label == "completion_block":
        return extract_completion_block(roi_bgr)
    
    else:
        return ExtractionResult("UNKNOWN", 0.0, {}, {"reason": f"unknown_block_type: {block_label}"})
