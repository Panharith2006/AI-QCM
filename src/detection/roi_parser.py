from __future__ import annotations

import cv2
import numpy as np


def compute_fill_score(roi_bgr: np.ndarray) -> float:
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return float(cv2.countNonZero(binary)) / float(binary.size)


def choose_option(fill_scores: dict[str, float], min_threshold: float = 0.12, margin: float = 0.03) -> tuple[str, float]:
    if not fill_scores:
        return "UNANSWERED", 0.0

    sorted_items = sorted(fill_scores.items(), key=lambda kv: kv[1], reverse=True)
    top_label, top_score = sorted_items[0]

    if top_score < min_threshold:
        return "UNANSWERED", top_score

    if len(sorted_items) > 1 and (top_score - sorted_items[1][1]) < margin:
        return "AMBIGUOUS", top_score

    return top_label, top_score
