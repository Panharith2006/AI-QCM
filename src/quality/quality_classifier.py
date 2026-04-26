from __future__ import annotations

import cv2
import numpy as np


def simple_quality_check(image: np.ndarray) -> tuple[str, float]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    mean_intensity = float(np.mean(gray))

    if lap_var < 60:
        return "blurry", 0.45
    if mean_intensity < 60:
        return "dark", 0.55
    if mean_intensity > 210:
        return "overexposed", 0.55
    return "clear", 0.9
