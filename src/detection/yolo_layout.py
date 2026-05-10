from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

try:
    from ultralytics import YOLO
except Exception:  # pragma: no cover - optional runtime dependency
    YOLO = None

# Data class to represent detected blocks in the document layout.
@dataclass
class BlockDetection:
    label: str
    confidence: float
    x1: int
    y1: int
    x2: int
    y2: int
    text: str = ""
    text_confidence: float = 0.0
    crop: np.ndarray = field(default_factory=lambda: np.array([]))  # Store cropped image

# YOLO-based layout detector for document images.
class YoloLayoutDetector:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        if YOLO is not None and Path(model_path).exists():
            self.model = YOLO(model_path)

    def detect(self, image: np.ndarray, conf: float = 0.25) -> list[BlockDetection]:
        if self.model is None:
            return []

        results = self.model.predict(source=image, conf=conf, verbose=False)
        out: list[BlockDetection] = []
        for result in results:
            names = result.names
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cls_id = int(box.cls[0].item())
                score = float(box.conf[0].item())
                
                det = BlockDetection(
                    label=str(names.get(cls_id, cls_id)),
                    confidence=score,
                    x1=int(x1),
                    y1=int(y1),
                    x2=int(x2),
                    y2=int(y2),
                )
                
                # Trust YOLO detection - already filtered by confidence threshold
                out.append(det)
        
        return out

# Utility function to crop a detected block from the image, ensuring coordinates are within bounds.    
def crop_detection(image: np.ndarray, det: BlockDetection) -> np.ndarray:
    h, w = image.shape[:2]
    x1 = max(0, min(det.x1, w - 1))
    y1 = max(0, min(det.y1, h - 1))
    x2 = max(0, min(det.x2, w))
    y2 = max(0, min(det.y2, h))
    return image[y1:y2, x1:x2].copy()
