from __future__ import annotations

from dataclasses import dataclass, field
import warnings

import cv2
import numpy as np
import easyocr
import torch

from src.detection.yolo_layout import BlockDetection, YoloLayoutDetector, crop_detection
from src.layout.grid_detector import GridDetector
from src.alignment.perspective import perspective_correct
from src.OCR.ocr_preprocess import ensure_bgr, preprocess_for_ocr
from src.OCR.ocr_reader import read_easyocr
from src.OCR.text_normalizer import extract_answer_from_text, post_process_text

warnings.filterwarnings('ignore', category=UserWarning)

# ============================================================
# RESULT STRUCTURE
# ============================================================
@dataclass
class MCQExtractionResult:
    answers: dict[str, str]
    confidence: dict[str, float]
    marked_regions: dict[str, dict] = field(default_factory=dict)
    overall_confidence: float = 0.0
    is_valid: bool = True
    errors: list[str] = field(default_factory=list)
    debug_info: dict = field(default_factory=dict)


# ============================================================
# OCR EXTRACTOR
# ============================================================
class OCRExtractor:

    def __init__(self, yolo_model_path: str, debug: bool = False):

        self.detector = YoloLayoutDetector(yolo_model_path)
        if self.detector.model is None:
            raise RuntimeError(f"YOLO model not found: {yolo_model_path}")

        # Safe GPU handling
        use_gpu = torch.cuda.is_available()
        self.reader = easyocr.Reader(['en'], gpu=use_gpu, verbose=False)

        self.grid_detector = GridDetector(debug=debug)

        self.image = None
        self.image_path = None
        self.debug = debug

        self.grid_visualizations = {}
        self.ocr_allowlist = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789IVXTFNG-_.:/()"

    # ========================================================
    # MAIN PIPELINE
    # ========================================================
    def extract(self, image_path: str, mcq_config: dict | None = None):

        try:
            self.image_path = image_path
            self.image = cv2.imread(image_path)

            if self.image is None:
                raise ValueError(f"Cannot read image: {image_path}")

            config = mcq_config or {}
            if config.get("align", True):
                try:
                    self.image = perspective_correct(self.image)
                    if self.debug:
                        print(f"Aligned document image: {self.image.shape}")
                except Exception as align_error:
                    if self.debug:
                        print(f"Document alignment skipped: {align_error}")

            answer_options = config.get("answer_options", ["A", "B", "C", "D"])
            self.ocr_allowlist = config.get(
                "ocr_allowlist",
                "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789IVXTFNG-_.:/()",
            )

            detections = self._detect_boxes()

            if not detections:
                return MCQExtractionResult(
                    answers={},
                    confidence={},
                    is_valid=False,
                    errors=["No boxes detected"]
                )

            detections = self._crop_boxes_from_detections(detections)

            return self._extract_with_easyocr(
                detections,
                answer_options
            )

        except Exception as e:
            import traceback
            return MCQExtractionResult(
                answers={},
                confidence={},
                is_valid=False,
                errors=[str(e), traceback.format_exc()],
                debug_info={"error_type": type(e).__name__}
            )

    # ========================================================
    # OCR + GRID PROCESSING
    # ========================================================
    def _extract_with_easyocr(self, detections, answer_options):

        answers = {}
        confidence_scores = {}
        marked_regions = {}
        questions_found = 0
        total_detections = len(detections)

        for i, det in enumerate(detections):

            crop = det.crop
            if crop is None or crop.size == 0:
                continue

            q_id = f"q{i + 1}"

            try:
                cells = self.grid_detector.detect_grid(crop)

                # fallback if grid fails
                if not cells:
                    if self.debug:
                        print(f"  No cells detected - using fallback whole-box extraction")
                    text, conf = self._read_easyocr(crop)
                    answer = self._extract_answer_from_text(text, answer_options)

                    answers[q_id] = {
                        "answer": answer,
                        "confidence": conf,
                        "type": "mcq",
                        "num_cells": 1,
                    }
                    confidence_scores[q_id] = conf
                    if answer:
                        questions_found += 1
                    continue

                cell_texts = []
                cell_conf = []

                if self.debug:
                    print(f"  Processing {len(cells)} cells from Q{i+1}...")

                # process grid cells - pass BGR directly to EasyOCR, NOT preprocessed binary
                for cell_idx, cell in enumerate(cells):
                    cell_bgr = self._ensure_bgr(cell.crop)
                    
                    # Log cell details for debugging
                    if self.debug:
                        print(f"    Cell[{cell_idx}] ({cell.row},{cell.col}): crop_shape={cell.crop.shape if cell.crop is not None else 'None'}, bgr_shape={cell_bgr.shape}, coords=[{cell.x1}:{cell.x2},{cell.y1}:{cell.y2}]")
                    
                    text, conf = self._read_easyocr(cell_bgr)
                    
                    if self.debug:
                        print(f"      → OCR: '{text}' (conf={conf:.2f})")

                    cell_texts.append(text)
                    cell_conf.append(conf)

                # Debug: show ALL cell results clearly
                if self.debug:
                    print(f"  Cell extraction summary:")
                    for idx, (txt, conf) in enumerate(zip(cell_texts, cell_conf)):
                        status = "✓" if txt else "✗"
                        print(f"    [{status}] Cell {idx}: '{txt}' (conf={conf:.2f})")

                # Debug: show combined results
                combined = " ".join([t for t in cell_texts if t])
                cleaned = self._post_process(combined)
                
                if self.debug:
                    print(f"  Combined from {len(cells)} cells: raw='{combined}' → cleaned='{cleaned}'")
                    print(f"  Cell confidences: {cell_conf} → avg={float(np.mean([c for c in cell_conf if c > 0])) if any(c > 0 for c in cell_conf) else 0.0:.2f}")

                # Generate grid visualization
                try:
                    grid_vis = self.grid_detector.visualize_grid(crop, cells)
                    self.grid_visualizations[q_id] = grid_vis
                    if self.debug:
                        print(f"  Grid visualization created for {q_id}")
                except Exception as e:
                    if self.debug:
                        print(f"Could not generate grid viz: {e}")

                final_answer = self._extract_answer_from_text(cleaned, answer_options)

                valid_conf = [c for c in cell_conf if c > 0]
                avg_conf = float(np.mean(valid_conf)) if valid_conf else 0.0

                answers[q_id] = {
                    "answer": final_answer,
                    "confidence": avg_conf,
                    "type": "mcq",
                    "num_cells": len(cells),
                }
                confidence_scores[q_id] = avg_conf
                
                if final_answer:
                    questions_found += 1

                marked_regions[q_id] = {
                    "raw_cells": cell_texts,
                    "combined": combined,
                    "cleaned": cleaned,
                    "confidence": avg_conf,
                    "num_cells": len(cells),
                    "extraction_method": "grid_detection + easyocr"
                }

            except Exception as e:
                if self.debug:
                    print(f"Error in {q_id}: {e}")

        overall = float(np.mean(list(confidence_scores.values()))) if confidence_scores else 0.0

        return MCQExtractionResult(
            answers=answers,
            confidence=confidence_scores,
            marked_regions=marked_regions,
            overall_confidence=overall,
            is_valid=len(answers) > 0,
            debug_info={
                "mode": "yolo+grid+easyocr",
                "total_boxes_detected": total_detections,
                "extraction_method": "yolo → grid_detection → easyocr",
                "questions_found": questions_found,
                "questions_processed": len([d for d in detections if d.crop is not None and d.crop.size > 0]),
            }
        )

    # ========================================================
    # PREPROCESSING PIPELINE
    # ========================================================
    def _preprocess_for_ocr(self, img, use_threshold: bool = False):
        return preprocess_for_ocr(img, use_threshold=use_threshold, debug=self.debug)

    def _crop_to_content(self, img):
        from src.OCR.ocr_preprocess import crop_to_content
        return crop_to_content(img, debug=self.debug)

    def _resize_preserve_aspect_ratio(self, img, min_side: int = 128, max_side: int = 320):
        from src.OCR.ocr_preprocess import resize_preserve_aspect_ratio
        return resize_preserve_aspect_ratio(img, min_side=min_side, max_side=max_side, debug=self.debug)

    def _apply_light_threshold(self, img):
        from src.OCR.ocr_preprocess import apply_light_threshold
        return apply_light_threshold(img, debug=self.debug)

    def _read_easyocr(self, img):
        return read_easyocr(self.reader, img, self.ocr_allowlist, debug=self.debug)

    # ========================================================
    # ANSWER EXTRACTION
    # ========================================================
    def _extract_answer_from_text(self, text, options):
        return extract_answer_from_text(text, options)

    # ========================================================
    # DETECTION + CROP
    # ========================================================
    def _detect_boxes(self):
        dets = self.detector.detect(self.image, conf=0.3)
        dets.sort(key=lambda d: (d.y1, d.x1))
        return dets

    def _crop_boxes_from_detections(self, detections):
        for d in detections:
            d.crop = crop_detection(self.image, d)
        return detections

    # ========================================================
    # IMAGE HELPERS
    # ========================================================
    def _ensure_bgr(self, img):
        return ensure_bgr(img)

    # ========================================================
    # CLEAN TEXT
    # ========================================================
    def _post_process(self, text):
        return post_process_text(text)