from __future__ import annotations

from dataclasses import dataclass, field
import re

import cv2
import numpy as np
import easyocr
import torch

from src.alignment.perspective import perspective_correct
from src.detection.yolo_layout import YoloLayoutDetector, crop_detection
from src.OCR.ocr_preprocess import ensure_bgr, preprocess_for_ocr
from src.OCR.ocr_reader import read_easyocr
from src.pipeline import PipelineResult


@dataclass
class BubbleAnalysis:
    question_id: str
    answer: str | None
    confidence: float
    score_map: dict[str, float] = field(default_factory=dict)
    num_choices: int = 4
    is_ambiguous: bool = False


class CircleFillPipeline:
    def __init__(
        self,
        model_path: str,
        fill_threshold: float = 0.12,
        align: bool = True,
    ):
        self.model_path = model_path
        self.fill_threshold = fill_threshold
        self.align = align
        self.detector = YoloLayoutDetector(model_path)
        self.reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available(), verbose=False)
        self.ocr_allowlist = "ABCDEFGHIJKLMNOPQRSTUVWXYZ()-.\/"
        self.debug = False
        self.choice_visualizations: dict[str, np.ndarray] = {}

    def process_image(
        self,
        image_input: str | np.ndarray,
        expected_question_ids: list[str] | None = None,
    ) -> PipelineResult:
        if isinstance(image_input, str):
            image = cv2.imread(image_input)
            if image is None:
                return PipelineResult(
                    is_valid=False,
                    errors=[f"Cannot read image: {image_input}"],
                    debug_info={"mode": "circle", "error": "image_read_failed"},
                )
        else:
            image = image_input.copy()

        if self.align:
            try:
                aligned = perspective_correct(image)
                if aligned is not None and aligned.size > 0:
                    image = aligned
            except Exception:
                if self.debug:
                    print("Circle-fill alignment skipped due to an exception")

        detections = self.detector.detect(image)
        detections.sort(key=lambda det: (det.y1, det.x1))

        result = PipelineResult(
            is_valid=bool(detections),
            debug_info={
                "mode": "circle",
                "total_boxes_detected": len(detections),
                "fill_threshold": self.fill_threshold,
            },
        )

        if not detections:
            result.is_valid = False
            result.errors.append("No question regions detected")
            return result

        extracted_items = []
        expected_ids = expected_question_ids or [f"Q{idx + 1}" for idx in range(len(detections))]

        for idx, det in enumerate(detections):
            qid = expected_ids[idx] if idx < len(expected_ids) else f"Q{idx + 1}"
            crop = crop_detection(image, det)
            analysis = self._analyze_crop(crop, qid)

            result.answers[qid] = {
                "answer": analysis.answer,
                "confidence": analysis.confidence,
                "type": "circle",
                "num_cells": analysis.num_choices,
                "is_ambiguous": analysis.is_ambiguous,
                "score_map": analysis.score_map,
            }
            result.extracted_answers.append((qid, analysis.answer, analysis.confidence, "circle"))
            extracted_items.append(analysis)

        result.debug_info["detected_questions"] = len(extracted_items)
        result.debug_info["questions_processed"] = len(extracted_items)
        result.debug_info["answers_found"] = sum(1 for item in extracted_items if item.answer is not None)
        result.debug_info["mode_detail"] = "yolo+omr"
        return result

    def _analyze_crop(self, crop: np.ndarray, qid: str) -> BubbleAnalysis:
        if crop.size == 0:
            return BubbleAnalysis(qid, None, 0.0, num_choices=0)

        text_candidates = []
        search_regions = [crop[:, : max(1, int(crop.shape[1] * 0.5))], crop]

        for region in search_regions:
            region = ensure_bgr(region)
            text, conf = read_easyocr(self.reader, region, self.ocr_allowlist, debug=self.debug)
            if text:
                text_candidates.append((text, conf))

        if not text_candidates:
            return BubbleAnalysis(qid, None, 0.0, score_map={}, num_choices=0)

        best_text, best_conf = max(text_candidates, key=lambda item: item[1])
        cleaned = self._normalize_text(best_text)
        answer = self._extract_parenthesized_answer(cleaned)

        if answer is None:
            answer = self._extract_labeled_answer(cleaned)

        score_map = self._extract_answer_scores(cleaned)
        self.choice_visualizations[qid] = crop.copy()

        if self.debug:
            print(f"  OCR crop text for {qid}: '{best_text}' -> '{cleaned}' -> answer='{answer}'")

        return BubbleAnalysis(
            question_id=qid,
            answer=answer,
            confidence=float(best_conf) if answer is not None else 0.0,
            score_map=score_map,
            num_choices=len(score_map) if score_map else 0,
            is_ambiguous=False,
        )

    def _normalize_text(self, text: str) -> str:
        if not text:
            return ""

        text = text.upper()
        text = text.replace("B..", "B.")
        text = text.replace("D/", "D.")
        text = text.replace("C/", "C.")
        text = text.replace("A/", "A.")
        text = text.replace("B/", "B.")
        text = text.replace("  ", " ")
        return " ".join(text.split())

    def _extract_parenthesized_answer(self, text: str) -> str | None:
        if not text:
            return None

        match = re.search(r"\(([A-E])\)", text)
        if match:
            return match.group(1)

        match = re.search(r"\b\(?\s*([A-E])\s*\)?\b", text)
        if match:
            return match.group(1)

        return None

    def _extract_labeled_answer(self, text: str) -> str | None:
        if not text:
            return None

        match = re.search(r"\b([A-E])\s*[\.)/]", text)
        if match:
            return match.group(1)

        match = re.search(r"[\(\[]\s*([A-E])\s*[\)\]]", text)
        if match:
            return match.group(1)

        return None

    def _extract_answer_scores(self, text: str) -> dict[str, float]:
        scores: dict[str, float] = {}
        for letter in "ABCDE":
            if f"({letter})" in text:
                scores[letter] = 1.0
            elif re.search(rf"\b{letter}\s*[\.)/]", text):
                scores[letter] = 0.8
            elif re.search(rf"\b{letter}\b", text):
                scores[letter] = 0.5
        return scores