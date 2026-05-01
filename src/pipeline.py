from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from src.OCR.ocr_extractor import OCRExtractor
from src.scoring.compare import compute_metrics


@dataclass
class PipelineResult:
    answers: dict[str, dict] = field(default_factory=dict)
    extracted_answers: list[tuple[str, str, float, str]] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
    debug_info: dict = field(default_factory=dict)
    is_valid: bool = True
    errors: list[str] = field(default_factory=list)


class OMRPipeline:
    def __init__(self, model_path: str, trocr_model: str = "microsoft/trocr-small-printed", fill_threshold: float = 0.3):
        self.model_path = model_path
        self.fill_threshold = fill_threshold
        self.debug = False
        self.extractor = OCRExtractor(model_path, trocr_model=trocr_model)

    def process_image(
        self,
        image_input: str | np.ndarray,
        question_mapping: dict[str, dict] | None = None,
        expected_question_ids: list[str] | None = None,
    ) -> PipelineResult:
        if not isinstance(image_input, str):
            raise ValueError("OMRPipeline expects an image file path when using OCR extraction.")

        ocr_result = self.extractor.extract(
            image_input,
            {
                "answer_options": ["A", "B", "C", "D"],
                "fill_threshold": self.fill_threshold,
            },
        )

        result = PipelineResult(
            is_valid=ocr_result.is_valid,
            errors=list(ocr_result.errors),
            debug_info=dict(ocr_result.debug_info),
        )
        result.debug_info["detected_questions"] = len(ocr_result.answers)

        extracted_items = list(ocr_result.answers.items())
        expected_ids = expected_question_ids or [f"Q{idx + 1}" for idx in range(len(extracted_items))]

        for idx, qid in enumerate(expected_ids):
            if idx < len(extracted_items):
                _, answer = extracted_items[idx]
                confidence = ocr_result.confidence.get(f"q{idx + 1}", 0.0)
                result.extracted_answers.append((qid, answer, confidence, "mcq"))
                result.answers[qid] = {
                    "answer": answer,
                    "confidence": confidence,
                    "type": "mcq",
                }
            else:
                result.answers[qid] = {
                    "answer": None,
                    "confidence": 0.0,
                    "type": None,
                }

        return result

    def process_sheet_comparison(
        self,
        student_image_input: str | np.ndarray,
        teacher_answer_map: dict[str, str],
        question_mapping: dict[str, dict] | None = None,
        expected_sheet_class: str | None = None,
    ) -> dict:
        expected_question_ids = list(teacher_answer_map.keys())
        result = self.process_image(student_image_input, question_mapping, expected_question_ids)
        student_map = result.answers

        metrics = compute_metrics(student_map, teacher_answer_map)
        result.metrics = metrics

        class_validation = {
            "detected_class": "mcq",
            "expected_class": expected_sheet_class,
            "class_match": True if expected_sheet_class is None else expected_sheet_class == "mcq",
        }

        return {
            "student_answers": student_map,
            "teacher_answers": teacher_answer_map,
            "metrics": metrics,
            "class_validation": class_validation,
            "debug": result.debug_info,
            "image_aligned": None,
            "image_detections": None,
        }

    def enable_debug(self):
        self.debug = True
