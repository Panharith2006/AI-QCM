from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from src.OCR.ocr_extractor import OCRExtractor
from src.scoring.compare import compute_metrics


@dataclass
class PipelineResult:
    
    # Structured result of the OMR pipeline, including extracted answers, confidence scores, and debug info.
    answers: dict[str, dict] = field(default_factory=dict)
    extracted_answers: list[tuple[str, str, float, str]] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
    debug_info: dict = field(default_factory=dict)
    is_valid: bool = True
    errors: list[str] = field(default_factory=list)


class OMRPipeline:
    # Main pipeline class to process OMR sheets using YOLO for layout detection and EasyOCR for answer extraction.
    def __init__(
        self,
        model_path: str,
        fill_threshold: float = 0.3
    ):
        self.model_path = model_path
        self.fill_threshold = fill_threshold
        self.debug = False
        # Initialize OCRExtractor with YOLO model (EasyOCR is pre-trained)
        self.extractor = OCRExtractor(
            yolo_model_path=model_path
        )
        
    # Process an input image (file path) and extract answers, returning a structured PipelineResult.
    def process_image(
        self,
        image_input: str | np.ndarray,
        question_mapping: dict[str, dict] | None = None,
        expected_question_ids: list[str] | None = None,
        ocr_config: dict | None = None,
    ) -> PipelineResult:
        if not isinstance(image_input, str):
            raise ValueError("OMRPipeline expects an image file path when using OCR extraction.")

        config = ocr_config or {}
        answer_options = config.get("answer_options", [])

        ocr_result = self.extractor.extract(
            image_input,
            {
                "answer_options": answer_options,
                "fill_threshold": self.fill_threshold,
                "ocr_allowlist": config.get(
                    "ocr_allowlist",
                    "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789IVXTFNG-_.:/()",
                ),
            },
        )

        extraction_type = "mcq" if answer_options else "ocr"

        result = PipelineResult(
            is_valid=ocr_result.is_valid,
            errors=list(ocr_result.errors),
            debug_info=dict(ocr_result.debug_info),
        )
        result.debug_info["detected_questions"] = len(ocr_result.answers)
        result.debug_info["extraction_type"] = extraction_type

        extracted_items = list(ocr_result.answers.items())
        expected_ids = expected_question_ids or [f"Q{idx + 1}" for idx in range(len(extracted_items))]

        for idx, qid in enumerate(expected_ids):
            if idx < len(extracted_items):
                q_key, answer_dict = extracted_items[idx]
                
                # answer_dict now contains: {"answer": ..., "confidence": ..., "type": ..., "num_cells": ...}
                answer = answer_dict.get("answer") if isinstance(answer_dict, dict) else answer_dict
                confidence = answer_dict.get("confidence", 0.0) if isinstance(answer_dict, dict) else ocr_result.confidence.get(q_key, 0.0)
                num_cells = answer_dict.get("num_cells", 0) if isinstance(answer_dict, dict) else 0
                
                result.extracted_answers.append((qid, answer, confidence, extraction_type))
                result.answers[qid] = {
                    "answer": answer,
                    "confidence": confidence,
                    "type": extraction_type,
                    "num_cells": num_cells,
                }
            else:
                result.answers[qid] = {
                    "answer": None,
                    "confidence": 0.0,
                    "type": extraction_type,
                    "num_cells": 0,
                }

        return result

    def process_sheet_comparison(
        self,
        student_image_input: str | np.ndarray,
        teacher_answer_map: dict[str, str],
        question_mapping: dict[str, dict] | None = None,
        expected_sheet_class: str | None = None,
        ocr_config: dict | None = None,
    ) -> dict:
        expected_question_ids = list(teacher_answer_map.keys())
        result = self.process_image(student_image_input, question_mapping, expected_question_ids, ocr_config=ocr_config)
        student_map = result.answers

        metrics = compute_metrics(student_map, teacher_answer_map)
        result.metrics = metrics

        class_validation = {
            "detected_class": result.debug_info.get("extraction_type", "ocr"),
            "expected_class": expected_sheet_class,
            "class_match": True if expected_sheet_class is None else expected_sheet_class == result.debug_info.get("extraction_type", "ocr"),
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
        self.extractor.debug = True
