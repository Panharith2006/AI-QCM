"""MCQ extraction using YOLO detection + TrOCR text recognition + image fill detection."""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from PIL import Image

from src.detection.yolo_layout import BlockDetection, YoloLayoutDetector, crop_detection
from transformers import TrOCRProcessor, VisionEncoderDecoderModel


@dataclass
class MCQExtractionResult:
    """Result from OCR-based MCQ extraction."""
    answers: dict[str, str]  # {"q1": "A", "q2": "B", ...}
    confidence: dict[str, float]  # {"q1": 0.98, "q2": 0.95, ...}
    marked_regions: dict[str, dict] = field(default_factory=dict)
    overall_confidence: float = 0.0
    is_valid: bool = True
    errors: list[str] = field(default_factory=list)
    debug_info: dict = field(default_factory=dict)


class OCRExtractor:
    """Extract MCQ answers using YOLO + TrOCR + image fill detection"""
    
    def __init__(
        self,
        yolo_model_path: str,
        trocr_model: str = "microsoft/trocr-small-printed"
    ):
        """
        Initialize OCR extractor.
        
        Args:
            yolo_model_path: Path to YOLO detection model
            trocr_model: TrOCR model name from HuggingFace
        """
        self.detector = YoloLayoutDetector(yolo_model_path)
        if self.detector.model is None:
            raise RuntimeError(f"YOLO model not found or could not be loaded: {yolo_model_path}")
        
        # Load TrOCR
        try:
            self.processor = TrOCRProcessor.from_pretrained(trocr_model)
            self.trocr_model = VisionEncoderDecoderModel.from_pretrained(trocr_model)
        except Exception as e:
            raise RuntimeError(f"Failed to load TrOCR: {e}")
        
        self.image = None
        self.image_path = None
    
    def extract(
        self,
        image_path: str,
        mcq_config: dict | None = None
    ) -> MCQExtractionResult:
        """
        Extract MCQ answers from worksheet image.
        
        Args:
            image_path: Path to worksheet image
            mcq_config: Configuration dict with:
                - answer_options: list (e.g., ["A", "B", "C", "D"])
                - fill_threshold: float (0-1, default 0.3, darkness threshold for filled box)
                - expected_questions: int (for validation)
        
        Returns:
            MCQExtractionResult with extracted answers
        """
        try:
            self.image_path = image_path
            self.image = cv2.imread(image_path)
            
            if self.image is None:
                raise ValueError(f"Cannot read image: {image_path}")
            
            config = mcq_config or {}
            answer_options = config.get("answer_options", ["A", "B", "C", "D"])
            fill_threshold = config.get("fill_threshold", 0.3)
            
            # Step 1: YOLO detection
            detections = self._detect_boxes()
            
            if not detections:
                return MCQExtractionResult(
                    answers={},
                    confidence={},
                    is_valid=False,
                    errors=["No boxes detected by YOLO"]
                )
            
            # Step 2: Extract text from boxes using TrOCR
            boxes_with_text = self._extract_text_from_boxes(detections)
            
            # Step 3: Group by question
            questions = self._group_boxes_by_question(boxes_with_text, answer_options)
            
            # Step 4: Determine which option is marked (using fill detection)
            answers = {}
            confidence_scores = {}
            marked_regions = {}
            
            for q_id, options in questions.items():
                marked_option, conf = self._find_marked_option(
                    options, fill_threshold
                )
                
                if marked_option:
                    answers[q_id] = marked_option
                    confidence_scores[q_id] = conf
                    marked_regions[q_id] = {
                        "option": marked_option,
                        "fill_ratio": conf
                    }
            
            # Calculate overall confidence
            overall_conf = (
                np.mean(list(confidence_scores.values()))
                if confidence_scores else 0.0
            )
            
            return MCQExtractionResult(
                answers=answers,
                confidence=confidence_scores,
                marked_regions=marked_regions,
                overall_confidence=overall_conf,
                is_valid=len(answers) > 0,
                debug_info={
                    "total_boxes_detected": len(detections),
                    "boxes_with_text": len(boxes_with_text),
                    "questions_found": len(questions)
                }
            )
        
        except Exception as e:
            return MCQExtractionResult(
                answers={},
                confidence={},
                is_valid=False,
                errors=[f"Extraction failed: {str(e)}"]
            )
    
    def _detect_boxes(self) -> list[BlockDetection]:
        """
        Step 1: Use YOLO to detect answer boxes.
        
        Returns:
            List of detections with coordinates
        """
        detections = self.detector.detect(self.image, conf=0.3)
        detections.sort(key=lambda det: (det.y1, det.x1))
        return detections
    
    def _extract_text_from_boxes(self, detections: list[BlockDetection]) -> list[BlockDetection]:
        """
        Step 2: Extract text from each box using TrOCR (or fallback to Tesseract).
        
        Returns:
            List of detections with extracted text
        """
        for det in detections:
            crop = crop_detection(self.image, det)
            
            if crop.size == 0:
                det.text = ""
                det.text_confidence = 0.0
                continue
            
            # Extract text with TrOCR
            text = self._ocr_with_trocr(crop)
            
            det.text = text.upper() if text else ""
            det.text_confidence = 0.9 if text else 0.0
        
        return detections
    
    def _ocr_with_trocr(self, image_crop: np.ndarray) -> str:
        """Extract text from image crop using TrOCR."""
        try:
            # Convert to PIL
            pil_image = Image.fromarray(cv2.cvtColor(image_crop, cv2.COLOR_BGR2RGB))
            
            # Preprocess
            pixel_values = self.processor(pil_image, return_tensors="pt").pixel_values
            
            # Generate text
            generated_ids = self.trocr_model.generate(pixel_values)
            text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            return text.strip()
        except Exception:
            return ""
    
    def _group_boxes_by_question(
        self,
        boxes: list[BlockDetection],
        answer_options: list[str]
    ) -> dict[str, dict]:
        """
        Step 3: Group boxes into questions based on position.
        
        Returns:
            {"q1": {"A": {...}, "B": {...}, ...}, "q2": {...}, ...}
        """
        if not boxes:
            return {}
        
        questions = {}
        question_num = 1
        current_group = []
        last_y = boxes[0].y1
        
        for box in boxes:
            # If far vertically, start new group
            if abs(box.y1 - last_y) > 50:
                if current_group:
                    q_id = f"q{question_num}"
                    questions[q_id] = self._map_group_to_options(
                        current_group, answer_options
                    )
                    question_num += 1
                current_group = []

            current_group.append(box)
            last_y = box.y1
        
        # Don't forget last group
        if current_group:
            q_id = f"q{question_num}"
            questions[q_id] = self._map_group_to_options(current_group, answer_options)
        
        return questions
    
    def _map_group_to_options(
        self,
        group: list[BlockDetection],
        answer_options: list[str]
    ) -> dict[str, dict]:
        """
        Map a group of boxes to answer options (A, B, C, D).
        
        Tries to match text extracted by TrOCR to option letters.
        Falls back to positional ordering if text extraction fails.
        """
        # Try to match by extracted text
        result = {}
        used_boxes = set()
        
        for option in answer_options:
            # Find box with matching text
            best_match = None
            best_score = 0.0
            
            for i, box in enumerate(group):
                if i in used_boxes:
                    continue
                
                # Check if text matches option letter
                if option in getattr(box, "text", ""):
                    best_match = (i, box, 1.0)
                    break
                elif getattr(box, "text_confidence", 0.0) > best_score:
                    best_score = getattr(box, "text_confidence", 0.0)
                    best_match = (i, box, best_score)
            
            if best_match:
                idx, box, score = best_match
                result[option] = {
                    "box": box,
                    "text": getattr(box, "text", ""),
                    "text_confidence": getattr(box, "text_confidence", 0.0),
                }
                used_boxes.add(idx)
        
        # If we didn't get all options by text matching, use positional ordering
        sorted_group = sorted(
            [(i, box) for i, box in enumerate(group) if i not in used_boxes],
            key=lambda x: x[1].x1
        )
        
        for option, (idx, box) in zip(
            [opt for opt in answer_options if opt not in result],
            sorted_group
        ):
            result[option] = {
                "box": box,
                "text": getattr(box, "text", ""),
                "text_confidence": getattr(box, "text_confidence", 0.0),
            }
        
        return result
    
    def _find_marked_option(
        self,
        options: dict[str, dict],
        fill_threshold: float = 0.3
    ) -> tuple[str | None, float]:
        """
        Step 4: Determine which option box is marked/filled.
        
        Uses image processing to detect fill ratio (darkness).
        
        Returns:
            (option_letter, confidence) e.g., ("A", 0.98)
        """
        marked_option = None
        highest_fill_ratio = 0.0
        
        for option_letter, box_info in options.items():
            crop = crop_detection(self.image, box_info["box"])
            
            if crop.size == 0:
                continue
            
            # Calculate fill ratio (darkness)
            fill_ratio = self._calculate_fill_ratio(crop, fill_threshold)
            
            if fill_ratio > highest_fill_ratio:
                highest_fill_ratio = fill_ratio
                marked_option = option_letter
        
        # Confidence based on how much darker the marked option is than average
        confidence = min(highest_fill_ratio * 1.2, 1.0) if marked_option else 0.0
        
        return marked_option, confidence
    
    def _calculate_fill_ratio(self, image_crop: np.ndarray, threshold: float = 0.3) -> float:
        """
        Calculate how filled/dark the box is.
        
        Returns:
            Fill ratio (0.0 = empty, 1.0 = completely filled)
        """
        # Convert to grayscale
        if len(image_crop.shape) == 3:
            gray = cv2.cvtColor(image_crop, cv2.COLOR_BGR2GRAY)
        else:
            gray = image_crop
        
        # Normalize to 0-1
        gray_normalized = gray.astype(float) / 255.0
        
        # Calculate darkness (lower values = darker)
        # If pixel is dark (< threshold), count as filled
        darkness = 1.0 - gray_normalized
        fill_ratio = np.mean(darkness)
        
        # Only consider significant darkness
        if fill_ratio > threshold:
            return fill_ratio
        else:
            return 0.0
