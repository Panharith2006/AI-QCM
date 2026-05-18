"""
Gemini-based OCR extractor for detecting and extracting text from boxes.

Uses:
1. YOLO for box detection
2. Gemini Vision API for text extraction from each box
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
import warnings

import cv2
import numpy as np
import torch
from PIL import Image

from src.detection.yolo_layout import BlockDetection, YoloLayoutDetector, crop_detection
from src.alignment.perspective import perspective_correct
from src.OCR.ocr_preprocess import ensure_bgr, preprocess_for_ocr
from src.OCR.text_normalizer import extract_answer_from_text, post_process_text

try:
    from src.llm.gemini_processor import GeminiProcessor
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

warnings.filterwarnings('ignore', category=UserWarning)
logger = logging.getLogger(__name__)


@dataclass
class GeminiExtractionResult:
    """Result from Gemini-based text extraction."""
    answers: dict[str, dict] = field(default_factory=dict)
    confidence: dict[str, float] = field(default_factory=dict)
    marked_regions: dict[str, dict] = field(default_factory=dict)
    overall_confidence: float = 0.0
    is_valid: bool = True
    errors: list[str] = field(default_factory=list)
    debug_info: dict = field(default_factory=dict)


class GeminiOCRExtractor:
    
    def __init__(
        self,
        yolo_model_path: str,
        gemini_api_key: str = None,
        debug: bool = False,
    ):
        
        self.debug = debug
        
        # Initialize YOLO for box detection
        self.detector = YoloLayoutDetector(yolo_model_path)
        if self.detector.model is None:
            raise RuntimeError(f"YOLO model not found: {yolo_model_path}")
        
        if self.debug:
            logger.info("✓ YOLO detector initialized")
        
        # Initialize Gemini for text extraction
        if not GEMINI_AVAILABLE:
            raise ImportError("Gemini processor not found. Install: pip install google-generativeai")
        
        try:
            self.gemini = GeminiProcessor(
                api_key=gemini_api_key,
                model_name="models/gemini-2.5-flash",
                debug=debug,
            )
            if self.debug:
                logger.info("✓ Gemini Vision API initialized")
        except ValueError as e:
            raise RuntimeError(f"Gemini setup failed: {e}. Set GOOGLE_API_KEY environment variable.")
        
        self.grid_detector = GridDetector(debug=debug)
    
    def extract(
        self,
        image_path: str,
        config: dict = None,
    ) -> GeminiExtractionResult:
        
        result = GeminiExtractionResult()
        config = config or {}
        
        try:
            # Read image
            image = cv2.imread(image_path)
            if image is None:
                result.errors.append(f"Failed to read image: {image_path}")
                result.is_valid = False
                return result
            
            image = ensure_bgr(image)
            
            if self.debug:
                logger.info(f"Processing image: {image_path}")
            
            # Detect boxes with YOLO
            detections = self.detector.detect(image)
            
            if not detections:
                result.errors.append("No boxes detected by YOLO")
                result.is_valid = False
                return result
            
            if self.debug:
                logger.info(f"Detected {len(detections)} boxes")
            
            # Extract text from each detected box
            answer_options = config.get("answer_options", [])
            
            for idx, detection in enumerate(detections):
                question_id = f"Q{idx + 1}"
                
                # Crop the detected region
                cropped_image = crop_detection(image, detection)
                
                if cropped_image is None or cropped_image.size == 0:
                    result.answers[question_id] = {
                        "answer": None,
                        "confidence": 0.0,
                        "type": "empty",
                    }
                    continue
                
                # Use Gemini to extract text from this box
                extracted = self._extract_from_box(
                    cropped_image,
                    question_id,
                    answer_options,
                )
                
                result.answers[question_id] = extracted
                result.confidence[question_id] = extracted.get("confidence", 0.0)
                
                if self.debug:
                    logger.info(
                        f"{question_id}: '{extracted.get('answer', '')}' "
                        f"(confidence: {extracted.get('confidence', 0.0):.2f})"
                    )
            
            # Calculate overall confidence
            confidences = [c for c in result.confidence.values() if c > 0]
            if confidences:
                result.overall_confidence = sum(confidences) / len(confidences)
            
            result.is_valid = True
            result.debug_info["total_boxes"] = len(detections)
            result.debug_info["successful_extractions"] = len(result.answers)
            
            return result
            
        except Exception as e:
            result.errors.append(f"Extraction failed: {str(e)}")
            result.is_valid = False
            logger.error(f"Extraction error: {e}")
            return result
    
    def _extract_from_box(
        self,
        box_image: np.ndarray,
        question_id: str,
        answer_options: list = None,
    ) -> dict:
       
        try:
            # Convert to PIL Image for Gemini
            if isinstance(box_image, np.ndarray):
                if len(box_image.shape) == 3 and box_image.shape[2] == 3:
                    # BGR to RGB
                    box_image = cv2.cvtColor(box_image, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(box_image)
            else:
                pil_image = box_image
            
            # Build prompt for Gemini
            prompt = "Extract all text visible in this image. Return only the text, nothing else."
            
            if answer_options:
                prompt = f"""Extract the answer from this box. 
The answer should be one of: {', '.join(answer_options)}
Return ONLY the letter/option, nothing else."""
            
            # Use Gemini to extract text
            gemini_result = self.gemini.extract_text_from_image(
                pil_image,
                context=f"Question {question_id}",
            )
            
            text = gemini_result.get("text", "").strip()
            confidence = gemini_result.get("confidence", 0.0)
            
            # If we have answer options, try to map to one
            if answer_options and text:
                mapped = self.gemini.map_to_options(
                    text,
                    answer_options,
                    question_context=question_id,
                )
                text = mapped.get("matched_option") or text
                confidence = max(confidence, mapped.get("confidence", 0.0))
            
            # Normalize the extracted text
            if text:
                text = text.upper().strip()
            
            return {
                "answer": text if text else None,
                "confidence": confidence,
                "type": "gemini_ocr",
                "raw": gemini_result.get("text", ""),
            }
            
        except Exception as e:
            logger.error(f"Box extraction failed for {question_id}: {e}")
            return {
                "answer": None,
                "confidence": 0.0,
                "type": "error",
                "error": str(e),
            }
    
    def enable_debug(self):
        """Enable debug logging."""
        self.debug = True
        if hasattr(self.gemini, 'debug'):
            self.gemini.debug = True
