from __future__ import annotations

from dataclasses import dataclass, field
import json
import logging
import warnings

import cv2
import numpy as np
from PIL import Image

from src.detection.yolo_layout import YoloLayoutDetector, crop_detection

try:
    from src.llm.gemini_processor import GeminiProcessor
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

try:
    from src.box_interpretation.content_interpreter import BoxContentInterpreter
    BOX_INTERPRETATION_AVAILABLE = True
except ImportError:
    BOX_INTERPRETATION_AVAILABLE = False

warnings.filterwarnings('ignore', category=UserWarning)
logger = logging.getLogger(__name__)


@dataclass
class GeminiExtractionResult:
    answers: dict[str, dict] = field(default_factory=dict)
    confidence: dict[str, float] = field(default_factory=dict)
    overall_confidence: float = 0.0
    is_valid: bool = True
    errors: list[str] = field(default_factory=list)
    debug_info: dict = field(default_factory=dict)
    box_interpretations: dict[str, dict] = field(default_factory=dict)  # NEW: Visual interpretation results


class GeminiOCRExtractor:
    CIRCLE_COMPLETION_PROMPT = """
Look at this quiz answer sheet crop carefully.

For the question shown:
1. Read the question number and question text if visible.
2. Read all answer options (A, B, C, D).
3. Identify which option is circled or clearly selected.
4. Return ONLY valid JSON in this exact format:
{
  "question_number": "1",
  "question_text": "...",
  "selected_answer": "B",
  "confidence": 0.94
}

Rules:
- Do not guess.
- If you cannot determine a selected answer, set "selected_answer" to an empty string.
- Keep confidence between 0.0 and 1.0.
- Do not add markdown, code fences, or extra text.
"""
    
    def __init__(
        self,
        yolo_model_path: str,
        gemini_api_key: str = None,
        debug: bool = False,
        prompt_style: str = "table",
    ):
        
        self.debug = debug
        self.prompt_style = prompt_style
        
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
        
        # Initialize Box Content Interpreter (optional for visual interpretation)
        self.box_interpreter = None
        if BOX_INTERPRETATION_AVAILABLE:
            try:
                self.box_interpreter = BoxContentInterpreter(
                    gemini_api_key=gemini_api_key,
                    debug=debug,
                )
                if self.debug:
                    logger.info("Box Content Interpreter initialized")
            except Exception as e:
                logger.warning(f"Box interpretation disabled: {e}")
    
    def extract(
        self,
        image_path: str,
        config: dict = None,
    ) -> GeminiExtractionResult:
        
        import time
        result = GeminiExtractionResult()
        config = config or {}
        
        try:
            # Read image
            image = cv2.imread(image_path)
            if image is None:
                result.errors.append(f"Failed to read image: {image_path}")
                result.is_valid = False
                return result
            
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
                
                # Add delay between requests to avoid rate limiting
                if idx > 0:
                    delay = 0.5  # 500ms between requests
                    time.sleep(delay)
                
                # Crop the detected region
                cropped_image = crop_detection(image, detection)
                
                if cropped_image is None or cropped_image.size == 0:
                    result.answers[question_id] = {
                        "answer": None,
                        "confidence": 0.0,
                        "type": "empty",
                    }
                    if self.debug:
                        logger.info(f"{question_id}: SKIPPED (empty crop)")
                    continue
                
                # Use Gemini to extract text from this box
                try:
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
                    
                    # NEW: Interpret the visual content of the box (if interpreter available)
                    if self.box_interpreter:
                        try:
                            interpretation = self.box_interpreter.interpret_box_content(
                                box_image=cropped_image,
                                box_id=question_id,
                                question_context=f"Multiple choice answer box for {question_id}",
                            )
                            
                            # Store interpretation results
                            result.box_interpretations[question_id] = {
                                "content_type": interpretation.box_content_type,
                                "explanation": interpretation.content_explanation,
                                "confidence": interpretation.confidence_interpretation,
                                "visual_clarity": interpretation.visual_clarity,
                                "visual_clarity_score": interpretation.visual_clarity_score,
                                "recommendations": interpretation.recommendations,
                            }
                            
                            if self.debug:
                                logger.info(
                                    f"{question_id} interpretation: {interpretation.box_content_type} "
                                    f"(clarity: {interpretation.visual_clarity})"
                                )
                        except Exception as interp_error:
                            logger.warning(f"Box interpretation failed for {question_id}: {interp_error}")
                            # Non-critical, continue without interpretation
                
                except Exception as box_error:
                    logger.error(f"Error processing {question_id}: {box_error}", exc_info=True)
                    result.answers[question_id] = {
                        "answer": None,
                        "confidence": 0.0,
                        "type": "error",
                        "error": str(box_error),
                    }
                    result.errors.append(f"{question_id} extraction failed: {box_error}")
                    # Continue processing other boxes instead of failing
                    if self.debug:
                        logger.warning(f"Continuing with next box despite error in {question_id}")
            
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
            
            # Use Gemini to extract text/answers from this box
            # extract_text_from_image now returns ALL text in the box (e.g., "A B D C")
            prompt_override = self.CIRCLE_COMPLETION_PROMPT if self.prompt_style == "circle" else None
            gemini_result = self.gemini.extract_text_from_image(
                pil_image,
                context=f"Question {question_id}",
                prompt_override=prompt_override,
            )
            
            text = gemini_result.get("text", "").strip()
            confidence = gemini_result.get("confidence", 0.0)
            answers_list: list[str] = []

            if self.prompt_style == "circle":
                parsed = self._extract_json_from_response(text)
                if parsed:
                    try:
                        payload = json.loads(parsed)
                        selected_answer = str(payload.get("selected_answer", "")).strip().upper()
                        if selected_answer:
                            text = selected_answer
                        else:
                            text = ""

                        confidence = float(payload.get("confidence", confidence))
                    except Exception:
                        pass
            
            # Check if we got multiple answers (space-separated or concatenated)
            # If there are spaces or multiple characters, parse them
            if text:
                if ' ' in text:
                    # Space-separated: "A B D C"
                    answers_list = [a.strip().upper() for a in text.split() if a.strip()]
                elif len(text) > 1:
                    # Concatenated: "ABDC" - treat each char as an answer
                    answers_list = [c.upper() for c in text if c.isalnum()]
                else:
                    # Single answer
                    answers_list = [text.upper()]
            
            # If answer_options provided, validate against them
            if answer_options and answers_list:
                validated = []
                for ans in answers_list:
                    for option in answer_options:
                        if ans.upper() == option.upper():
                            validated.append(option.upper())
                            break
                    else:
                        # If not a valid option, keep it anyway but lower confidence
                        validated.append(ans)
                        confidence = max(0.5, confidence - 0.1)
                answers_list = validated
            
            # Return results
            if len(answers_list) > 1:
                # Multiple answers in this box (e.g., a row)
                result_text = " ".join(answers_list)
            elif answers_list:
                result_text = answers_list[0]
            else:
                result_text = None
            
            if self.debug:
                logger.info(f"{question_id}: extracted {len(answers_list)} answer(s): {answers_list}")
            
            return {
                "answer": result_text,
                "answers": answers_list,  # Also return as list for flexibility
                "answer_count": len(answers_list),
                "confidence": confidence,
                "type": "gemini_ocr",
                "raw": gemini_result.get("text", ""),
            }
            
        except Exception as e:
            logger.error(f"Box extraction failed for {question_id}: {e}", exc_info=True)
            return {
                "answer": None,
                "answers": [],
                "answer_count": 0,
                "confidence": 0.0,
                "type": "error",
                "error": str(e),
            }

    
    def enable_debug(self):
        self.debug = True
        if hasattr(self.gemini, 'debug'):
            self.gemini.debug = True

    def _extract_json_from_response(self, response_text: str) -> str | None:
        cleaned = response_text.strip()

        if cleaned.startswith("```"):
            cleaned = cleaned.removeprefix("```json").removeprefix("```").strip()
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3].strip()

        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            return cleaned[start : end + 1]

        return None
