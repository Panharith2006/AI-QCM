from __future__ import annotations

import logging
import base64
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import io

from src.llm.gemini_processor import GeminiProcessor

logger = logging.getLogger(__name__)


@dataclass
class InterpretationResult:
    """Result of box content interpretation"""
    
    box_id: str
    box_content_type: str  # "filled", "partial", "empty", "unclear", "multiple"
    content_explanation: str  # Detailed explanation of what's in the box
    confidence_interpretation: float  # Confidence in the interpretation (0.0-1.0)
    visual_clarity: str  # "clear", "moderate", "poor"
    visual_clarity_score: float  # 0.0-1.0
    recommendations: list[str] = field(default_factory=list)  # Suggestions for verification
    raw_interpretation: str = ""  # Raw Gemini response
    is_valid: bool = True
    errors: list[str] = field(default_factory=list)


class BoxContentInterpreter:
    
    def __init__(
        self,
        gemini_api_key: Optional[str] = None,
        model_name: str = "models/gemini-2.5-flash-lite",
        debug: bool = False,
    ):
        
        self.debug = debug
        self.gemini = GeminiProcessor(
            api_key=gemini_api_key,
            model_name=model_name,
            debug=debug,
        )
        
        if debug:
            logger.info("✓ BoxContentInterpreter initialized with Gemini Vision API")
    
    def encode_image_to_base64(self, image_path_or_array) -> str:
        
        if isinstance(image_path_or_array, str) or isinstance(image_path_or_array, Path):
            # File path
            with open(image_path_or_array, "rb") as f:
                return base64.standard_b64encode(f.read()).decode("utf-8")
        elif isinstance(image_path_or_array, np.ndarray):
            # Numpy array
            _, buffer = cv2.imencode('.png', image_path_or_array)
            return base64.standard_b64encode(buffer).decode("utf-8")
        elif isinstance(image_path_or_array, Image.Image):
            # PIL Image
            buffer = io.BytesIO()
            image_path_or_array.save(buffer, format="PNG")
            return base64.standard_b64encode(buffer.getvalue()).decode("utf-8")
        else:
            raise ValueError(f"Unsupported image type: {type(image_path_or_array)}")
    
    def interpret_box_content(
        self,
        box_image,
        box_id: str,
        question_context: Optional[str] = None,
        expected_answer_type: Optional[str] = None,
    ) -> InterpretationResult:
       
        try:
            # Encode image to base64
            image_b64 = self.encode_image_to_base64(box_image)
            
            # Build interpretation prompt
            prompt = self._build_interpretation_prompt(
                question_context,
                expected_answer_type,
            )
            
            # Call Gemini Vision API
            if self.debug:
                logger.info(f"Analyzing box content for {box_id}...")
            
            # Create message with image for Gemini
            interpretation = self._call_gemini_vision(prompt, image_b64)
            
            # Parse the interpretation
            result = self._parse_interpretation(interpretation, box_id)
            
            if self.debug:
                logger.info(
                    f"Box {box_id}: {result.box_content_type} "
                    f"(clarity: {result.visual_clarity}, confidence: {result.confidence_interpretation:.2f})"
                )
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to interpret box {box_id}: {e}", exc_info=True)
            return InterpretationResult(
                box_id=box_id,
                box_content_type="error",
                content_explanation=f"Failed to interpret: {str(e)}",
                confidence_interpretation=0.0,
                visual_clarity="unknown",
                visual_clarity_score=0.0,
                is_valid=False,
                errors=[str(e)],
            )
    
    def interpret_multiple_boxes(
        self,
        boxes_data: list[dict],
        question_contexts: Optional[dict] = None,
    ) -> dict[str, InterpretationResult]:
       
        results = {}
        
        for box_data in boxes_data:
            box_id = box_data.get("box_id", "unknown")
            box_image = box_data.get("image")
            context = None
            
            if question_contexts:
                context = question_contexts.get(box_id)
            
            result = self.interpret_box_content(
                box_image=box_image,
                box_id=box_id,
                question_context=context,
            )
            
            results[box_id] = result
        
        return results
    
    def _build_interpretation_prompt(
        self,
        question_context: Optional[str] = None,
        expected_answer_type: Optional[str] = None,
    ) -> str:
        
        prompt = """Analyze this answer box image and provide a detailed interpretation:

1. Content Type: Identify what's in the box:
   - "filled" = contains a clear, filled answer
   - "partial" = contains a partially visible/marked answer
   - "empty" = no content or marks
   - "unclear" = ambiguous, hard to determine what's marked
   - "multiple" = multiple marks/answers visible

2. Visual Clarity: Rate the clarity:
   - "clear" = easy to read/determine what's marked
   - "moderate" = somewhat readable but has noise
   - "poor" = very hard to determine

3. Explanation: Describe exactly what you see in the box (content, marks, handwriting, clarity issues)

4. Confidence: Rate your confidence in this interpretation (0-100%)

5. Recommendations: List any concerns (e.g., "multiple answers", "requires verification", "very faint")
"""
        
        if question_context:
            prompt += f"\nQuestion Context: {question_context}\n"
        
        if expected_answer_type:
            prompt += f"Expected Answer Type: {expected_answer_type}\n"
        
        prompt += """
Provide your analysis in this exact format:
CONTENT_TYPE: [filled|partial|empty|unclear|multiple]
VISUAL_CLARITY: [clear|moderate|poor]
CONFIDENCE: [0-100]
EXPLANATION: [detailed description of what you see]
RECOMMENDATIONS: [comma-separated list, or "none"]
"""
        
        return prompt
    
    def _call_gemini_vision(self, prompt: str, image_b64: str) -> str:
        """Call Gemini Vision API with image and prompt."""
        
        try:
            import google.genai as genai
            
            response = self.gemini.client.models.generate_content(
                model=self.gemini.model_name,
                contents=[
                    prompt,
                    {
                        "inline_data": {
                            "mime_type": "image/png",
                            "data": image_b64,
                        }
                    }
                ],
            )
            
            return response.text.strip()
            
        except Exception as e:
            logger.error(f"Gemini Vision API call failed: {e}", exc_info=True)
            raise
    
    def _parse_interpretation(self, response: str, box_id: str) -> InterpretationResult:
        
        lines = response.split('\n')
        parsed = {}
        
        for line in lines:
            if ':' in line:
                key, value = line.split(':', 1)
                parsed[key.strip()] = value.strip()
        
        # Extract fields with defaults
        content_type = parsed.get("CONTENT_TYPE", "unclear").lower()
        visual_clarity = parsed.get("VISUAL_CLARITY", "moderate").lower()
        confidence_str = parsed.get("CONFIDENCE", "50")
        explanation = parsed.get("EXPLANATION", response)
        recommendations_str = parsed.get("RECOMMENDATIONS", "none")
        
        # Parse confidence
        try:
            confidence = float(confidence_str.replace('%', '').strip()) / 100.0
        except:
            confidence = 0.5
        
        # Parse visual clarity score
        clarity_scores = {"clear": 0.9, "moderate": 0.6, "poor": 0.3}
        clarity_score = clarity_scores.get(visual_clarity, 0.5)
        
        # Parse recommendations
        if recommendations_str.lower() != "none":
            recommendations = [r.strip() for r in recommendations_str.split(',')]
        else:
            recommendations = []
        
        # Validate content type
        valid_types = ["filled", "partial", "empty", "unclear", "multiple"]
        if content_type not in valid_types:
            content_type = "unclear"
        
        return InterpretationResult(
            box_id=box_id,
            box_content_type=content_type,
            content_explanation=explanation,
            confidence_interpretation=confidence,
            visual_clarity=visual_clarity,
            visual_clarity_score=clarity_score,
            recommendations=recommendations,
            raw_interpretation=response,
            is_valid=True,
        )
