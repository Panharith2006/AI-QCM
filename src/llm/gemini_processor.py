"""Gemini API integration for text processing and image understanding."""

from __future__ import annotations

import logging
import base64
from typing import Optional
from pathlib import Path

import google.genai as genai
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


class GeminiProcessor:
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "models/gemini-2.5-flash-lite",
        debug: bool = False,
    ):
        
        import os
        
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError(
                "GOOGLE_API_KEY not set. Set environment variable: "
                "$env:GOOGLE_API_KEY = 'your-api-key'"
            )
        
        self.model_name = model_name
        self.debug = debug
        
        # Initialize Gemini client with new google.genai API
        self.client = genai.Client(api_key=self.api_key)
        self._max_retries = 3  # retries on 429 rate-limit errors
        
        if debug:
            logger.info(f"Initialized Gemini client with model: {model_name}")

    def _call_gemini(self, contents, retries: int = None) -> str:
      
        import time
        max_retries = retries if retries is not None else self._max_retries
        for attempt in range(max_retries + 1):
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=contents,
                )
                return response.text.strip()
            except Exception as e:
                err_str = str(e)
                if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                    # Try to extract retry-after seconds from error message
                    wait = 20 * (attempt + 1)  # default: 20s, 40s, 60s
                    import re
                    m = re.search(r"retry in ([\d.]+)s", err_str, re.IGNORECASE)
                    if m:
                        wait = int(float(m.group(1))) + 2
                    if attempt < max_retries:
                        logger.warning(
                            f"Gemini rate limit hit (attempt {attempt+1}/{max_retries+1}). "
                            f"Waiting {wait}s before retry..."
                        )
                        time.sleep(wait)
                    else:
                        raise
                else:
                    raise
    
    def normalize_text(
        self,
        raw_text: str,
        context: Optional[str] = None,
        expected_format: Optional[str] = None,
    ) -> dict:
        
        if not raw_text:
            return {"normalized": "", "confidence": 0.0, "original": raw_text}
        
        prompt = f"""Clean and normalize this OCR-extracted text:

Text: "{raw_text}"
"""
        
        if context:
            prompt += f"Context: {context}\n"
        
        if expected_format:
            prompt += f"Expected format: {expected_format}\n"
        
        prompt += """
Rules:
1. Correct obvious OCR errors (O→0, I→1 where appropriate)
2. Remove extra spaces
3. Convert to uppercase for answers
4. Keep only essential characters

Return ONLY the normalized text, nothing else:"""
        
        try:
            normalized = self._call_gemini(prompt)
            
            if self.debug:
                logger.info(f"Normalized: '{raw_text}' → '{normalized}'")
            
            return {
                "normalized": normalized,
                "confidence": 0.85,
                "original": raw_text,
                "method": "gemini",
            }
        except Exception as e:
            logger.error(f"Gemini normalization failed: {e}")
            # Fallback to basic normalization
            return {
                "normalized": raw_text.upper().strip(),
                "confidence": 0.5,
                "original": raw_text,
                "method": "fallback",
                "error": str(e),
            }
    
    def map_to_options(
        self,
        extracted_text: str,
        options: list[str],
        question_context: Optional[str] = None,
    ) -> dict:
        
        if not extracted_text or not options:
            return {
                "matched_option": None,
                "confidence": 0.0,
                "method": "none",
                "original": extracted_text,
            }
        
        # Try exact match first
        text_upper = extracted_text.upper().strip()
        for option in options:
            if text_upper == option.upper():
                return {
                    "matched_option": option,
                    "confidence": 1.0,
                    "method": "exact",
                    "original": extracted_text,
                }
        
        prompt = f"""Match extracted text to the correct option:

Extracted text: "{extracted_text}"
Available options: {options}
"""
        
        if question_context:
            prompt += f"Question: {question_context}\n"
        
        prompt += """
Return ONLY the matching option letter, nothing else."""
        
        try:
            matched = self._call_gemini(prompt).upper()
            
            # Validate it's in options
            for option in options:
                if matched == option.upper():
                    if self.debug:
                        logger.info(f"Matched: {extracted_text} → {option}")
                    
                    return {
                        "matched_option": option,
                        "confidence": 0.9,
                        "method": "gemini",
                        "original": extracted_text,
                    }
            
            # No valid match
            return {
                "matched_option": None,
                "confidence": 0.0,
                "method": "gemini",
                "original": extracted_text,
            }
            
        except Exception as e:
            logger.error(f"Gemini mapping failed: {e}")
            return {
                "matched_option": None,
                "confidence": 0.0,
                "method": "error",
                "original": extracted_text,
                "error": str(e),
            }
    
    def extract_text_from_image(
        self,
        image: np.ndarray | str | Image.Image,
        context: Optional[str] = None,
        prompt_override: Optional[str] = None,
    ) -> dict:
        
        try:
            # Convert to PIL Image
            if isinstance(image, str):
                image = Image.open(image)
            elif isinstance(image, np.ndarray):
                if image.dtype != np.uint8:
                    image = (image * 255).astype(np.uint8)
                # OpenCV uses BGR, convert to RGB for PIL
                if len(image.shape) == 3 and image.shape[2] == 3:
                    image = image[:, :, ::-1]  # BGR → RGB
                image = Image.fromarray(image)
            
            # Use caller-supplied prompt if provided, otherwise default
            if prompt_override:
                prompt = prompt_override
            else:
                prompt = (
                    "This is a single answer box from a handwritten exam sheet. "
                    "Look carefully and extract the single letter or number written inside. "
                    "The writing may be handwritten and slightly messy. "
                    "Return ONLY that one character (e.g. A, B, C, D, or a digit), nothing else."
                )

            if context:
                prompt = f"{context}\n{prompt}"
            
            text = self._call_gemini([prompt, image])
            
            if self.debug:
                logger.info(f"Gemini extracted from image: '{text}'")
            
            return {
                "text": text,
                "confidence": 0.85,
                "method": "gemini_vision",
            }
            
        except Exception as e:
            logger.error(f"Image extraction failed: {e}")
            return {
                "text": "",
                "confidence": 0.0,
                "method": "error",
                "error": str(e),
            }
    
    def correct_ocr_errors(self, text: str) -> dict:
        """Correct OCR errors using Gemini."""
        prompt = f"""Fix OCR errors in this text:

Text: "{text}"

Return corrected text only:"""
        
        try:
            corrected = self._call_gemini(prompt)
            
            return {
                "corrected": corrected,
                "confidence": 0.8,
                "original": text,
            }
        except Exception as e:
            logger.error(f"OCR correction failed: {e}")
            return {
                "corrected": text,
                "confidence": 0.0,
                "original": text,
                "error": str(e),
            }

    def extract_row_answers(
        self,
        image: np.ndarray | str | Image.Image,
        num_questions: Optional[int] = None,
    ) -> dict:
        
        try:
            # Convert to PIL Image
            if isinstance(image, str):
                pil = Image.open(image)
            elif isinstance(image, np.ndarray):
                if image.dtype != np.uint8:
                    image = (image * 255).astype(np.uint8)
                # BGR → RGB
                if len(image.shape) == 3 and image.shape[2] == 3:
                    image = image[:, :, ::-1]
                pil = Image.fromarray(image)
            else:
                pil = image

            hint = f" There are {num_questions} boxes." if num_questions else ""
            prompt = (
                "This image shows a row of answer boxes from a student's handwritten exam sheet.\n"
                "Each box has a number label (0, 1, 2, 3 ...) and a handwritten answer written below it.\n"
                f"Read every box from LEFT to RIGHT and extract the handwritten answer in each one.{hint}\n"
                "Rules:\n"
                "- Return ONLY the answers separated by commas, in order, nothing else.\n"
                "- If a box is empty or unreadable write a dash (-).\n"
                "- Do NOT include the box numbers, just the answers.\n"
                "Example output: MK, C, E, 4, J, M, B, M, D, G"
            )

            raw = self._call_gemini([prompt, pil])

            if self.debug:
                logger.info(f"Gemini row extraction raw: '{raw}'")

            # Parse comma-separated answers
            answers = [a.strip().upper() for a in raw.split(",") if a.strip()]

            return {
                "answers": answers,
                "raw": raw,
                "confidence": 0.85,
                "method": "gemini_row_extraction",
            }

        except Exception as e:
            logger.error(f"Row answer extraction failed: {e}")
            return {
                "answers": [],
                "raw": "",
                "confidence": 0.0,
                "method": "error",
                "error": str(e),
            }

