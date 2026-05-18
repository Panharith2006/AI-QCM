from __future__ import annotations

from dataclasses import dataclass, field
import logging
import re

import cv2
import numpy as np

from src.alignment.perspective import perspective_correct
from src.detection.yolo_layout import YoloLayoutDetector, crop_detection
from src.pipeline import PipelineResult

try:
    from src.llm.gemini_processor import GeminiProcessor
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

logger = logging.getLogger(__name__)


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
        
        # Initialize Gemini for text extraction
        if not GEMINI_AVAILABLE:
            raise ImportError("Gemini processor not found. Install: pip install google-generativeai")
        
        try:
            self.gemini = GeminiProcessor(debug=False)
        except ValueError as e:
            raise RuntimeError(f"Gemini setup failed: {e}. Set GOOGLE_API_KEY environment variable")
        
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
                pass

        result = PipelineResult(
            is_valid=True,
            debug_info={
                "mode":               "circle",
                "extraction_method":  "gemini_whole_image",
                "total_boxes_detected": 0,
            },
        )

        # ── Send whole image to Gemini ─────────────────────────────────────
        prompt = (
            "This is a multiple-choice exam answer sheet. "
            "Students have circled one answer option for each question. "
            "For EVERY question on this sheet, identify which option (A, B, C, D, or E) is circled. "
            "Return ONLY a comma-separated list in order: Q1=B, Q2=C, Q3=A, ... "
            "Use this exact format. If a question has no clear answer write Q?=-. "
            "Do NOT include any explanation, just the list."
        )

        try:
            gemini_result = self.gemini.extract_text_from_image(image, prompt_override=prompt)
            raw = gemini_result.get("text", "").strip()
            conf = gemini_result.get("confidence", 0.85)

            if self.debug:
                print(f"Gemini whole-image raw: '{raw}'")

            # Parse "Q1=B, Q2=C, Q3=A, ..."
            import re
            pairs = re.findall(r"Q(\d+)\s*=\s*([A-E\-])", raw, re.IGNORECASE)

            if not pairs:
                result.is_valid = False
                result.errors.append(f"Gemini returned unparseable output: '{raw}'")
                result.debug_info["raw_gemini"] = raw
                return result

            for qnum, letter in pairs:
                qid    = f"Q{qnum}"
                answer = letter.upper() if letter != "-" else None
                result.answers[qid] = {
                    "answer":     answer,
                    "confidence": conf if answer else 0.0,
                    "type":       "circle",
                    "num_cells":  5,
                }
                result.extracted_answers.append((qid, answer, conf, "circle"))

            found = sum(1 for a in result.answers.values() if a["answer"])
            result.debug_info.update({
                "total_boxes_detected": len(pairs),
                "questions_found":      found,
                "raw_gemini":           raw,
            })

        except Exception as e:
            result.is_valid = False
            result.errors.append(f"Gemini extraction failed: {e}")
            logger.error(f"Circle-fill Gemini error: {e}")

        return result


    def _analyze_crop(self, crop: np.ndarray, qid: str) -> BubbleAnalysis:
        if crop.size == 0:
            return BubbleAnalysis(qid, None, 0.0, num_choices=0)

        prompt = (
            "This image shows one question from a multiple-choice exam sheet. "
            "The student has circled or marked one of the answer options (A, B, C, D, or E). "
            "Look carefully at which option has a circle or mark drawn around it. "
            "Return ONLY the single letter of the selected option (A, B, C, D, or E). "
            "If no option is clearly marked, return a dash (-). "
            "Return nothing else — just one character."
        )

        try:
            result = self.gemini.extract_text_from_image(crop, prompt_override=prompt)
            text = result.get("text", "").strip().upper()
            conf = result.get("confidence", 0.0)

            if self.debug:
                print(f"  {qid}: Gemini → '{text}' (conf={conf:.2f})")

            # Accept only A-E
            answer = text[0] if text and text[0] in "ABCDE" else None
            score_map = {answer: 1.0} if answer else {}

        except Exception as e:
            if self.debug:
                logger.error(f"Gemini extraction failed for {qid}: {e}")
            answer, conf, score_map = None, 0.0, {}

        self.choice_visualizations[qid] = crop.copy()

        return BubbleAnalysis(
            question_id=qid,
            answer=answer,
            confidence=float(conf) if answer else 0.0,
            score_map=score_map,
            num_choices=5,
            is_ambiguous=False,
        )


    def _preprocess_cell(self, img: np.ndarray) -> np.ndarray:
        """Preprocess cell image for better Gemini extraction."""
        if img is None or img.size == 0:
            return img
        
        try:
            # 1. Convert to grayscale
            if len(img.shape) == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                gray = img
            
            # 2. Upscale if too small
            h, w = gray.shape
            if h < 64 or w < 64:
                scale = max(2, 64 // min(h, w))
                gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            
            # 3. Enhance contrast using CLAHE
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            gray = clahe.apply(gray)
            
            # 4. Denoise
            gray = cv2.bilateralFilter(gray, 5, 75, 75)
            
            # 5. Adaptive thresholding
            binary = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            )
            
            # 6. Convert back to BGR
            preprocessed = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
            return preprocessed
            
        except Exception as e:
            if self.debug:
                logger.warning(f"Preprocessing failed: {e}")
            return img

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