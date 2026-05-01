"""OCR-based answer extraction (TrOCR + YOLO - No LLM)."""

from src.OCR.ocr_extractor import OCRExtractor, MCQExtractionResult

__all__ = ["OCRExtractor", "MCQExtractionResult"]
