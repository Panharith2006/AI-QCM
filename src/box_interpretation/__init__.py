"""
Box Content Interpretation Module

Interprets and explains the content inside detected answer boxes using Gemini Vision API.
Bridges the gap between YOLO box detection and text extraction.
"""

from .content_interpreter import BoxContentInterpreter, InterpretationResult

__all__ = ["BoxContentInterpreter", "InterpretationResult"]
