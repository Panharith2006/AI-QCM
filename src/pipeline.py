from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PipelineResult:
    
    # Structured result of the OMR pipeline, including extracted answers, confidence scores, and debug info.
    answers: dict[str, dict] = field(default_factory=dict)
    extracted_answers: list[tuple[str, str, float, str]] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
    debug_info: dict = field(default_factory=dict)
    is_valid: bool = True
    errors: list[str] = field(default_factory=list)
