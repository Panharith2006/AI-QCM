from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AnswerItem:
    question_id: str
    predicted: str
    confidence: float
    scope: str


def to_answer_map(items: list[AnswerItem]) -> dict[str, str]:
    return {item.question_id: item.predicted for item in items}
