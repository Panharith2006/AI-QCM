from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ScoreResult:
    total_questions: int
    correct: int
    wrong: int
    unanswered: int
    percentage: float
    details: dict[str, dict[str, str | bool]]


def compare_answer_maps(teacher: dict[str, str], student: dict[str, str]) -> ScoreResult:
    total = len(teacher)
    correct = 0
    unanswered = 0
    details: dict[str, dict[str, str | bool]] = {}

    for qid, expected in teacher.items():
        got = student.get(qid, "UNANSWERED")
        is_correct = got == expected
        if is_correct:
            correct += 1
        if got == "UNANSWERED":
            unanswered += 1

        details[qid] = {
            "expected": expected,
            "student": got,
            "correct": is_correct,
        }

    wrong = total - correct
    percentage = (correct / total * 100.0) if total > 0 else 0.0
    return ScoreResult(
        total_questions=total,
        correct=correct,
        wrong=wrong,
        unanswered=unanswered,
        percentage=percentage,
        details=details,
    )
