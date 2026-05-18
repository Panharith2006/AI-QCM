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


def compare_answer_maps(teacher: dict[str, str], student: dict[str, dict | str]) -> ScoreResult:
    total = len(teacher)
    correct = 0
    unanswered = 0
    details: dict[str, dict[str, str | bool]] = {}

    for qid, expected in teacher.items():
        # Extract student answer from nested or simple format
        student_entry = student.get(qid, None)
        
        if student_entry is None:
            got = "UNANSWERED"
        elif isinstance(student_entry, dict):
            # New nested format
            got = student_entry.get("answer", "UNANSWERED")
        else:
            # Old simple format or string
            got = student_entry
        
        is_correct = got == expected
        if is_correct:
            correct += 1
        if got == "UNANSWERED" or got is None:
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


def compute_metrics(student: dict[str, str], teacher: dict[str, str]) -> dict:
    result = compare_answer_maps(teacher, student)
    return {
        "total_questions": result.total_questions,
        "correct": result.correct,
        "wrong": result.wrong,
        "unanswered": result.unanswered,
        "percentage": result.percentage,
        "details": result.details,
    }
