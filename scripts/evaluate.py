from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.scoring.compare import compare_answer_maps


def main() -> None:
    teacher = {"Q1": "A", "Q2": "C", "Q3": "B"}
    student = {"Q1": "A", "Q2": "D", "Q3": "UNANSWERED"}

    result = compare_answer_maps(teacher, student)
    print("Evaluation sample")
    print(f"Total: {result.total_questions}")
    print(f"Correct: {result.correct}")
    print(f"Wrong: {result.wrong}")
    print(f"Unanswered: {result.unanswered}")
    print(f"Percentage: {result.percentage:.2f}")


if __name__ == "__main__":
    main()
