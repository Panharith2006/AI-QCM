from __future__ import annotations

from config import REFERENCE_FILE


def _question_sort_key(question: str) -> tuple[int, int | str, str]:
    text = str(question).strip()
    digits = ""

    for char in text:
        if char.isdigit():
            digits += char
        elif digits:
            break

    if digits:
        return (0, int(digits), text)

    return (1, text, text)


def parse_student_answers(extracted_answers_dict: dict) -> list[str]:
    parsed: list[str] = []

    if not extracted_answers_dict:
        return parsed

    is_nested_answer_map = all(isinstance(value, dict) for value in extracted_answers_dict.values())

    if is_nested_answer_map:
        items = extracted_answers_dict.items()
    else:
        items = sorted(extracted_answers_dict.items(), key=lambda item: _question_sort_key(item[0]))

    for _, value in items:
        if isinstance(value, dict):
            ans = value.get("answer") or "—"

            if value.get("answers"):
                parsed.extend(value.get("answers", []))
            elif isinstance(ans, str) and " " in ans:
                parsed.extend(ans.split())
            elif isinstance(ans, str):
                for char in ans:
                    if char.isalnum():
                        parsed.append(char)
            else:
                parsed.append(str(ans))
        else:
            ans = value or "—"

            if isinstance(ans, str) and " " in ans:
                parsed.extend(ans.split())
            elif isinstance(ans, str) and len(ans) > 1:
                for char in ans:
                    if char.isalnum():
                        parsed.append(char)
            elif str(ans).strip() and str(ans) != "—":
                parsed.append(str(ans))

    return [answer.upper().strip() for answer in parsed if answer.strip() and answer != "—"]


def flatten_reference(reference_grid: list[list[str]]) -> list[str]:
    flattened: list[str] = []
    for row in reference_grid:
        flattened.extend(row)
    return [answer.upper().strip() for answer in flattened if answer.strip() and answer != "—"]


def compare_answers(
    student_answers_dict: dict,
    reference_grid: list[list[str]]
) -> dict:
    if not reference_grid or not student_answers_dict:
        return {
            "score": 0,
            "total": 0,
            "correct": 0,
            "incorrect": 0,
            "details": [],
        }

    student_list = parse_student_answers(student_answers_dict)
    reference_list = flatten_reference(reference_grid)

    correct_count = 0
    total_count = max(len(reference_list), len(student_list))
    details = []

    for idx in range(total_count):
        ref_ans = reference_list[idx] if idx < len(reference_list) else "—"
        student_ans = student_list[idx] if idx < len(student_list) else "—"
        is_correct = student_ans.upper() == ref_ans.upper()

        if is_correct:
            correct_count += 1

        details.append({
            "question": idx + 1,
            "reference": ref_ans,
            "student": student_ans,
            "correct": is_correct,
        })

    return {
        "score": (correct_count / total_count * 100) if total_count > 0 else 0,
        "total": total_count,
        "correct": correct_count,
        "incorrect": total_count - correct_count,
        "details": details,
    }