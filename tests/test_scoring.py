from src.scoring.compare import compare_answer_maps


def test_compare_answer_maps_basic() -> None:
    teacher = {"Q1": "A", "Q2": "B", "Q3": "C"}
    student = {"Q1": "A", "Q2": "D", "Q3": "UNANSWERED"}

    result = compare_answer_maps(teacher, student)

    assert result.total_questions == 3
    assert result.correct == 1
    assert result.wrong == 2
    assert result.unanswered == 1
    assert round(result.percentage, 2) == 33.33
