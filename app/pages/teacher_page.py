from __future__ import annotations

import re

import streamlit as st

from config import MIN_ROWS, MAX_ROWS, MIN_COLS, MAX_COLS, DEFAULT_ROWS, DEFAULT_COLS
from reference_manager import load_reference, save_reference


def _ordered_answers(answers: dict[str, str]) -> list[tuple[str, str]]:
    def sort_key(item: tuple[str, str]) -> tuple[int, int | str, str]:
        key = str(item[0]).strip()
        match = re.search(r"\d+", key)
        if match:
            return (0, int(match.group()), key)
        return (1, key, key)

    ordered = []
    for key, value in sorted(answers.items(), key=sort_key):
        normalized = str(value).strip().upper() if value is not None else ""
        ordered.append((str(key), normalized or "—"))
    return ordered


def _answers_to_grid(answers: dict[str, str], rows: int, cols: int) -> list[list[str]]:
    ordered_answers = [answer for _, answer in _ordered_answers(answers)]
    total_slots = rows * cols
    padded_answers = ordered_answers[:total_slots] + ["—"] * max(0, total_slots - len(ordered_answers))
    return [padded_answers[index:index + cols] for index in range(0, total_slots, cols)]


def _render_reference_section(
    title: str,
    save_button_label: str,
    save_button_key: str,
    note: str,
    reference_type: str,
) -> None:
    st.subheader(title)
    st.markdown(note)

    col1, col2 = st.columns(2)
    with col1:
        num_rows = st.number_input(
            "Number of rows",
            min_value=MIN_ROWS,
            max_value=MAX_ROWS,
            value=DEFAULT_ROWS,
            key=f"{save_button_key}_rows",
        )
    with col2:
        num_cols = st.number_input(
            "Number of columns (answers per row)",
            min_value=MIN_COLS,
            max_value=MAX_COLS,
            value=DEFAULT_COLS,
            key=f"{save_button_key}_cols",
        )

    st.write(f"**Grid: {num_rows} row(s) × {num_cols} column(s) = {num_rows * num_cols} answers total**")

    reference_grid: list[list[str]] = []
    for row in range(num_rows):
        st.write(f"**Row {row + 1}:**")
        cols = st.columns(num_cols)
        row_answers = []

        for col in range(num_cols):
            with cols[col]:
                answer = st.text_input(
                    label=f"Q{row * num_cols + col + 1}",
                    max_chars=1,
                    value="",
                    key=f"{save_button_key}_answer_{row}_{col}",
                )
                row_answers.append(answer.upper() if answer else "—")

        reference_grid.append(row_answers)

    st.markdown("---")

    all_reference = [answer for row in reference_grid for answer in row]
    st.write("**Answer Key Summary:**")
    st.code(" | ".join(all_reference))

    if st.button(save_button_label, type="primary", key=save_button_key):
        save_reference(reference_grid, reference_type=reference_type)
        st.success(f"Reference saved! {len(all_reference)} answers stored.")
        st.write(f"**Answers:** {' | '.join(all_reference)}")

    saved_ref = load_reference(reference_type)
    if saved_ref:
        st.info("Currently Saved Reference:")
        flat_saved = [answer for row in saved_ref for answer in row]
        st.code(" | ".join(flat_saved))


def render_teacher_page():
    st.header("Teacher: Create Answer Reference")
    tab_bubble, tab_circle, tab_table = st.tabs(["Bubble Reference", "Circle Completion Reference", "Table Structure Reference"])

    with tab_bubble:
        _render_reference_section(
            title="Bubble Reference",
            save_button_label="Save Bubble Reference",
            save_button_key="save_bubble_reference",
            note="Create the bubble reference used by the Bubble Detector tab in the main app.",
            reference_type="bubble",
        )

    with tab_circle:
        _render_reference_section(
            title="Circle Completion Reference",
            save_button_label="Save Circle Reference",
            save_button_key="save_circle_reference",
            note="Create the circle-completion reference used by the student circle detector and grading pages.",
            reference_type="circle",
        )

    with tab_table:
        _render_reference_section(
            title="Table Structure Reference",
            save_button_label="Save Table Reference",
            save_button_key="save_table_reference",
            note="Create the table-structure reference used by the student table detector.",
            reference_type="table",
        )
