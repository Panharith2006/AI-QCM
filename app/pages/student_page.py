from pathlib import Path
import tempfile

import cv2
import numpy as np
import streamlit as st

from utils import resolve_model_path, pick_default_model_path
from reference_manager import load_reference
from answer_processing import compare_answers
from config import (
    ROOT,
    DEFAULT_MODEL_PATHS,
    MAX_DISPLAY_COLS,
    INTERPRETATION_MAX_COLS,
    SCORE_GOOD_THRESHOLD,
    SCORE_OKAY_THRESHOLD,
)
from src.alignment.perspective import perspective_correct
from src.detection.yolo_layout import YoloLayoutDetector


def _mode_title(sheet_mode: str) -> str:
    return "Circle Completion" if sheet_mode == "circle" else "Table Structure"


def _reference_type_for_mode(sheet_mode: str) -> str:
    return "circle" if sheet_mode == "circle" else "table"


def render_student_page():
    _render_sheet_page("table")


def render_circle_completion_page():
    _render_sheet_page("circle")


def _render_sheet_page(sheet_mode: str):
    st.header("Student: Upload Answer Sheet")
    st.subheader(_mode_title(sheet_mode))
    st.caption("This flow uses the same YOLO model path and Gemini pipeline as the other detectors. Only the teacher reference set changes.")
    
    model_path_text = st.text_input(
        "YOLO model path",
        value=pick_default_model_path(),
        key=f"student_model_path_{sheet_mode}",
    )
    uploaded = st.file_uploader(
        "Upload sheet image",
        type=["jpg", "jpeg", "png"],
        key=f"student_upload_{sheet_mode}",
    )

    # Check if reference exists
    reference_answers = load_reference(_reference_type_for_mode(sheet_mode))
    if not reference_answers:
        st.warning(f"No {sheet_mode} reference answers found. Teacher must create the matching reference first in the Teacher tab.")
    
    # Main processing logic when an image is uploaded
    if uploaded is not None:
        process_uploaded_image(uploaded, model_path_text, reference_answers, sheet_mode)


def process_uploaded_image(uploaded, model_path_text: str, reference_answers, sheet_mode: str):
  
    model_path = resolve_model_path(model_path_text)
    if not model_path.exists():
        st.error(
            "YOLO model not found. Use a valid path such as "
            f"{DEFAULT_MODEL_PATHS[0]}."
        )
        st.stop()

    data = np.frombuffer(uploaded.read(), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)

    if image is None:
        st.error("Could not decode image")
        st.stop()

    # IMPORTANT: Detect on ORIGINAL image (YOLO was trained on originals)
    detector = YoloLayoutDetector(str(model_path))
    detections = detector.detect(image)
    
    # Then align for better processing
    aligned = perspective_correct(image)
    
    # Check perspective correction quality
    if aligned is None or aligned.size == 0:
        st.error("Perspective correction failed")
        st.stop()
    
    # Show quality metrics
    alignment_quality = "Good" if aligned.shape == (2500, 1800, 3) else "Check"
    st.caption(
        f"Alignment: {alignment_quality} | Shape: {aligned.shape} | "
        f"Boxes detected on original: {len(detections)}"
    )

    vis = image.copy()  # Draw on ORIGINAL
    for det in detections:
        cv2.rectangle(vis, (det.x1, det.y1), (det.x2, det.y2), (0, 255, 0), 2)
        cv2.putText(
            vis,
            f"{det.label}:{det.confidence:.2f}",
            (det.x1, max(20, det.y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

    temp_input = None
    temp_aligned = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
            cv2.imwrite(tmp.name, image)  # Save ORIGINAL image
            temp_input = Path(tmp.name)

        # Also save aligned for visualization
        temp_aligned = temp_input.with_name(f"{temp_input.stem}_aligned.png")
        cv2.imwrite(str(temp_aligned), aligned)

        from src.OCR.gemini_ocr_extractor import GeminiOCRExtractor
        extractor = GeminiOCRExtractor(str(model_path), debug=True)
        extractor.prompt_style = sheet_mode
        with st.spinner("Running Gemini Vision extraction..."):
            ocr_result = extractor.extract(str(temp_input))

        # Wrap into a simple object matching what the display code expects
        class _Result:
            pass
        result          = _Result()
        result.answers  = ocr_result.answers   # keys = Box_0, Box_1 …
        result.is_valid = ocr_result.is_valid
        result.errors   = ocr_result.errors
        result.debug_info = ocr_result.debug_info
        result.box_interpretations = ocr_result.box_interpretations  # NEW: Transfer interpretations
        
        # Make extractor available so cell_crops can be read
        pipeline        = _Result()
        pipeline.extractor = extractor

    except Exception as exc:
        st.error(f"Pipeline failed: {exc}")
        st.stop()
    finally:
        for temp_path in (temp_input, temp_aligned):
            if temp_path is not None and temp_path.exists():
                temp_path.unlink(missing_ok=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Step 1: Original image")
        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), width="stretch")
    with col2:
        st.subheader("Step 2: YOLO detections")
        st.image(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB), width="stretch")

    display_extraction_results(result, sheet_mode)
    
    if reference_answers:
        display_grading_section(result, reference_answers)

    display_debug_info(result)


def display_extraction_results(result, sheet_mode: str):
    st.subheader("Step 3: Gemini Vision — Extracted Answers")

    if result.answers:
        # ── Answer cards (text only, no crop images) ───────────────────
        st.markdown("####  Extracted Answers")

        answers_list = list(result.answers.items())
        num_cols     = min(len(answers_list), MAX_DISPLAY_COLS)
        rows         = [answers_list[i:i+num_cols]
                        for i in range(0, len(answers_list), num_cols)]

        for row in rows:
            cols = st.columns(len(row))
            for col, (qid, adict) in zip(cols, row):
                answer    = adict.get("answer") or "—"
                answer_count = adict.get("answer_count", 1)
                box_label = qid.replace("_", " ")   # "Box_0" → "Box 0"
                
                # Format answer display
                if answer_count > 1:
                    # Multiple answers - show them nicely
                    answer_display = " | ".join(answer.split()) if " " in answer else answer
                else:
                    answer_display = answer
                
                with col:
                    st.markdown(
                        f"""
                        <div style="
                            border:2px solid #5a5a8a;
                            border-radius:10px;
                            padding:14px 6px 10px;
                            text-align:center;
                            background:linear-gradient(135deg,#1e1e2e,#2a2a40);
                            margin-bottom:8px;
                        ">
                            <div style="font-size:11px;color:#888;margin-bottom:4px;">{box_label}</div>
                            <div style="font-size:{'24px' if answer_count > 1 else '32px'};font-weight:bold;color:#c8c8ff;
                                        letter-spacing:{'1px' if answer_count > 1 else '3px'};line-height:1.1;">{answer_display}</div>
                            {'<div style="font-size:9px;color:#888;margin-top:2px;">×' + str(answer_count) + '</div>' if answer_count > 1 else ''}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

        # ── Summary line ───────────────────────────────────────────────
        st.markdown("---")
        all_answers = [(adict.get("answer") or "—") for _, adict in result.answers.items()]
        st.markdown("**All answers (left → right):**")
        st.code("  |  ".join(all_answers), language=None)
        st.success(
            f" Extracted **{len(result.answers)}** boxes"
        )
        
        # ── Toggle for box interpretation (hidden) ────────────────────
        st.markdown("---")
        # The Box Content Interpretation UI is intentionally hidden.
        # To re-enable, restore the checkbox below.
        # show_interpretations = st.checkbox(
        #     "Show Box Content Interpretation & Visual Quality Analysis",
        #     value=True,
        #     help="Gemini analyzes the visual content of each box to assess clarity and mark quality",
        #     key=f"show_box_interpretations_{sheet_mode}",
        # )
        show_interpretations = False

        # ── BOX INTERPRETATION & VISUAL QUALITY ─────────────────────────
        if show_interpretations and hasattr(result, 'box_interpretations') and result.box_interpretations:
            display_box_interpretations(result)
        elif show_interpretations:
            st.info(" Box interpretation data not available. Ensure Gemini API is properly configured.")

    else:
        st.warning(" No answers extracted — check image quality and YOLO detection")

    st.write(f"Valid result: {result.is_valid}")
    if result.errors:
        st.error(f"Errors: {'; '.join(result.errors)}")


def display_box_interpretations(result):
    # Step 3b is intentionally disabled per user request.
    # The original UI and data display are left in the file for reference
    # but are not executed. To restore, remove this return statement
    # and re-enable the checkbox above.
    return

    # --- original implementation (commented out) ---
    # st.markdown("---")
    # st.subheader("Step 3b: Box Content Interpretation & Visual Quality")
    # st.caption("Gemini analyzes the visual content of each box to assess clarity and confidence")
    # ...


def display_grading_section(result, reference_answers):
    st.markdown("---")
    st.subheader("Step 4: Grade Against Reference")
    
    # Compare using the smart parsing functions
    grading = compare_answers(result.answers, reference_answers)
    
    # Display score
    if grading["score"] >= SCORE_GOOD_THRESHOLD:
        score_color = "#00ff00"
    elif grading["score"] >= SCORE_OKAY_THRESHOLD:
        score_color = "#ffaa00"
    else:
        score_color = "#ff0000"
    
    st.markdown(f"""
    <div style="
        border:3px solid {score_color};
        border-radius:15px;
        padding:20px;
        text-align:center;
        background:rgba(0,0,0,0.3);
        margin:10px 0;
    ">
        <div style="font-size:14px;color:#888;margin-bottom:8px;">SCORE</div>
        <div style="font-size:48px;font-weight:bold;color:{score_color};">{grading['score']:.1f}%</div>
        <div style="font-size:14px;color:#aaa;margin-top:8px;">{grading['correct']}/{grading['total']} Correct</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Show comparison table
    st.write("**Answer Comparison:**")
    comp_data = []
    for detail in grading["details"]:
        comp_data.append({
            "Question": detail["question"],
            "Reference": detail["reference"],
            "Student": detail["student"],
            "Status": "✅ Correct" if detail["correct"] else "❌ Wrong"
        })
    
    st.dataframe(comp_data, width="stretch", hide_index=True)


def display_debug_info(result):
    """Display debug information and detailed answer data."""
    with st.expander(" Debug Info - All Boxes"):
        for qid, adict in result.answers.items():
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**{qid}**")
            with col2:
                st.write(f"Answer: {adict.get('answer', '—')}")
            
            # Show any errors for this box
            if adict.get('error'):
                st.warning(f"Error for {qid}: {adict.get('error')}")
            
            # Show interpretation details if available
            if hasattr(result, 'box_interpretations') and qid in result.box_interpretations:
                interp = result.box_interpretations[qid]
                st.write(f"**Interpretation:**")
                st.write(f"- Type: {interp.get('content_type')}")
                st.write(f"- Clarity: {interp.get('visual_clarity')} (score: {interp.get('visual_clarity_score'):.2f})")
                st.write(f"- Confidence: {interp.get('confidence'):.0%}")
                st.write(f"- Explanation: {interp.get('explanation')}")
                if interp.get('recommendations'):
                    st.write(f"- Recommendations: {', '.join(interp.get('recommendations'))}")
        
        st.json(result.debug_info)
