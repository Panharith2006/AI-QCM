from __future__ import annotations

from pathlib import Path
import sys
import tempfile

import warnings
warnings.filterwarnings("ignore")

import cv2
import numpy as np
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.alignment.perspective import perspective_correct
from src.detection.yolo_layout import YoloLayoutDetector
from src.circle_pipeline import CircleFillPipeline
from src.xai.yolo_xai import YOLOXAI
from src.pipeline import OMRPipeline

DEFAULT_MODEL_PATHS = [
    ROOT / "artifacts" / "yolo" / "best.pt",
]
# Utility functions for model path resolution and default selection.
def resolve_model_path(path_text: str) -> Path:
    path = Path(path_text).expanduser()
    if path.is_absolute():
        return path
    return (ROOT / path).resolve()

# Pick the first existing model path from the defaults, or return the first one if none exist.
def pick_default_model_path() -> str:
    for candidate in DEFAULT_MODEL_PATHS:
        if candidate.exists():
            return str(candidate)
    return str(DEFAULT_MODEL_PATHS[0])

# Streamlit app for AI-enhanced OMR processing with YOLO layout detection and EasyOCR extraction.
st.set_page_config(page_title="AI OMR", layout="wide")
st.title("AI-Enhanced Multi-Format OMR")
st.caption("Pipeline: YOLO detection → Gemini Vision API text extraction")

model_path_text = st.text_input("YOLO model path", value=pick_default_model_path())
sheet_class = st.selectbox(
    "Sheet class",
    ["Box answer sheet", "Circle fill sheet"],
    index=0,
)
uploaded = st.file_uploader("Upload sheet image", type=["jpg", "jpeg", "png"])

# Main processing logic when an image is uploaded
def get_pipeline(yolo_path: str) -> OMRPipeline:
    return OMRPipeline(yolo_path)


def get_circle_pipeline(yolo_path: str) -> CircleFillPipeline:
    return CircleFillPipeline(yolo_path)

if uploaded is not None:
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
        f"Class: {sheet_class} | Alignment: {alignment_quality} | Shape: {aligned.shape} | "
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

        if sheet_class == "Box answer sheet":
            from src.OCR.ocr_extractor import OCRExtractor
            extractor = OCRExtractor(str(model_path), debug=True)
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
            # Make extractor available so cell_crops can be read
            pipeline        = _Result()
            pipeline.extractor = extractor

        else:
            pipeline = get_circle_pipeline(str(model_path))
            pipeline.debug = True
            with st.spinner("Running circle-fill extraction pipeline..."):
                result = pipeline.process_image(str(temp_input))

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
        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), use_container_width=True)
    with col2:
        st.subheader("Step 2: YOLO detections")
        st.image(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB), use_container_width=True)

    # ── Result header ─────────────────────────────────────────────────────────
    if sheet_class == "Box answer sheet":
        st.subheader("Step 3: Gemini Vision — Extracted Answers")
    else:
        st.subheader("Step 3: Circle-fill OMR results")

    extraction_mode   = result.debug_info.get("mode", "unknown")
    extraction_method = result.debug_info.get("extraction_method", "unknown")
    st.info(
        f"Detected **{result.debug_info.get('total_boxes_detected', 0)}** region(s) | "
        f"Mode: `{extraction_mode}` | Method: `{extraction_method}`"
    )

    if result.answers:
        if sheet_class == "Box answer sheet":
            # ── Answer cards (text only, no crop images) ───────────────────
            st.markdown("####  Extracted Answers")

            answers_list = list(result.answers.items())
            num_cols     = min(len(answers_list), 6)
            rows         = [answers_list[i:i+num_cols]
                            for i in range(0, len(answers_list), num_cols)]

            for row in rows:
                cols = st.columns(len(row))
                for col, (qid, adict) in zip(cols, row):
                    answer    = adict.get("answer") or "—"
                    conf      = adict.get("confidence", 0.0)
                    box_label = qid.replace("_", " ")   # "Box_0" → "Box 0"
                    badge     = "" if conf > 0.7 else "" if conf > 0.3 else ""
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
                                <div style="font-size:32px;font-weight:bold;color:#c8c8ff;
                                            letter-spacing:3px;line-height:1.1;">{answer}</div>
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
                f" Extracted **{len(result.answers)}** boxes | "
                f"Non-empty: **{result.debug_info.get('questions_found', 0)}**"
            )


        else:
            # Circle fill sheet display (unchanged)
            st.success(f"Extracted {len(result.answers)} answers")
            for qid, answer_dict in result.answers.items():
                answer   = answer_dict.get("answer", "?")
                conf     = answer_dict.get("confidence", 0.0)
                status   = "" if conf > 0.7 else "" if conf > 0.4 else ""
            st.markdown("#### Extracted Answers")
            answers_list = list(result.answers.items())
            num_cols     = min(len(answers_list), 6)
            rows         = [answers_list[i:i+num_cols]
                            for i in range(0, len(answers_list), num_cols)]
            for row in rows:
                cols = st.columns(len(row))
                for col, (qid, adict) in zip(cols, row):
                    answer = adict.get("answer") or "—"
                    with col:
                        st.markdown(
                            f"""
                            <div style="
                                border:2px solid #5a8a5a;
                                border-radius:10px;
                                padding:14px 6px 10px;
                                text-align:center;
                                background:linear-gradient(135deg,#1e2e1e,#2a402a);
                                margin-bottom:8px;
                            ">
                                <div style="font-size:11px;color:#888;margin-bottom:4px;">{qid}</div>
                                <div style="font-size:32px;font-weight:bold;color:#c8ffc8;
                                            letter-spacing:3px;line-height:1.1;">{answer}</div>
                                <div style="font-size:10px;color:#666;margin-top:4px;">{badge} {conf:.0%}</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

            st.markdown("---")
            all_answers = [(adict.get("answer") or "—") for _, adict in result.answers.items()]
            st.markdown("**All answers (left → right):**")
            st.code("  |  ".join(all_answers), language=None)
            found = sum(1 for _, a in result.answers.items() if a.get("answer"))
            st.success(f" Extracted **{len(result.answers)}** questions | Non-empty: **{found}**")


    else:
        st.warning(" No answers extracted — check image quality and YOLO detection")

    st.write(f"Valid result: {result.is_valid}")
    if result.errors:
        st.error(f"Errors: {'; '.join(result.errors)}")

    with st.expander(" Debug Info"):
        st.json(result.debug_info)

