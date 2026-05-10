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
st.caption("Pipeline: YOLO detection → Grid Detection → EasyOCR character extraction")

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
            pipeline = get_pipeline(str(model_path))
            pipeline.extractor.debug = True  # Enable debug output
            with st.spinner("Running box-sheet extraction pipeline..."):
                result = pipeline.process_image(str(temp_input))  # Use ORIGINAL image
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
        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), width='stretch')
    with col2:
        st.subheader("Step 2: YOLO detections (original image)")
        st.image(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB), width='stretch')

    if sheet_class == "Box answer sheet":
        st.subheader("Step 3: Grid Detection + EasyOCR results")
    else:
        st.subheader("Step 3: Circle-fill OMR results")
    
    # Show detection count and method
    extraction_mode = result.debug_info.get('mode', 'unknown')
    extraction_method = result.debug_info.get('extraction_method', 'unknown')
    st.info(f"Detected {result.debug_info.get('total_boxes_detected', 0)} boxes | Mode: **{extraction_mode}** | Method: **{extraction_method}**")
    
    if result.answers:
        st.success(f"Extracted {len(result.answers)} answers")
        for qid, answer_dict in result.answers.items():
            answer = answer_dict.get('answer', '?')
            conf = answer_dict.get('confidence', 0.0)
            num_cells = answer_dict.get('num_cells', 0)
            status = "🟢" if conf > 0.7 else "🟡" if conf > 0.4 else "🔴"
            if sheet_class == "Circle fill sheet":
                score_map = answer_dict.get("score_map", {})
                st.write(
                    f"{status} {qid}: **{answer}** | Choices: {num_cells} | Confidence: {conf:.2f} | "
                    f"Scores: {score_map}"
                )
            else:
                st.write(f"{status} {qid}: **{answer}** | Cells: {num_cells} | Confidence: {conf:.2f}")

        if sheet_class == "Box answer sheet":
            st.subheader("Grid Detection Visualizations")
            if pipeline.extractor.grid_visualizations:
                cols = st.columns(2)
                for idx, (qid, grid_image) in enumerate(pipeline.extractor.grid_visualizations.items()):
                    col_idx = idx % 2
                    with cols[col_idx]:
                        st.write(f"**{qid}** - Grid overlay")
                        st.image(cv2.cvtColor(grid_image, cv2.COLOR_BGR2RGB), width='stretch')
            else:
                st.info("No grid visualizations available")
        else:
            st.subheader("Circle-fill Visualizations")
            if pipeline.choice_visualizations:
                cols = st.columns(2)
                for idx, (qid, overlay) in enumerate(pipeline.choice_visualizations.items()):
                    col_idx = idx % 2
                    with cols[col_idx]:
                        st.write(f"**{qid}** - Bubble overlay")
                        st.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), width='stretch')
            else:
                st.info("No circle visualizations available")
    else:
        st.warning(" No answers extracted - check image quality and detection")
    
    st.write(f"Valid result: {result.is_valid}")
    if result.errors:
        st.error(f" Errors: {'; '.join(result.errors)}")
    
    # Show debug info
    with st.expander(" Debug Info"):
        st.json(result.debug_info)

