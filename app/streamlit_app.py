from __future__ import annotations

from pathlib import Path
import sys

import cv2
import numpy as np
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.alignment.perspective import perspective_correct
from src.detection.yolo_layout import YoloLayoutDetector


st.set_page_config(page_title="AI OMR", layout="wide")
st.title("AI-Enhanced Multi-Format OMR")
st.caption("Optimized build: YOLO block detection + OpenCV extraction")

model_path = st.text_input("YOLO model path", value="artifacts/yolo/best.pt")
uploaded = st.file_uploader("Upload sheet image", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    data = np.frombuffer(uploaded.read(), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)

    if image is None:
        st.error("Could not decode image")
        st.stop()

    aligned = perspective_correct(image)

    detector = YoloLayoutDetector(model_path)
    detections = detector.detect(aligned)

    vis = aligned.copy()
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

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Aligned image")
        st.image(cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB), use_container_width=True)
    with col2:
        st.subheader("Detections")
        st.image(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB), use_container_width=True)

    st.write(f"Detected blocks: {len(detections)}")
