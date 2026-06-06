from __future__ import annotations

import copy
import importlib
import json
import sys
import tempfile
import warnings
from io import StringIO
from pathlib import Path
import tempfile
import os

warnings.filterwarnings("ignore")

import cv2
import numpy as np
import pandas as pd
import streamlit as st

# Add parent directory to path for imports
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import PAGE_CONFIG
from answer_processing import compare_answers
from reference_manager import load_reference
from gemini_omr import GeminiOMRChecker
from pages.teacher_page import render_teacher_page
from pages.student_page import render_circle_completion_page, render_student_page

OMRCHECKER_ROOT = ROOT / "OMRChecker"
OMRCHECKER_PATH = str(OMRCHECKER_ROOT)


def get_sample_templates() -> dict[str, str]:
    templates: dict[str, str] = {}
    samples_dir = OMRCHECKER_ROOT / "samples"

    if not samples_dir.exists():
        return templates

    for template_file in samples_dir.rglob("template.json"):
        relative_path = template_file.relative_to(samples_dir)
        parts = list(relative_path.parent.parts)

        if parts:
            template_name = " → ".join(parts) if len(parts) > 1 else parts[0]
        else:
            template_name = "root"

        templates[template_name] = str(template_file)

    return templates


@st.cache_resource
def load_omrchecker_components() -> dict[str, object] | None:
    if not OMRCHECKER_ROOT.exists():
        return None

    if OMRCHECKER_PATH not in sys.path:
        sys.path.insert(0, OMRCHECKER_PATH)

    root_src_modules = {
        name: module
        for name, module in sys.modules.items()
        if name == "src" or name.startswith("src.")
    }

    try:
        for name in list(root_src_modules):
            sys.modules.pop(name, None)

        importlib.invalidate_caches()

        template_module = importlib.import_module("src.template")
        defaults_module = importlib.import_module("src.defaults")
        core_module = importlib.import_module("src.core")
        image_module = importlib.import_module("src.utils.image")

        return {
            "Template": template_module.Template,
            "CONFIG_DEFAULTS": defaults_module.CONFIG_DEFAULTS,
            "ImageInstanceOps": core_module.ImageInstanceOps,
            "ImageUtils": image_module.ImageUtils,
        }
    except Exception as exc:
        st.error(f"OMRChecker import error: {exc}")
        return None
    finally:
        for name in list(sys.modules):
            if name == "src" or name.startswith("src."):
                if name not in root_src_modules:
                    sys.modules.pop(name, None)
        sys.modules.update(root_src_modules)


@st.cache_resource
def load_template(template_path: str):
    components = load_omrchecker_components()
    if not components:
        st.error("OMRChecker not available")
        return None

    try:
        template_cls = components["Template"]
        config_defaults = components["CONFIG_DEFAULTS"]
        return template_cls(Path(template_path), config_defaults)
    except Exception as exc:
        st.error(f"Failed to load template: {exc}")
        return None


def build_omrchecker_config(auto_align: bool):
    components = load_omrchecker_components()
    if not components:
        return None

    config = copy.deepcopy(components["CONFIG_DEFAULTS"])
    config.alignment_params.auto_align = auto_align
    return config


def process_omr_image(image: np.ndarray, template, config=None) -> dict:
    try:
        components = load_omrchecker_components()
        if not components:
            return {
                "success": False,
                "error": "OMRChecker is not available",
                "response": None,
                "processed_image": None,
            }

        if config is None:
            config = build_omrchecker_config(auto_align=False)

        if config is None:
            return {
                "success": False,
                "error": "Failed to build OMRChecker configuration",
                "response": None,
                "processed_image": None,
            }

        image_utils = components["ImageUtils"]
        image_instance_ops = components["ImageInstanceOps"](config)

        if image.ndim == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        image = image_utils.resize_util(
            image,
            config.dimensions.processing_width,
            config.dimensions.processing_height,
        )

        processed = image_instance_ops.apply_preprocessors("temp", image, template)
        response = image_instance_ops.read_omr_response(template, processed, "temp")

        answers = {}
        is_invalid = None
        multi_mark_count = None

        if isinstance(response, (list, tuple)):
            if len(response) > 0 and isinstance(response[0], dict):
                answers = response[0]
            if len(response) > 2:
                is_invalid = response[2]
            if len(response) > 3:
                multi_mark_count = response[3]
        elif isinstance(response, dict):
            answers = response

        return {
            "success": True,
            "response": answers,
            "processed_image": processed,
            "raw_response": response,
            "is_invalid": is_invalid,
            "multi_mark_count": multi_mark_count,
        }
    except Exception as exc:
        return {
            "success": False,
            "error": str(exc),
            "response": None,
            "processed_image": None,
            "raw_response": None,
            "is_invalid": None,
            "multi_mark_count": None,
        }


def render_template_single_page():
    st.markdown("### Process a single OMRChecker sheet")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Template Selection")
        template_source = st.radio(
            "Template source",
            ["Use Sample", "Upload Custom"],
            label_visibility="collapsed",
            key="omr_template_source_single",
        )

        template_path = None
        if template_source == "Use Sample":
            templates = get_sample_templates()
            if templates:
                template_name = st.selectbox(
                    "Select template",
                    list(templates.keys()),
                    label_visibility="collapsed",
                    key="omr_template_select_single",
                )
                template_path = templates[template_name]
                st.success(f"Template: {template_name}")
            else:
                st.warning("No sample templates found in OMRChecker/samples/")
                st.stop()
        else:
            template_file = st.file_uploader(
                "Upload template JSON",
                type="json",
                label_visibility="collapsed",
                key="omr_template_upload_single",
            )
            if template_file:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as tmp:
                    tmp.write(template_file.getbuffer())
                    template_path = tmp.name
                st.success("Template uploaded")
            else:
                st.warning("Please upload a template JSON file")
                st.stop()

    with col2:
        st.markdown("#### Image Upload")
        uploaded_img = st.file_uploader(
            "Upload OMR sheet",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed",
            key="omr_image_upload_single",
        )

    if uploaded_img and template_path:
        template = load_template(template_path)
        if not template:
            st.stop()

        img_data = np.frombuffer(uploaded_img.read(), dtype=np.uint8)
        image = cv2.imdecode(img_data, cv2.IMREAD_COLOR)

        if image is None:
            st.error("Could not decode image")
            st.stop()

        with st.expander("⚙️ Processing Configuration"):
            auto_align = st.checkbox("Auto alignment", value=True, key="omr_auto_align_single")
            show_raw_response = st.checkbox(
                "Show raw response details",
                value=False,
                key="omr_raw_response_single",
            )

        if st.button("🔍 Process Image", type="primary", key="omr_process_single"):
            with st.spinner("Processing OMR sheet..."):
                config = build_omrchecker_config(auto_align=auto_align)
                result = process_omr_image(image, template, config=config)

            if result["success"]:
                st.success("Processing complete!")

                col1, col2 = st.columns(2)
                with col1:
                    st.image(image, caption="Original Image", width="stretch")
                with col2:
                    if result["processed_image"] is not None:
                        processed_image = result["processed_image"]
                        if processed_image.ndim == 2 or processed_image.shape[-1] == 1:
                            processed_rgb = processed_image
                        else:
                            processed_rgb = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
                        st.image(
                            processed_rgb,
                            caption="Processed Image",
                            width="stretch",
                        )

                st.subheader("📊 Extracted Answers")
                if result["response"] and isinstance(result["response"], dict) and len(result["response"]) > 0:
                    answers = result["response"]
                    results_df = pd.DataFrame(list(answers.items()), columns=["Question", "Answer"])
                    st.dataframe(results_df, width="stretch")

                    csv_buffer = StringIO()
                    results_df.to_csv(csv_buffer, index=False)

                    col1, col2 = st.columns(2)
                    with col1:
                        st.download_button(
                            label="📥 Download CSV",
                            data=csv_buffer.getvalue(),
                            file_name="omr_result.csv",
                            mime="text/csv",
                            key="omr_download_csv_single",
                        )
                    with col2:
                        json_str = json.dumps(answers, indent=2)
                        st.download_button(
                            label="📥 Download JSON",
                            data=json_str,
                            file_name="omr_result.json",
                            mime="application/json",
                            key="omr_download_json_single",
                        )

                    if show_raw_response:
                        st.json(result.get("raw_response"))
                else:
                    st.warning("No answers extracted from the sheet")

                if result.get("multi_mark_count") is not None:
                    st.caption(f"Multi-mark count: {result['multi_mark_count']}")
                if result.get("is_invalid") is not None:
                    st.caption(f"Invalid result: {result['is_invalid']}")
            else:
                st.error(f"Processing failed: {result['error']}")


def render_template_batch_page():
    st.markdown("### Batch process multiple OMRChecker sheets")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Template & Images")
        template_source = st.radio(
            "Template source",
            ["Use Sample", "Upload Custom"],
            label_visibility="collapsed",
            key="omr_template_source_batch",
        )

        template_path = None
        if template_source == "Use Sample":
            templates = get_sample_templates()
            if templates:
                template_name = st.selectbox(
                    "Select template",
                    list(templates.keys()),
                    label_visibility="collapsed",
                    key="omr_template_select_batch",
                )
                template_path = templates[template_name]
            else:
                st.warning("No sample templates found")
                st.stop()
        else:
            template_file = st.file_uploader(
                "Upload template JSON",
                type="json",
                label_visibility="collapsed",
                key="omr_template_upload_batch",
            )
            if template_file:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as tmp:
                    tmp.write(template_file.getbuffer())
                    template_path = tmp.name
            else:
                st.warning("Please upload a template")
                st.stop()

    with col2:
        uploaded_images = st.file_uploader(
            "Upload multiple images",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
            label_visibility="collapsed",
            key="omr_batch_images",
        )

    if uploaded_images and template_path:
        with st.expander("⚙️ Batch Configuration"):
            auto_align = st.checkbox("Auto alignment", value=True, key="omr_auto_align_batch")

        if st.button("🔄 Process All Images", type="primary", key="omr_process_batch"):
            results_list = []

            template = load_template(template_path)
            if not template:
                st.stop()

            progress_bar = st.progress(0)
            status_text = st.empty()
            config = build_omrchecker_config(auto_align=auto_align)

            for idx, uploaded_img in enumerate(uploaded_images):
                status_text.text(f"Processing {idx + 1}/{len(uploaded_images)}...")

                img_data = np.frombuffer(uploaded_img.read(), dtype=np.uint8)
                image = cv2.imdecode(img_data, cv2.IMREAD_COLOR)

                if image is not None:
                    result = process_omr_image(image, template, config=config)
                    results_list.append(
                        {
                            "filename": uploaded_img.name,
                            "success": result["success"],
                            "response": result.get("response"),
                            "error": result.get("error"),
                        }
                    )
                else:
                    results_list.append(
                        {
                            "filename": uploaded_img.name,
                            "success": False,
                            "response": None,
                            "error": "Could not decode image",
                        }
                    )

                progress_bar.progress((idx + 1) / len(uploaded_images))

            st.success(f"✅ Processed {len(results_list)} images")

            st.subheader("Batch Results")
            results_df = pd.DataFrame(
                [
                    {
                        "File": r["filename"],
                        "Status": "✅ Success" if r["success"] else "❌ Failed",
                        "Details": json.dumps(r["response"]) if r["success"] and r["response"] is not None else r["error"],
                    }
                    for r in results_list
                ]
            )

            st.dataframe(results_df, width="stretch")

            csv_buffer = StringIO()
            results_df.to_csv(csv_buffer, index=False)
            st.download_button(
                label="📥 Download Results CSV",
                data=csv_buffer.getvalue(),
                file_name="omr_batch_results.csv",
                mime="text/csv",
                key="omr_download_batch_csv",
            )


def render_bubble_detector_page():
    st.markdown("### Bubble Detector")
    st.markdown("Use Gemini to detect marked bubbles, then compare the result with the bubble reference already saved in the app.")

    reference_answers = load_reference("bubble")
    if not reference_answers:
        st.warning("No bubble reference found. Create and save the bubble reference first in the Teacher tab.")

    api_key = os.environ.get("GOOGLE_API_KEY")
    model_name = "gemini-2.5-flash"

    uploaded_img = st.file_uploader(
        "Upload answer sheet image",
        type=["jpg", "jpeg", "png"],
        key="bubble_detector_upload",
    )

    if st.button("Detect Bubbles", type="primary", key="bubble_detector_run"):
        if not api_key:
            st.error("Set GOOGLE_API_KEY in the environment before running Bubble Detector.")
        elif not uploaded_img:
            st.warning("Upload an answer sheet image first.")
        elif not reference_answers:
            st.warning("Save the teacher reference first before running bubble detection.")
        else:
            temp_path = None
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_img.name).suffix or ".png") as tmp:
                    tmp.write(uploaded_img.getbuffer())
                    temp_path = Path(tmp.name)

                checker = GeminiOMRChecker(api_key=api_key, model_name=model_name)
                result = checker.extract_from_image(str(temp_path))

                grading = compare_answers(result.answers, reference_answers)

                col1, col2 = st.columns(2)
                with col1:
                    st.image(uploaded_img, caption="Uploaded Sheet", width="stretch")
                with col2:
                    st.metric("Gemini confidence", f"{result.confidence:.2f}")
                    st.metric("Questions found", str(result.questions_found))
                    st.metric("Needs review", "Yes" if result.needs_review else "No")

                st.subheader("Detected Answers")
                detected_df = pd.DataFrame(
                    [(question, answer) for question, answer in sorted(result.answers.items(), key=lambda item: item[0])],
                    columns=["Question", "Answer"],
                )
                st.dataframe(detected_df, width="stretch", hide_index=True)

                st.markdown("---")
                st.subheader("Grading Against Teacher Reference")
                st.markdown(f"**Score:** {grading['score']:.1f}%  |  **Correct:** {grading['correct']}  |  **Incorrect:** {grading['incorrect']}")

                comparison_df = pd.DataFrame(
                    [
                        {
                            "Question": item["question"],
                            "Reference": item["reference"],
                            "Detected": item["student"],
                            "Status": "✅ Correct" if item["correct"] else "❌ Wrong",
                        }
                        for item in grading["details"]
                    ]
                )
                st.dataframe(comparison_df, width="stretch", hide_index=True)

                if result.needs_review:
                    st.info("Gemini confidence is below the review threshold.")

                with st.expander("Raw Gemini response"):
                    st.code(result.raw_response or "No raw response captured.")
            except Exception as exc:
                st.error(f"Bubble detection failed: {exc}")
            finally:
                if temp_path is not None and temp_path.exists():
                    temp_path.unlink(missing_ok=True)


def main():
    st.set_page_config(**PAGE_CONFIG)
    st.title("AI-Enhanced Multi-Format OMR")
    st.caption("Pipeline: YOLO detection → Gemini Vision API text extraction | Teacher reference builder | Template-based OMRChecker")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "👨‍🏫 Teacher - Create Reference",
        "👨‍🎓 Table Structure Detector",
        "⭕ Circle Completion",
        "📋 Template OMRChecker",
        "🫧 Bubble Detector",
    ])

    # ==================== TEACHER MODE ====================
    with tab1:
        render_teacher_page()

    # ==================== STUDENT MODE ====================
    with tab2:
        render_student_page()

    # ==================== CIRCLE COMPLETION MODE ====================
    with tab3:
        render_circle_completion_page()

    # ==================== TEMPLATE-BASED OMR MODE ====================
    with tab4:
        if not OMRCHECKER_ROOT.exists():
            st.warning("OMRChecker folder was not found in this workspace.")
            st.stop()

        subtab1, subtab2 = st.tabs(["Single Image", "Batch Process"])
        with subtab1:
            render_template_single_page()
        with subtab2:
            render_template_batch_page()

    # ==================== BUBBLE DETECTOR ====================
    with tab5:
        render_bubble_detector_page()


if __name__ == "__main__":
    main()

