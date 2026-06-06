
from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import json
from io import StringIO

import warnings
warnings.filterwarnings("ignore")

import cv2
import numpy as np
import pandas as pd
import streamlit as st
# PIL not used directly in this file (marker UI removed)

# Setup path
ROOT = Path(__file__).resolve().parents[1]

# Add OMRChecker to path first (before ROOT)
OMRCHECKER_PATH = str(ROOT / "OMRChecker")
if OMRCHECKER_PATH not in sys.path:
    sys.path.insert(0, OMRCHECKER_PATH)

# Then add ROOT
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Import OMRChecker components
try:
    from src.template import Template
    from src.defaults import CONFIG_DEFAULTS
    from src.core import ImageInstanceOps
    from src.utils.image import ImageUtils
    OMRCHECKER_AVAILABLE = True
except ImportError as e:
    OMRCHECKER_AVAILABLE = False
    print(f"OMRChecker import error: {e}")
    import traceback
    traceback.print_exc()

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_sample_templates() -> dict:
    templates = {}
    samples_dir = ROOT / "OMRChecker" / "samples"
    
    if not samples_dir.exists():
        return templates
    
    # Recursively search for template.json files
    for template_file in samples_dir.rglob("template.json"):
        # Create a readable name from the path
        relative_path = template_file.relative_to(samples_dir)
        # Get parent directory names
        parts = list(relative_path.parent.parts)
        
        if parts:
            # Use path like "answer-key/using-csv" or "community/Antibodyy"
            template_name = " → ".join(parts) if len(parts) > 1 else parts[0]
        else:
            template_name = "root"
        
        templates[template_name] = str(template_file)
    
    return templates

@st.cache_resource
def load_template(template_path: str):
    try:
        if OMRCHECKER_AVAILABLE:
            # Convert string to Path object (Template expects Path, not str)
            template_path = Path(template_path)
            template = Template(template_path, CONFIG_DEFAULTS)
            return template
        else:
            st.error("OMRChecker not available")
            return None
    except Exception as e:
        st.error(f"Failed to load template: {e}")
        return None

def process_omr_image(image: np.ndarray, template, config=None) -> dict:
    try:
        if config is None:
            config = CONFIG_DEFAULTS
        
        # Resize to template dimensions
        image = ImageUtils.resize_util(
            image,
            config.dimensions.processing_width,
            config.dimensions.processing_height
        )
        
        # Create image ops instance
        img_ops = ImageInstanceOps(config)
        
        # Apply preprocessors
        processed = img_ops.apply_preprocessors("temp", image, template)
        
        # Read OMR response
        response = img_ops.read_omr_response(template, processed, "temp")
        
        # Extract answers from response
        # response can be a tuple/list: (answers_dict, processed_image, is_invalid, multi_mark_count)
        answers = {}
        if isinstance(response, (list, tuple)):
            if len(response) > 0 and isinstance(response[0], dict):
                answers = response[0]  # Get answers dict (first element)
        elif isinstance(response, dict):
            answers = response  # Already a dict
        
        return {
            "success": True,
            "response": answers,
            "processed_image": processed
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "response": None,
            "processed_image": None
        }

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="OMRChecker - Streamlit",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📋 OMRChecker - Streamlit App")
st.markdown("*Process OMR sheets with template-based detection and answer extraction*")


# ============================================================================
# MAIN APP
# ============================================================================

# Tabs for different modes
tab1, tab2 = st.tabs(["Single Image", "Batch Process"])

# ========== TAB 1: SINGLE IMAGE ==========
with tab1:
    st.markdown("### Process Single OMR Sheet")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Template Selection")
        template_source = st.radio(
            "Template source",
            ["Use Sample", "Upload Custom"],
            label_visibility="collapsed"
        )
        
        if template_source == "Use Sample":
            templates = get_sample_templates()
            if templates:
                template_name = st.selectbox(
                    "Select template",
                    list(templates.keys()),
                    label_visibility="collapsed"
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
                label_visibility="collapsed"
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
            label_visibility="collapsed"
        )
    
    if uploaded_img and template_path:
        # Load template
        template = load_template(template_path)
        if not template:
            st.stop()
        
        # Load image
        img_data = np.frombuffer(uploaded_img.read(), dtype=np.uint8)
        image = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
        
        if image is None:
            st.error("Could not decode image")
            st.stop()
        
        # Processing options
        with st.expander("⚙️ Processing Configuration"):
            auto_align = st.checkbox("Auto alignment", value=True)
            show_debug = st.checkbox("Show debug visualization", value=False)
            bubble_threshold = st.slider(
                "Bubble fill threshold",
                0.0, 1.0, 0.5, 0.05
            )
        
        # Process
        if st.button("🔍 Process Image", type="primary"):
            with st.spinner("Processing OMR sheet..."):
                result = process_omr_image(image, template)
            
            if result["success"]:
                st.success("Processing complete!")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.image(image, caption="Original Image", width="stretch")
                
                with col2:
                    if result["processed_image"] is not None:
                        # Convert BGR to RGB for display
                        processed_rgb = cv2.cvtColor(result["processed_image"], cv2.COLOR_BGR2RGB)
                        st.image(
                            processed_rgb,
                            caption="Processed Image",
                            width="stretch"
                        )
                
                # Results
                st.subheader("📊 Extracted Answers")
                if result["response"] and isinstance(result["response"], dict) and len(result["response"]) > 0:
                    answers = result["response"]
                    
                    results_df = pd.DataFrame(
                        list(answers.items()),
                        columns=["Question", "Answer"]
                    )
                    st.dataframe(results_df, width="stretch")
                    
                    # Download options
                    csv_buffer = StringIO()
                    results_df.to_csv(csv_buffer, index=False)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.download_button(
                            label="📥 Download CSV",
                            data=csv_buffer.getvalue(),
                            file_name="omr_result.csv",
                            mime="text/csv"
                        )
                    with col2:
                        json_str = json.dumps(answers, indent=2)
                        st.download_button(
                            label="📥 Download JSON",
                            data=json_str,
                            file_name="omr_result.json",
                            mime="application/json"
                        )
                else:
                    st.warning("No answers extracted from the sheet")
            else:
                st.error(f"Processing failed: {result['error']}")

# ========== TAB 2: BATCH PROCESS ==========
with tab2:
    st.markdown("### Batch Process Multiple Sheets")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Template & Images")
        template_source = st.radio(
            "Template source",
            ["Use Sample", "Upload Custom"],
            label_visibility="collapsed",
            key="batch_template"
        )
        
        if template_source == "Use Sample":
            templates = get_sample_templates()
            if templates:
                template_name = st.selectbox(
                    "Select template",
                    list(templates.keys()),
                    label_visibility="collapsed",
                    key="batch_select"
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
                key="batch_upload"
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
            label_visibility="collapsed"
        )
    
    if uploaded_images and template_path:
        # Configuration
        with st.expander("⚙️ Batch Configuration"):
            auto_align = st.checkbox("Auto alignment", value=True, key="batch_align")
        
        if st.button("🔄 Process All Images", type="primary"):
            results_list = []
            
            template = load_template(template_path)
            if not template:
                st.stop()
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for idx, uploaded_img in enumerate(uploaded_images):
                status_text.text(f"Processing {idx + 1}/{len(uploaded_images)}...")
                
                img_data = np.frombuffer(uploaded_img.read(), dtype=np.uint8)
                image = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
                
                if image is not None:
                    result = process_omr_image(image, template)
                    results_list.append({
                        "filename": uploaded_img.name,
                        "success": result["success"],
                        "response": result.get("response"),
                        "error": result.get("error")
                    })
                
                progress_bar.progress((idx + 1) / len(uploaded_images))
            
            st.success(f"✅ Processed {len(results_list)} images")
            
            # Display results
            st.subheader("Batch Results")
            
            results_df = pd.DataFrame([
                {
                    "File": r["filename"],
                    "Status": "✅ Success" if r["success"] else "❌ Failed",
                    "Details": json.dumps(r["response"]) if r["success"] else r["error"]
                }
                for r in results_list
            ])
            
            st.dataframe(results_df, width="stretch")
            
            # Export
            csv_buffer = StringIO()
            results_df.to_csv(csv_buffer, index=False)
            st.download_button(
                label="📥 Download Results CSV",
                data=csv_buffer.getvalue(),
                file_name="omr_batch_results.csv",
                mime="text/csv"
            )

