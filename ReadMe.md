# AI-Enhanced Multi-Format OMR System

This document explains the full technical stack and system architecture in concrete detail for implementation, training, and deployment.

## 1. System Objective

The system processes teacher and student answer sheets, detects answer regions, extracts answers, compares results, and returns final score with review indicators.

Core requirement:
1. Robust to camera angle, lighting variation, and layout changes.
2. Works with multiple answer scopes.
3. Provides visual and numerical outputs for human verification.

## 2. Technical Stack in Detail

### 2.1 Language and Runtime

Python is the primary language for all pipeline stages.

Recommended runtime:
1. Python 3.10 or 3.11
2. Virtual environment for dependency isolation
3. Windows PowerShell for local execution

### 2.2 Core Libraries and Their Roles

1. OpenCV
- Responsibility: image decode, color conversion, thresholding, perspective correction, drawing annotations.
- Used in: preprocessing, alignment, and visualization stages.

2. NumPy
- Responsibility: image array operations and numerical transformations.
- Used in: image loading buffers and matrix operations.

3. Ultralytics YOLOv8
- Responsibility: object detection of answer block regions.
- Used in: layout detection model training and inference.

4. Streamlit
- Responsibility: user interface for uploading sheets and viewing results.
- Used in: demo and operator workflow.

5. PyYAML
- Responsibility: parsing dataset configuration and class mapping.
- Used in: YOLO configuration (training handled in Colab).

6. Pytest
- Responsibility: unit and regression tests.
- Used in: scoring and logic verification.

### 2.3 Project Modules and Responsibilities

1. src/preprocessing
- Image enhancement helpers and normalization utilities.

2. src/alignment
- Perspective correction to normalize sheet pose.

3. src/detection
- YOLO wrapper and detected region parsing.

4. src/mapping
- Converts region outputs into structured answer map.

5. src/scoring
- Teacher versus student comparison and metric calculation.

6. src/visualization
- Overlay annotations, confidence text, status banners.

### 2.4 Scripts and Execution Interfaces

1. scripts/infer_sheet.py
- Interface: input image path, model weights path, output image path.
- Output: annotated image with detection summary.

2. scripts/evaluate.py
- Interface: currently sample comparison logic execution.
- Output: score summary and metrics printout.

3. app/streamlit_app.py
- Interface: browser UI for model path input and image upload.
- Output: aligned image, detections image, and detection count text.

### 2.5 Model Artifacts

1. Production checkpoint
- Expected at artifacts/yolo/best.pt
- Pre-trained YOLO model for inference

2. Runtime outputs
- Inference visual outputs in outputs directory

## 3. System Architecture in Detail

### 3.1 Architectural Pattern

The system uses a staged pipeline architecture with clear boundaries:
1. Input acquisition stage
2. Preprocessing and geometric normalization stage
3. AI-based layout detection stage
4. Region parsing and answer extraction stage
5. Answer mapping and scoring stage
6. Visualization and reporting stage

Each stage has one primary responsibility and clean input and output contracts.

### 3.2 End-to-End Data Flow

1. Input acquisition
- Source: uploaded image from UI or local file path.
- Validation: verify image can be decoded.

3. Alignment
- Perspective correction normalizes sheet orientation.
- Output is canonical aligned image for stable downstream detection.

4. Layout detection
- YOLO model detects block-level regions such as mcq_block, tfng_block, az_block, roman_block, completion_block.
- Output: list of bounding boxes with class label and confidence.

5. Region-level extraction
- For each detected box, OpenCV and deterministic rules extract filled options.
- Output: scope-level extracted tokens.

6. Answer map generation
- Converts extracted outputs into standardized key-value mapping by question id.

7. Scoring and comparison
- Compares student answer map to teacher answer map.
- Computes correct, wrong, unanswered, and percentage.

8. Visualization and report
- Draws boxes and labels on output image.
- Displays summary and review indicators in UI.

### 3.3 Internal Contracts Between Stages

1. Detection contract
- Input: aligned image matrix.
- Output per detection:
	- label as class string
	- confidence as float
	- x1, y1, x2, y2 pixel coordinates

2. Extraction contract
- Input: cropped region image plus expected scope type.
- Output: normalized answer token and confidence.

3. Scoring contract
- Input: teacher answer map and student answer map.
- Output:
	- total_questions
	- correct
	- wrong
	- unanswered
	- percentage

### 3.4 Why This Architecture Works

1. YOLO handles layout variation
- Sheet can change arrangement and still be localized.

2. OpenCV handles precision inside blocks
- Deterministic extraction is fast and explainable.

3. Separation of concerns
- Training, inference, scoring, and UI can evolve independently.

4. Easy debugging
- Intermediate outputs exist at each stage and can be visualized.

## 4. Training Architecture Versus Runtime Architecture

### 4.1 Model Training (Colab)

YOLO model training is performed in Google Colab:
1. Image collection and annotation done externally
2. Dataset preparation and data.yaml configuration
3. YOLO training and validation in Colab environment
4. Model checkpoint promotion to artifacts/yolo/best.pt

Note: Training setup and execution is handled in Colab; pre-trained model is expected locally.

### 4.2 Runtime Architecture

Purpose: process one sheet and return result.

Pipeline:
1. Load image
2. Align
3. Detect blocks
4. Parse answers
5. Compare and score
6. Visualize and return

## 5. Deployment and Operations Stack

### 5.1 Local Development

1. Infer with scripts/infer_sheet.py
2. Demo with app/streamlit_app.py
3. Score with scripts/evaluate.py

### 5.2 Production-Oriented Option

1. Keep model in versioned artifact store.
2. Expose inference via API layer.
3. Keep Streamlit as operator panel or internal QA console.

### 5.3 Monitoring and Quality Control

Track the following:
1. Detection confidence distribution
2. Per-scope extraction accuracy
3. Manual review rate
4. End-to-end processing time per sheet

Use threshold policy for automatic acceptance versus review.

## 6. Suggested Version Baseline

1. Python 3.10+
2. ultralytics 8.2+
3. opencv-python 4.9+
4. streamlit 1.34+
5. numpy 1.26+

## 7. Local Setup Checklist

1. Ensure pre-trained YOLO checkpoint exists at artifacts/yolo/best.pt.
2. Verify all dependencies are installed from requirements.txt.
3. Run single-image inference with scripts/infer_sheet.py and inspect output.
4. Run Streamlit UI with app/streamlit_app.py and test upload workflow.
5. Run scoring script with scripts/evaluate.py and verify outputs.
6. Set up logging and monitoring for inference results.

## 8. Architecture Risks and Mitigations

1. Risk: poor labels reduce detector quality
- Mitigation: strict labeling QA and periodic audit batches.

2. Risk: low-light or skewed capture failures
- Mitigation: increase such conditions in training set and improve alignment.

3. Risk: extraction ambiguity inside dense blocks
- Mitigation: confidence thresholds and manual review flags.

4. Risk: model drift when new sheet styles are introduced
- Mitigation: periodic model retraining in Colab with expanded dataset.

This architecture separates training (Colab) from inference (local), enabling practical deployment while maintaining model quality through centralized training workflows.
