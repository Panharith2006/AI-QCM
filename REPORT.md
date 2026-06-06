# AI Project — OMR & Vision‑LLM Pipeline (Detailed Research Report)

This document is a detailed, research‑style report for the AI OMR / answer‑extraction project contained in this repository. Per your request, all analysis and references in this report explicitly exclude the `OMRChecker` folder.

## Executive Summary

This project integrates learned layout detection (YOLO) with a vision LLM (Google Gemini) to build a robust OMR / answer extraction pipeline that works on photographed and scanned answer sheets. The pipeline: (1) localizes question/answer regions using a YOLO-based layout model, (2) aligns and crops detected regions, (3) calls Gemini Vision for localized interpretation and OCR, (4) post-processes results and compares to teacher references, and (5) surfaces results in a Streamlit UI with explanations and review controls. The approach trades off operational cost and latency for broader robustness to template and mark variability, and incorporates explainability tools for auditability.

## 1. Introduction

1.1 Background

Automated grading systems reduce cost and turnaround time for mass assessments. Traditional OMR relies on strict templates and scanner input; recent advances in deep learning allow robust detection from mobile photos and unconstrained templates. Vision LLMs (Gemini, etc.) enable interpretation of complex visual crops (handwritten marks, partial fills) that heuristic area‑based scorers struggle with.

1.2 Problem Statement

Design an automated pipeline that accepts photographic or scanned answer sheets and extracts student responses reliably across diverse templates and marking styles while providing interpretable outputs for audit and review.

1.3 Objectives

- Build a YOLO‑based layout detector to find semantic regions (answer rows, bubbles, metadata).
- Use a vision LLM to interpret cropped regions and return structured answers with confidence.
- Provide alignment and preprocessing to reduce geometric variance.
- Deliver a UI for teachers/students (Streamlit) that supports reference creation, single/batch processing, and human review for low‑confidence sheets.

1.4 Significance

The system enables smartphone-based submissions, reduces manual grading load, and improves resilience to non-standard marks. Explainability supports trust and auditability in high-stakes contexts.

## 2. Related Work (Literature Review)

2.1 Classical OMR and document analysis

Early OMR systems use fixed templates, thresholding, morphological filters, and geometric heuristics (projection profiles, Hough transforms). These systems are efficient but brittle to rotation, lighting, and template changes.

2.2 Learned Layout & Detection

Single‑stage detectors like YOLO provide fast bounding‑box detection and generalize across templates when trained on representative bounding box labels. Many modern document analysis pipelines separate layout detection (object detection) from per‑region recognition (OCR).

2.3 Vision LLMs and Hybrid Pipelines

Vision LLMs combine visual features and generative reasoning to produce structured outputs from image crops. Recent research explores combining learned detectors with LLM-backed recognition for robust, interpretable extraction, balancing heuristic fallback for reliability.

2.4 Explainable AI (XAI)

XAI methods such as Grad‑CAM, integrated gradients, and example‑based explanation help inspect model decision paths. For OMR pipelines, XAI can highlight which pixels or crop regions influenced a decision, assisting manual review and model debugging.

## 3. Methodology

3.1 Research Design

This is an applied engineering research project employing iterative development and empirical evaluation. The methodology includes system design, implementation, and a defined evaluation protocol for detection and interpretation stages.

3.2 Data & Dataset Curation (How to prepare for experiments)

- Collect representative scanned and photographed sheets across templates, lighting conditions, pen/marker types, and student handwriting styles.
- Label bounding boxes for layout detection (YOLO) and provide per‑question ground truth answers for evaluation.
- Partition: train/val/test splits ensuring template and photographer diversity in each split.

3.3 Preprocessing & Alignment

- Resize and perspective‑correct images to canonical dimensions using `src/alignment/perspective.py` to stabilize model inputs.
- Optional denoising and adaptive thresholding for heuristic fallbacks.

3.4 Layout Detection (YOLO)

- Implementation: `src/detection/yolo_layout.py` — `YoloLayoutDetector` wraps Ultralytics YOLO inference.
- Training recommendations: fine-tune a YOLOv5/YOLOv8 model on annotated boxes (classes: question_box, answer_row, bubble_group, marker).
- Inference: detect boxes on original image (detector.detect(image, conf=0.25)). Use non‑max suppression and size filters to remove spurious boxes.

3.5 Per‑Box Interpretation / OCR (Gemini)

- Implementation: `src/OCR/gemini_ocr_extractor.py` and `src/llm/gemini_processor.py`.
- Workflow: crop detection → convert to PIL image → call Gemini with a structured prompt (schema enforced: return JSON or fixed format) → parse results.
- Prompt engineering: use explicit schema, examples, and rules to reduce hallucination. Enforce JSON extraction and parse with robust regex to handle code fences.
- Fallbacks: if Gemini confidence is low or API fails, compute a heuristic fill ratio (CV) on the crop and apply a threshold; mark for manual review when ambiguous.

3.6 Box Visual Interpretation (XAI & Gemini‑based explanation)

- Implementation: `src/box_interpretation/content_interpreter.py` uses Gemini to return a plain‑text structured interpretation and recommendations for manual review.
- For model explanations on CNN/CV components (if used), apply Grad‑CAM on classification heads; for LLM outputs, provide the raw response and structured confidence.

3.7 Post‑processing & Mapping

- Parse Gemini outputs into canonical answer tokens, normalize via `src/OCR/text_normalizer.py`, then map to teacher reference grid via `app/answer_processing.py` and `app/reference_manager.py`.

3.8 Evaluation Protocol

- Detection metrics: mAP@0.5, per‑class precision/recall, IoU distribution for bounding boxes.
- Interpretation metrics: per‑box accuracy (predicted choice vs ground truth), confusion matrices, F1 for multi‑label cases.
- End‑to‑end metrics: per‑question accuracy, per‑sheet accuracy (exact match), average score error.
- Operational metrics: average inference latency per sheet, API calls per sheet, and per‑sheet estimated cost.

## 4. System Architecture (Flowchart)

The high‑level pipeline is represented below. The diagram is included as a Mermaid flowchart (renderable by supporting Markdown tools).

```mermaid
flowchart TD
    A[Input Image (photo or scan)] --> B[Preprocessing]
    B --> C{Is document contour found?}
    C -- Yes --> D[Perspective Correction]
    C -- No  --> D
    D --> E[YOLO Layout Detector]
    E --> F{Detected Regions}
    F -->|Question Boxes| G[Crop & Normalize]
    F -->|Metadata / Roll No etc| H[Metadata OCR]
    G --> I[Gemini Vision: Per-Box Extraction]
    I --> J[Parse & Normalize Answers]
    J --> K[Compare to Teacher Reference]
    K --> L[Results & Scores]
    I --> M[Box Interpretation (explanations)]
    L --> N[Streamlit UI / CSV / JSON]
    M --> N
    subgraph Fallbacks
      I -->|Low confidence| O[CV Heuristic Fallback]
      O --> P[Mark for Manual Review]
      I -->|API error| P
    end

```

## 5. Implementation Details & File Map

- `app/streamlit_app.py` — Main UI orchestration (teacher/student/bubble detector). [file](app/streamlit_app.py#L1)
- `app/gemini_omr.py` — CLI/utility wrapper for Gemini-based OMR evaluation and CSV output. [file](app/gemini_omr.py#L1)
- `app/answer_processing.py` — Parsing and comparison utilities. [file](app/answer_processing.py#L1)
- `src/detection/yolo_layout.py` — YOLO detector wrapper and crop utility. [file](src/detection/yolo_layout.py#L1)
- `src/alignment/perspective.py` — Perspective correction utilities. [file](src/alignment/perspective.py#L1)
- `src/OCR/gemini_ocr_extractor.py` — Per‑crop extraction pipeline using `GeminiProcessor`. [file](src/OCR/gemini_ocr_extractor.py#L1)
- `src/llm/gemini_processor.py` — Gemini client wrapper and helper functions. [file](src/llm/gemini_processor.py#L1)
- `src/box_interpretation/content_interpreter.py` — Visual interpretation and recommendations. [file](src/box_interpretation/content_interpreter.py#L1)
- `src/pipeline.py` — `PipelineResult` dataclass for structured outputs. [file](src/pipeline.py#L1)

## 6. Explainability (XAI) Strategy

- For CNN or classifier components (if added): use Grad‑CAM to produce heatmaps for crop classification heads.
- For Gemini outputs: enforce structured JSON outputs, and attach raw responses and confidence. Use `BoxContentInterpreter` outputs as human‑readable explanations and include recommendations when confidence < threshold.
- Log all raw Gemini responses to enable offline audit and retraining of failure cases.

## 7. Experimental Plan and Reproducibility

7.1 Training YOLO (recommended)

- Annotate ~1k–10k bounding boxes across templates; more diverse templates reduce overfitting.
- Use Ultralytics training pipeline: typical hyperparameters — 300 epochs, batch size 16 (adjust for GPU), image size 640 or 1280 depending on GPU memory; early stopping on validation mAP.

7.2 Evaluation

- Run detection evaluation (mAP) on held‑out test set.
- Run end‑to‑end extraction on a separate answer‑key labeled set and compute per‑question and per‑sheet accuracy.

7.3 Ablation & Robustness tests

- Lighting variations: synthetic augmentation (brightness/contrast, blur).
- Mark style variations: pens, pencils, crosses, ticks, partial fills.
- Geometric perturbations: rotate/scale/crop to test alignment robustness.

## 8. Deployment & Operational Considerations

- Latency: Gemini calls per box add significant latency; mitigate by batching crops where possible or using a local OCR model for low-cost bulk processing.
- Cost: track API usage and implement caching for repeated crops.
- Privacy: avoid sending PII to third‑party APIs; for privacy‑sensitive deployments consider local models.
- Reliability: implement retries with exponential backoff (already present in `GeminiProcessor`) and local fallback heuristics for outages.

## 9. Limitations & Ethical Considerations

- Hallucination risk from generative outputs — enforce schema, validate against allowed options, and flag low‑confidence outputs.
- Bias: ensure representative data across demographics and handwriting styles to avoid systematic misgrading.

## 10. Future Work

- Integrate a lightweight local vision model (on‑device) to reduce API dependence.
- Use uncertain‑case active learning: present low‑confidence crops to annotators and retrain models iteratively.
- Add automatic template discovery and self‑supervised layout adaptation.

## 11. How to Run & Reproduce (practical)

1. Install dependencies (See `requirements.txt`).

```bash
python -m pip install -r requirements.txt
```

2. Set Gemini API key in `.env` or environment variable `GOOGLE_API_KEY`.

3. Start Streamlit UI:

```bash
streamlit run app/streamlit_app.py
```

4. Quick CLI: use `app/gemini_omr.py` or `src/llm/gemini_processor.py` wrappers for batch processing. Example:

```bash
python app/gemini_omr.py -i path/to/image.jpg --api-key $GOOGLE_API_KEY
```

## 12. References (select)

- Redmon, J. et al. "You Only Look Once: Unified, Real-Time Object Detection" (YOLO) — foundational single‑stage detector.
- Ultralytics YOLO documentation and model cards (practical implementation used here).
- Selvaraju, R. R., et al. "Grad‑CAM: Visual Explanations from Deep Networks via Gradient‑based Localization".
- Samek, W., et al. — Surveys on Explainable AI methods for vision models.
- Google Gemini / Google Generative AI documentation (API usage and guidance).

---

This report was generated from the code and documentation present in the repository (excluded `OMRChecker`). If you want a PDF export, I can add a small conversion script and produce `REPORT.pdf` in the workspace.

