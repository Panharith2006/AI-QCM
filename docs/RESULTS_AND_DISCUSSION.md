# Results and Discussion

## 1. Background: Traditional bubble-detection methods
Traditional OMR and bubble-detection pipelines rely on classical image-processing and computer-vision techniques. Representative techniques include:

- Preprocessing: adaptive thresholding, histogram equalization, illumination correction, and morphological filtering to reduce noise and normalize mark intensities.
- Segmentation and candidate extraction: connected-component labeling, contour detection, and template matching to locate potential mark regions.
- Shape-based analysis: contour area, aspect ratio, circularity, and solidity filters to identify bubble-like regions; Hough Circle Transform is sometimes used for detecting circular marks.
- Projection/profiles and alignment: horizontal/vertical projection profiles and axis-aligned templates to find rows/columns of bubbles; geometric heuristics for skew and perspective correction.
- Heuristic mark scoring: comparing filled-area ratios or pixel intensity statistics against thresholds to decide marked vs. unmarked, often with hand-tuned thresholds per template.

Strengths of these approaches:
- Low compute and latency; can run fully offline on low-resource hardware.
- Transparent, interpretable heuristics that are easy to debug and tune for a single, well-defined template.

Limitations:
- Fragile to layout variation, diverse templates, printing/scanning artifacts, and non-standard marking styles (crosses, ticks, faint fills).
- Requires careful per-template calibration and complex pre-processing to be robust to illumination, rotation, or scanning noise.
- Hard to generalize: each new template usually needs adjustments or new templates in code.

Empirical studies in the literature typically report high accuracy when templates, scanning hardware, and mark styles are controlled, but performance drops on noisy or unseen templates.

---

## 2. Our approach: YOLO + Gemini pipeline
Overview:
- Layout detection: a YOLO model (ultralytics / similar) detects semantic regions in the exam sheet (bubble groups, circle-completion areas, tables, question blocks). YOLO replaces brittle template-matching and projection-based layout steps with a learned, object-detection stage that generalizes across diverse templates.
- Crop + localized analysis: detected regions are cropped and normalized (perspective correction when needed) before per-box analysis.
- Vision LLM (Gemini) for local interpretation: instead of fixed, threshold-based scoring, a generative vision model processes each crop and emits structured outputs (JSON) describing the selected option, confidence, or extracted answer. The LLM can infer context (e.g., handwritten labels, ambiguous marks) and produce human-readable explanations.

Advantages over classical CV:
- Robust layout generalization: YOLO learns the visual appearance of blocks across templates, reducing per-template engineering.
- Flexible mark interpretation: Gemini can handle a wider variety of mark styles (colored pens, ticks, partial fills) because it reasons about the whole crop rather than relying on simple pixel-area heuristics.
- Structured outputs and richer metadata: generative responses can include confidences, alternative hypotheses, and explanations that help downstream adjudication and human-in-the-loop verification.
- Faster development cycle: adding support for new sheet types often requires annotated bounding-box examples rather than re-implementing heuristics.

Costs and trade-offs:
- Dependence on large models and APIs: Gemini usage introduces operational complexity (latency, cost, and potential connectivity/reliability issues). We observed intermittent network/transport errors (e.g., remote connection closures) during generation calls which require robust retry and fallback handling.
- Non-determinism and hallucination risk: generative models can produce confident but incorrect outputs. Careful prompt engineering, structured-output enforcement (JSON schema), and validation checks are essential.
- Training data for YOLO: achieving high detection accuracy requires labeled bounding boxes across representative templates and mark styles.

---

## 3. How this affects accuracy and robustness
- Detection (YOLO): evaluate with mAP (mean Average Precision) and per-class precision/recall. Good detection reduces downstream cropping and interpretation errors. YOLO is robust to moderate skew/perspective but benefits from perspective normalization of crops.
- Interpretation (Gemini): treat as an OCR/labeling model with probabilistic outputs. Use per-box confidence thresholds and aggregate across questions for sheet-level decisions. Evaluate with:
  - Per-question accuracy (correct option predicted vs. ground truth)
  - Precision/recall/F1 for selected options (especially important for multi-select questions)
  - End-to-end sheet accuracy (percentage of sheets with all questions correct)

Hybrid decisions:
- Use Gemini's confidence plus heuristic signals (filled-area ratio from CV) to reduce false positives and detect hallucinations.
- When Gemini confidence is low or network calls fail, fall back to classical CV scoring for resilience and to avoid missing results.

---

## 4. Evaluation plan and metrics
Suggested evaluation steps:
1. Dataset: collect a diverse, labeled dataset of scanned/photographed sheets across templates, lighting, marking styles, and printers. Include edge cases: faint marks, erasures, multi-marks, and rotated sheets.
2. Detection metrics: report mAP@0.5 and per-class precision/recall for YOLO on a held-out detection test set.
3. Interpretation metrics: per-box classification accuracy, per-question F1, and confusion matrices comparing predicted vs. ground truth options.
4. End-to-end metrics: sheet-level accuracy and failure-mode analysis (detection error, crop quality problem, misinterpretation, hallucination, or connectivity failure).
5. Latency and cost: measure average per-sheet processing time and per-sheet API calls/costs to evaluate deployment feasibility.

---

## 5. Limitations, failure modes, and mitigations
Failure modes observed or expected:
- API/network failures: transient connection closures or timeouts from external LLM services can interrupt processing; implement retries, exponential backoff, and per-box timeouts plus local fallback.
- Hallucinations: Gemini may return incorrect structured outputs; enforce strict JSON schema extraction, validate against allowed options, and use a voting/ensemble or heuristic fallback.
- Mis-detections: missing or incorrect bounding boxes from YOLO lead to missing answers. Improve by augmenting training data, adding synthetic variations, and using non-max suppression tuning.
- Cost and latency: cloud LLM calls increase operational cost and slow batch processing. Mitigate via batching, caching common crops, or running a local/cheaper vision model for high-volume deployments.

