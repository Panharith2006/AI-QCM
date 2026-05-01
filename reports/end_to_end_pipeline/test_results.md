# Test Results Summary

## Overview

**Date Tested:** __________  
**Test Set Size:** __ sheets  
**Test Environment:** CPU / GPU  
**Python Version:** __  

---

## Overall Performance

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| Overall Accuracy | __% | 95%+ | ✓/✗ |
| Processing Time | __ ms | < 900 ms | ✓/✗ |
| MCQ Accuracy | __% | 96%+ | ✓/✗ |
| Roman Accuracy | __% | 94%+ | ✓/✗ |
| TFNG Accuracy | __% | 97%+ | ✓/✗ |
| Completion Accuracy | __% | 82%+ | ✓/✗ |

---

## Per-Layer Results

### Layer 1: Input Acquisition
- Average time: __ ms
- Failures: __ / __ sheets
- Error rate: __%

### Layer 2: Preprocessing
- CLAHE time: __ ms
- Normalization effective: Yes / No
- Issues: _none_ / _..._

### Layer 3: Alignment
- Successful alignment: __ / __ sheets
- Fallback count: __
- Average time: __ ms
- Max angle handled: __ degrees

### Layer 4: YOLO Detection
- Average detections per sheet: __
- Detection rate: __%
- False positive rate: __%
- Average inference time: __ ms
- Confidence distribution: ___

### Layer 5 & 6: Extraction
- MCQ accuracy: __%
- Roman accuracy: __%
- TFNG accuracy: __%
- Completion accuracy: __%
- Average extraction time: __ ms

### Layer 7: Mapping
- Correct answer mapping: 100%
- Null handling: Correct / Incorrect
- Time: __ ms

---

## Error Analysis

### False Positives
_Mark detected when there shouldn't be one:_
- MCQ: __ cases
- Roman: __ cases
- TFNG: __ cases
- Cause: _..._

### False Negatives
_Mark not detected when there should be:_
- MCQ: __ cases (light marks, partial erasure, etc.)
- Roman: __ cases
- TFNG: __ cases
- Cause: _..._

### Confidence Score Distribution

**Correctly answered:**
```
Confidence | Count
0.0 - 0.2  | __
0.2 - 0.4  | __
0.4 - 0.6  | __
0.6 - 0.8  | __
0.8 - 1.0  | __
```

**Incorrectly answered:**
```
Confidence | Count
0.0 - 0.2  | __
0.2 - 0.4  | __
0.4 - 0.6  | __
0.6 - 0.8  | __
0.8 - 1.0  | __
```

---

## Edge Cases Encountered

### Case 1: Double-Marked Bubbles
- Frequency: __%
- Detection: __ / __ correctly flagged as AMBIGUOUS
- Comment: _..._

### Case 2: Heavily Creased Sheets
- Count: __
- Accuracy impact: __%
- Resolution: _..._

### Case 3: Poor Image Quality
- Count: __
- Accuracy impact: __%
- Resolution: _..._

### Case 4: Unusual Mark Types
- Pen marks: accuracy __%, count __
- Light pencil: accuracy __%, count __
- Heavy marks: accuracy __%, count __

---

## Confidence-Based Filtering

**Low-confidence threshold: 0.5**

- Questions below threshold: __ / __
- Actually incorrect: __ / __
- Actually correct: __ / __
- Precision of low-confidence flag: __%

**Recommendation for human review threshold:** __

---

## Scoring Accuracy

**Test Teacher Key: __ questions**

| Marking Scheme | Sheets Tested | Accuracy |
|----------------|---------------|----------|
| All-or-nothing | __ | __% |
| Partial credit | __ | __% |
| Negative marking | __ | __% |

---

## Performance Scaling

**Single Sheet:** __ ms  
**Batch of 10:** __ ms avg  
**Batch of 100:** __ ms avg  

Scaling: _Linear / Sublinear / Superlinear_

---

## Comparison: Before vs. After Fixes

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Overall Accuracy | __% | __% | +__% |
| Processing Time | __ ms | __ ms | __% faster |
| Confidence Scoring | Simple | Differential | Better |
| Answer Structure | Flat | Nested | Richer |

---

## Recommendations

### Immediate Actions
1. _..._
2. _..._
3. _..._

### Future Improvements
1. _..._
2. _..._
3. _..._

### Parameter Tuning Opportunities
- CLAHE clip_limit: Tested __ values
- Canny thresholds: Tested __ combinations
- Fill ratio thresholds: Tested __ values

### Conclusion
_Overall assessment and summary_
