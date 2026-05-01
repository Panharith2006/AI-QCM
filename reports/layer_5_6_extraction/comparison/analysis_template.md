# Layer 5 & 6 - Extraction Analysis

## Your Custom Analysis

### Extraction Accuracy by Type

| Question Type | Tested | Correct | Error Rate | Comments |
|---------------|--------|---------|-----------|----------|
| MCQ (4 opt) | __ | __ | __% | |
| MCQ (5 opt) | __ | __ | __% | |
| Roman | __ | __ | __% | |
| TFNG | __ | __ | __% | |
| Completion | __ | __ | __% | |

### Fill Ratio Statistics

**Current Thresholds:**
```python
mcq_min_threshold = 0.12
mcq_ambiguity_margin = 0.03
roman_min_threshold = 0.10
tfng_min_threshold = 0.08
```

**Distribution of fill ratios:**
- Actual marks (marked as correct answer): __%
  - Min: __, Max: __, Mean: __
- Unmarked bubbles: __%
  - Min: __, Max: __, Mean: __
- Partial marks / erased attempts: __%
  - Min: __, Max: __, Mean: __

### Confidence Scores

**Current: Differential confidence = top - second_highest**

- Correctly answered questions: avg confidence __
- Incorrectly answered: avg confidence __
- Ambiguous marks: avg confidence __
- Unanswered: avg confidence __

### Problem Cases

**Double-marked bubbles:**
- Frequency: __ %
- Detected correctly as AMBIGUOUS: __ %
- False negatives: __ %

**Heavily erased answers:**
- How many remnants remain: ___%
- Detection rate: __ %

**Mark quality variations:**
- Light pencil marks: accuracy __% (threshold: __)
- Pen marks: accuracy __% (threshold: __)
- Heavy/dark marks: accuracy __% (threshold: __)

### Algorithm Improvements

Could we improve extraction by:
- [ ] Morphological operations (dilation/erosion)
- [ ] Connected component analysis
- [ ] Machine learning on bubble regions
- [ ] Better threshold calculation

### Processing Time

- MCQ extraction: __ ms per block
- Roman extraction: __ ms per block
- TFNG extraction: __ ms per block
- Completion extraction: __ ms per block

### Recommendation
_Your final recommendation for this layer_
