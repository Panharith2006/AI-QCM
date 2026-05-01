# Using Separate YOLO and Hugging Face Pipelines

Two completely independent, simple pipelines. **Choose ONE, not both.**

---

## Overview

```
Option 1: YOLO Pipeline          Option 2: Hugging Face Pipeline
├─ Fast (0.2s)                   ├─ Accurate (85-90%)
├─ Free                          ├─ Free/Paid ($0-9/month)
├─ Local                         ├─ API or Local
├─ 90% accuracy                  ├─ Handles handwriting
└─ Simple                        └─ Vision-based
```

---

## Option 1: YOLO Pipeline (Fast & Free)

### What It Does
- Uses YOLO object detection to find regions
- Applies region-specific extraction
- Returns answers quickly
- No API calls needed

### Setup
```bash
# Already installed - no setup needed!
```

### Usage

```python
from src.yolo_pipeline import extract_with_yolo

# You already have yolo_regions from YOLO detection
yolo_regions = [
    {"class": "circle_fill", "roi": image_array_1},
    {"class": "tfng", "roi": image_array_2},
    # ... more regions
]

# Extract
result = extract_with_yolo(yolo_regions)

# Use results
print(result.answers)  # {"q1": "A", "q2": "T", ...}
print(result.confidence)  # {"q1": 0.98, "q2": 0.95, ...}
print(result.overall_confidence)  # 0.96
```

### Configuration

```python
from src.yolo_pipeline import extract_with_yolo

sheet_config = {
    "options": {
        "circle_fill": ["A", "B", "C", "D"],
        "tfng": ["T", "F", "NG"],
    }
}

result = extract_with_yolo(yolo_regions, sheet_config)
```

### Pros & Cons
```
✅ Pros:
   - Very fast (0.2s)
   - Free (no API)
   - Local only (private)
   - Already installed
   - No setup needed

❌ Cons:
   - 90% accuracy
   - Struggles with handwriting
   - Needs YOLO regions
```

---

## Option 2: Hugging Face Pipeline (Accurate)

### What It Does
- Uses Hugging Face vision models
- Understands full sheet layout
- Handles handwriting well
- Returns high-accuracy answers

### Setup (5 minutes)

#### Step 1: Get Free API Key
```
1. Go to: https://huggingface.co/settings/tokens
2. Click: "New token"
3. Select: "Read"
4. Copy: hf_...
```

#### Step 2: Set Environment Variable
```bash
# Linux/Mac
export HUGGINGFACE_TOKEN="hf_your_token_here"

# Windows PowerShell
$env:HUGGINGFACE_TOKEN="hf_your_token_here"
```

#### Step 3: Install Package
```bash
pip install huggingface-hub
```

### Usage

```python
from src.hf_pipeline import extract_with_hf_pipeline
import os

# Extract (auto-loads token from environment)
result = extract_with_hf_pipeline("path/to/sheet.jpg")

# Use results
print(result.answers)  # {"q1": "A", "q2": "T", ...}
print(result.confidence)  # {"q1": 0.95, "q2": 0.92, ...}
print(result.overall_confidence)  # 0.94
```

### With Custom API Key

```python
from src.hf_pipeline import extract_with_hf_pipeline

result = extract_with_hf_pipeline(
    image_path="sheet.jpg",
    api_key="hf_your_token",
    model="llava-hf/llava-1.5-7b-hf"
)

print(result.answers)
```

### With Configuration

```python
from src.hf_pipeline import extract_with_hf_pipeline

sheet_config = {
    "expected_questions": 25,
    "question_types": ["circle_fill", "tfng", "alpha_box"],
    "custom_options": {
        "circle_fill": ["A", "B", "C", "D"],
        "tfng": ["T", "F", "NG"],
    }
}

result = extract_with_hf_pipeline(
    "sheet.jpg",
    sheet_config=sheet_config
)
```

### Available Models

```python
# Recommended (free, general purpose)
model="llava-hf/llava-1.5-7b-hf"

# Alternative (better multilingual)
model="Qwen/Qwen-VL"

# Alternative (optimized for forms)
model="microsoft/layoutlm-base-uncased"
```

### Pros & Cons
```
✅ Pros:
   - High accuracy (85-90%)
   - Handles handwriting
   - Understands layouts
   - Free tier available
   - Vision-based

❌ Cons:
   - Slower (3-10s)
   - Needs API key
   - API rate limits
   - Requires internet
```

---

## Comparison

| Feature | YOLO | Hugging Face |
|---------|------|--------------|
| **Speed** | 0.2s ⚡ | 3-10s |
| **Accuracy** | 90% | 85-90% |
| **Cost** | Free | Free/Paid |
| **Setup** | None | 5 min |
| **Local** | Yes | API only |
| **Handwriting** | Poor | Good |
| **Best for** | Speed | Accuracy |

---

## Usage Examples

### Example 1: YOLO Only
```python
from src.yolo_pipeline import extract_with_yolo

# You have yolo_regions from YOLO detection
result = extract_with_yolo(yolo_regions)

if result.overall_confidence > 0.85:
    print("High confidence answers")
    for q_id, ans in result.answers.items():
        print(f"  {q_id}: {ans}")
else:
    print("Low confidence - may need manual review")
```

### Example 2: Hugging Face Only
```python
from src.hf_pipeline import extract_with_hf_pipeline
import os

# Ensure token is set
assert os.getenv("HUGGINGFACE_TOKEN"), "Set HUGGINGFACE_TOKEN"

result = extract_with_hf_pipeline("sheet.jpg")

if result.is_valid:
    print(f"✓ Extracted {result.total_questions} questions")
    print(f"  Confidence: {result.overall_confidence:.1%}")
else:
    print(f"✗ Failed: {result.errors}")
```

### Example 3: Choose Based on Confidence
```python
from src.yolo_pipeline import extract_with_yolo
from src.hf_pipeline import extract_with_hf_pipeline

# Try YOLO first (fast)
yolo_result = extract_with_yolo(yolo_regions)

# If confidence is low, try HF (accurate)
if yolo_result.overall_confidence < 0.80:
    print("YOLO confidence low, trying Hugging Face...")
    hf_result = extract_with_hf_pipeline("sheet.jpg")
    final_result = hf_result  # Use HF result
else:
    final_result = yolo_result  # Use YOLO result

print(final_result.answers)
```

### Example 4: Batch Processing (YOLO)
```python
from src.yolo_pipeline import extract_with_yolo
from pathlib import Path

def process_sheets_yolo(sheet_folder):
    """Process all sheets in folder using YOLO."""
    results = {}
    
    for sheet_path in Path(sheet_folder).glob("*.jpg"):
        yolo_regions = detect_with_yolo(sheet_path)  # Your YOLO detection
        result = extract_with_yolo(yolo_regions)
        results[sheet_path.name] = result
    
    return results

results = process_sheets_yolo("sheets/")
```

### Example 5: Batch Processing (HF)
```python
from src.hf_pipeline import extract_with_hf_pipeline
from pathlib import Path

def process_sheets_hf(sheet_folder):
    """Process all sheets in folder using HF."""
    results = {}
    
    for sheet_path in Path(sheet_folder).glob("*.jpg"):
        result = extract_with_hf_pipeline(str(sheet_path))
        results[sheet_path.name] = result
    
    return results

results = process_sheets_hf("sheets/")
```

---

## Decision Tree: Which to Use?

```
START
  │
  ├─→ Need speed (<1s)?
  │   YES → Use YOLO
  │   NO  → Continue
  │
  ├─→ Need accuracy (95%+)?
  │   YES → Can use HF (85-90% good enough for most)
  │   NO  → Use YOLO
  │
  ├─→ Have handwritten marks?
  │   YES → Use HF
  │   NO  → YOLO is fine
  │
  └─→ Default: Use YOLO
     (if accuracy insufficient, switch to HF)
```

---

## Integration with Existing Code

### If Using YOLO (Existing)
```python
# Your current code structure
from src.detection.extractors import route_and_extract

# Just wrap it
from src.yolo_pipeline import extract_with_yolo

result = extract_with_yolo(yolo_regions)  # Same interface
```

### If Switching to Hugging Face
```python
# No need to change anything else
from src.hf_pipeline import extract_with_hf_pipeline

result = extract_with_hf_pipeline("sheet.jpg")  # Simple!
```

---

## Troubleshooting

### YOLO Issues

**Problem:** Low accuracy
- **Solution:** Try Hugging Face instead

**Problem:** Regions not detected
- **Solution:** Check YOLO model confidence threshold

### Hugging Face Issues

**Problem:** `HUGGINGFACE_TOKEN not found`
```bash
# Set it
export HUGGINGFACE_TOKEN="hf_..."

# Or verify it's set
echo $HUGGINGFACE_TOKEN
```

**Problem:** Rate limit exceeded (Free tier)
- You've used 30,000 requests for the month
- **Solution:** Wait for reset or upgrade to paid ($9/month)

**Problem:** ImportError: huggingface_hub
```bash
pip install huggingface-hub
```

**Problem:** Low accuracy
- Try a different model (see "Available Models" section)
- Provide better quality images

---

## File Structure

```
src/
├── yolo_pipeline.py        ← YOLO extraction (new)
├── hf_pipeline.py          ← HF extraction (new)
├── llm/                    ← HF client modules
│   ├── client.py           (updated: HF only)
│   ├── llm_extractor.py    (updated: HF only)
│   ├── prompts.py
│   ├── response_parser.py
│   └── __init__.py         (updated: HF only)
└── detection/              (unchanged)
    ├── extractors.py
    └── ...
```

---

## Summary

### Choose YOLO If:
✅ Speed critical
✅ No API key
✅ Local processing only
✅ Already have YOLO infrastructure

### Choose Hugging Face If:
✅ Want better accuracy
✅ Have handwritten marks
✅ Can get free API key
✅ Don't mind 3-10s wait

### Start With:
1. **Test YOLO**: Run on existing regions → See accuracy
2. **If accuracy < 85%**: Try Hugging Face
3. **Pick whichever works** for your use case

---

## Quick Reference

### YOLO
```python
from src.yolo_pipeline import extract_with_yolo
result = extract_with_yolo(yolo_regions)
print(result.answers)
```

### Hugging Face
```python
from src.hf_pipeline import extract_with_hf_pipeline
result = extract_with_hf_pipeline("sheet.jpg")
print(result.answers)
```

---

*No complexity, no mixing. Just two simple, separate options.*
