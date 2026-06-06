# YOLO + Gemini OCR Integration

Extract text from boxes detected by YOLO using Google Gemini API.

## Overview
**Workflow:**
1. **YOLO** detects boxes/regions in your exam sheet image
2. **Gemini Vision API** extracts text from each detected box
3. Returns text + confidence scores for each box

## Setup
### 1. Get Gemini API Key (FREE)

- Visit: https://ai.google.dev
- Click "Get API Key"
- Copy the key (free tier: 60 requests/min)

### 2. Set Environment Variable

```powershell
$env:GOOGLE_API_KEY = "your-api-key-here"
```

### 3. Install Dependencies

```powershell
pip install google-generativeai
```

## Quick Start

### Single Image Processing

```python
from src.OCR.gemini_ocr_extractor import GeminiOCRExtractor

# Initialize
extractor = GeminiOCRExtractor(
    yolo_model_path="artifacts/yolo/best.pt",
    debug=True
)

# Process image
result = extractor.extract(
    "exam.jpg",
    config={
        "answer_options": ["A", "B", "C", "D"],
    }
)

# View results
for qid, answer_data in result.answers.items():
    print(f"{qid}: {answer_data['answer']} (confidence: {answer_data['confidence']:.2%})")
```

### Run Example

```powershell
python example_yolo_gemini.py
```

## Configuration

```python
config = {
    # For multiple choice questions
    "answer_options": ["A", "B", "C", "D"],
    
    # For text-based questions (leave empty)
    # "answer_options": [],
    
    # Allowed characters in OCR
    "ocr_allowlist": "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
    
    # YOLO detection threshold
    "fill_threshold": 0.3,
}
```

## How It Works

### Step 1: YOLO Detection
```
Image → YOLO Model → Detect boxes → Get coordinates
```

Your trained YOLO model detects:
- Question boxes
- Text regions
- Answer boxes
- Any marked regions

### Step 2: Box Cropping
```
Full Image + Box Coordinates → Crop region → Single box image
```

For each detected box, the system crops that region from the image.

### Step 3: Gemini Text Extraction
```
Box Image → Gemini Vision API → Extract text
```

Gemini reads the text inside each cropped box:
- Handles printed text
- Handles handwritten marks
- Returns confidence score
- Maps to answer options if configured

### Step 4: Result Assembly
```
Text from all boxes → Structured answer dict → Return results
```

Results are returned in format:
```python
{
    "Q1": {"answer": "A", "confidence": 0.95, "type": "gemini_ocr"},
    "Q2": {"answer": "B", "confidence": 0.88, "type": "gemini_ocr"},
    ...
}
```

## Output Structure

```python
result.answers = {
    "Q1": {
        "answer": "A",              # Extracted answer
        "confidence": 0.95,         # 0.0 to 1.0
        "type": "gemini_ocr",      # Always "gemini_ocr"
        "raw": "A"                  # Raw text from Gemini
    },
    "Q2": {...},
    ...
}

result.overall_confidence = 0.91    # Average confidence
result.is_valid = True              # All boxes processed
result.errors = []                  # Any errors
result.debug_info = {
    "total_boxes": 50,
    "successful_extractions": 50
}
```

## API Limits (Free Tier)

- **Requests per minute:** 60
- **Requests per day:** 1,500
- **Input size:** Up to 4MB per image
- **No credit card required**
