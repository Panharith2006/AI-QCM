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

## Cost Estimation

### Free Tier (Default)
- Up to 1,500 requests/day
- No cost

### Paid Tier (Optional)
- $0.00035 per 1K input tokens
- $0.0007 per 1K output tokens
- Vision same pricing as text

### Example Costs
- 100 exam sheets with 50 questions: ~$0.01/day
- 1000 sheets: ~$0.10/day

## Advantages vs EasyOCR

| Feature | EasyOCR | Gemini |
|---------|---------|--------|
| Text extraction | Good | ✅ Better |
| Handwriting | Limited | ✅ Better |
| Language support | Limited | ✅ All languages |
| Multiple answers | Manual | ✅ Automatic |
| API calls needed | No | Yes |
| Speed | Fast | Slower |
| Accuracy | ~85% | ~95% |
| Cost | Free (local) | ~$0.00001 per box |

## When to Use

### Use Gemini OCR when:
- ✅ High accuracy needed
- ✅ Complex layouts
- ✅ Handwritten marks
- ✅ Multiple languages
- ✅ Need semantic understanding

### Use EasyOCR when:
- ✅ Speed is critical
- ✅ No internet connection
- ✅ Running locally without APIs
- ✅ Large batch processing

## Troubleshooting

### "GOOGLE_API_KEY not set"
```powershell
$env:GOOGLE_API_KEY = "your-key"
```

### "API Error: 429 Rate Limit"
- Free tier: 60 requests/min
- Space requests by 1 second
- Or upgrade to paid plan

### "API Error: 403 Invalid Key"
- Check key at https://ai.google.dev
- Key may be expired
- Generate new key

### No boxes detected
- Check YOLO model path
- Verify image format
- Increase/decrease detection threshold

### Low confidence scores
- Image quality issues
- Adjust YOLO threshold
- Try different Gemini model (gemini-pro vs gemini-pro-vision)

## Integration with Pipeline

### Replace EasyOCR
```python
# Old: EasyOCR-based extraction
from src.OCR.ocr_extractor import OCRExtractor

# New: Gemini-based extraction
from src.OCR.gemini_ocr_extractor import GeminiOCRExtractor

# Drop-in replacement
extractor = GeminiOCRExtractor(yolo_model_path="artifacts/yolo/best.pt")
result = extractor.extract("image.jpg", config)
```

### Use in OMR Pipeline
```python
from src.pipeline import OMRPipeline
from src.OCR.gemini_ocr_extractor import GeminiOCRExtractor

# Current pipeline uses EasyOCR
# To use Gemini, replace OCRExtractor with GeminiOCRExtractor
```

## Advanced Usage

### Custom Prompt for Different Question Types

**Multiple Choice:**
```python
prompt = "Extract the selected answer option (A, B, C, or D)"
```

**Short Answer:**
```python
prompt = "Extract the written answer text"
```

**Yes/No:**
```python
prompt = "Is this marked? Return: Yes or No"
```

### Batch Processing

```python
from pathlib import Path

images = Path("raw_dataset/").glob("*.jpg")

for image_path in images:
    result = extractor.extract(str(image_path), config)
    print(f"{image_path.name}: {result.overall_confidence:.2%}")
```

### Error Handling

```python
result = extractor.extract(image_path, config)

if not result.is_valid:
    print("Errors:", result.errors)
    for qid, answer_data in result.answers.items():
        if answer_data.get("confidence", 0) < 0.7:
            print(f"Low confidence for {qid}: {answer_data}")
```

## Files

- **src/OCR/gemini_ocr_extractor.py** - Main extractor class
- **example_yolo_gemini.py** - Working example
- **setup_gemini.py** - Setup verification

## Next Steps

1. ✅ Set API key: `$env:GOOGLE_API_KEY = "your-key"`
2. ✅ Test setup: `python setup_gemini.py`
3. ✅ Run example: `python example_yolo_gemini.py`
4. ✅ Integrate into your pipeline
5. ✅ Tune for your exam sheet format

## Support

- Gemini Docs: https://ai.google.dev/docs
- GitHub: https://github.com/google/generative-ai-python
- Free tier perfect for testing!

---

Last updated: May 2026
