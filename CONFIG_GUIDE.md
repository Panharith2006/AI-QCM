# Configuration Guide for Rule-Based Extractors

## Overview
All 4 scope extractors are now **fully configurable** to handle any variations on your exam papers.

---

## 1. MCQ Block (Multiple Choice Questions)

### Default
- Options: A, B, C, D

### Custom
```python
question_mapping = {
    "block_0": {
        "question_id": "Q1",
        "options": ["A", "B", "C", "D", "E"]  # 5 options instead of 4
    }
}
```

---

## 2. Roman Block (Roman Numeral Matching)

### Default
- Options: I to X (10 rows)

### Custom Examples
```python
# Example 1: Only 5 options (I-V)
question_mapping = {
    "block_1": {
        "question_id": "Q2",
        "num_options": 5  # I, II, III, IV, V
    }
}

# Example 2: 15 options (I-XV)
question_mapping = {
    "block_2": {
        "question_id": "Q3",
        "num_options": 15  # I through XV
    }
}

# Example 3: 20 options (I-XX)
question_mapping = {
    "block_3": {
        "question_id": "Q4",
        "num_options": 20  # I through XX
    }
}
```

**Supports up to 100 options (I to C)**

---

## 3. TFNG Block (True/False/No Given)

### Default
- Options: T, F, NG

### Custom
```python
# Example: Custom labels
question_mapping = {
    "block_4": {
        "question_id": "Q5",
        "options": ["Correct", "Wrong", "Skip"]  # Custom text
    }
}

# Example: Binary choice (Yes/No)
question_mapping = {
    "block_5": {
        "question_id": "Q6",
        "options": ["Yes", "No"]  # Only 2 options
    }
}
```

---

## 4. Completion Block

### Default
- Position-based: LEFT/RIGHT

### Custom
```python
# Currently position-based, can be extended for custom labels
question_mapping = {
    "block_6": {
        "question_id": "Q7"
    }
}
```

---

## Complete Example: Mixed Paper Type

```python
from src.pipeline import OMRPipeline

# Initialize
pipeline = OMRPipeline("artifacts/yolo/best.pt")

# Define your paper structure
question_mapping = {
    # MCQ with 5 options
    "block_0": {
        "question_id": "Q1",
        "options": ["A", "B", "C", "D", "E"]
    },
    # Roman matching with 7 options (I-VII)
    "block_1": {
        "question_id": "Q2",
        "num_options": 7
    },
    # TFNG
    "block_2": {
        "question_id": "Q3",
        "options": ["T", "F", "NG"]
    },
    # MCQ standard (4 options)
    "block_3": {
        "question_id": "Q4",
        "options": ["A", "B", "C", "D"]
    },
    # Roman with 15 options (I-XV)
    "block_4": {
        "question_id": "Q5",
        "num_options": 15
    },
}

# Process
result = pipeline.process_image("student_sheet.jpg", question_mapping=question_mapping)

# View results
print(result.answer_map)
# Output: {"Q1": "B", "Q2": "V", "Q3": "T", "Q4": "C", "Q5": "XII"}
```

---

## Key Points

✅ **MCQ Block**: Use `"options"` parameter (list of labels)
✅ **Roman Block**: Use `"num_options"` parameter (int, 1-100)
✅ **TFNG Block**: Use `"options"` parameter (list of custom labels)
✅ **Completion Block**: Position-based (no config needed)

✅ **No hardcoding**: Every extraction is configurable
✅ **Beginner-friendly**: Just specify what you have on your paper
✅ **Flexible**: Different papers can have different configurations

---

## Roman Numeral Reference

The system auto-generates Roman numerals:
- 1-10: I, II, III, IV, V, VI, VII, VIII, IX, X
- 11-20: XI, XII, XIII, XIV, XV, XVI, XVII, XVIII, XIX, XX
- 21-30: XXI, XXII, ..., XXX
- And so on up to 100 (C)

Just set `num_options` to the count you need!
