import os
import json
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import csv

from dotenv import load_dotenv

import google.genai as genai
from PIL import Image

load_dotenv()


@dataclass
class OMRResult:
    answers: Dict[str, str] = None  # {'Q1': 'A', 'Q2': 'B', ...}
    questions_found: int = 0
    correct_count: int = 0
    incorrect_count: int = 0
    score_percentage: float = 0.0
    question_results: Dict[str, bool] = None  # {'Q1': True, 'Q2': False}
    confidence: float = 0.0
    needs_review: bool = False
    raw_response: str = ""
    
    def __post_init__(self):
        if self.answers is None:
            self.answers = {}
        if self.question_results is None:
            self.question_results = {}


class GeminiOMRChecker:
     
    # Prompt template for Gemini - instructs the AI how to extract OMR data
    OMR_EXTRACTION_PROMPT = """
You are an OMR (Optical Mark Recognition) sheet evaluator. Analyze the provided OMR sheet image and extract the following information.

**Instructions:**
1. Identify all question numbers and the bubble that is marked for each
2. If a question has multiple bubbles marked, select the DARKEST one
3. If no bubble is clearly marked for a question, leave it empty
4. Do NOT guess or hallucinate answers - only return what you can clearly see

**Output Format:**
Return ONLY valid JSON with this exact structure:
{
    "answers": {
        "1": "A",
        "2": "B",
        "3": "C"
    },
    "confidence": 0.95,
    "questions_found": 3
}

**Rules:**
- Question numbers should be used as keys (as strings)
- Answer choices are single letters: A, B, C, D, etc.
- Confidence should be between 0.0 and 1.0
- If you cannot find student details, use empty strings
- Return ONLY the JSON, no other text or markdown
"""
    
    def __init__(self, api_key: Optional[str] = None, model_name: str = "gemini-2.5-flash"):
    
        # Configure API key
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("Gemini API key required. Set GOOGLE_API_KEY or pass api_key parameter")
        
        self.client = genai.Client(api_key=self.api_key)
        self.model_name = model_name
        
        # Set default answer key (can be overridden)
        self.answer_key: Dict[str, str] = {}
        
        # Confidence threshold - below this, mark for review
        self.confidence_threshold: float = 0.7
    
    def set_answer_key(self, answer_key: Dict[str, str]) -> None:
       
        self.answer_key = answer_key
    
    def load_answer_key_from_csv(self, csv_path: str) -> None:
       
        answer_key = {}
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                answer_key[row['question'].strip()] = row['correct_answer'].strip().upper()
        self.set_answer_key(answer_key)
    
    def extract_from_image(self, image_path: str) -> OMRResult:
        result = OMRResult()
        
        try:
            image = Image.open(image_path)

            response = self._generate_content_with_retry([self.OMR_EXTRACTION_PROMPT, image])
            
            response_text = getattr(response, "text", "") or ""
            result.raw_response = response_text
            
            # Parse JSON from response (handles markdown code blocks)
            json_str = self._extract_json_from_response(response_text)
            
            if json_str:
                data = json.loads(json_str)
                result.answers = self._normalize_answers(data.get("answers", {}))
                result.questions_found = int(data.get("questions_found", len(result.answers) or 0) or 0)
                result.confidence = data.get("confidence", 0.5)
                
                # Check if confidence is below threshold
                if result.confidence < self.confidence_threshold:
                    result.needs_review = True
            
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
            result.needs_review = True
            result.raw_response = f"Error: {str(e)}"
        
        return result

    def _generate_content_with_retry(self, contents, max_retries: int = 4, base_delay: float = 2.0):
        last_error = None

        for attempt in range(max_retries + 1):
            try:
                return self.client.models.generate_content(
                    model=self.model_name,
                    contents=contents,
                )
            except Exception as exc:
                last_error = exc
                error_text = str(exc)
                is_retryable = (
                    "503" in error_text
                    or "UNAVAILABLE" in error_text
                    or "429" in error_text
                    or "RESOURCE_EXHAUSTED" in error_text
                )

                if not is_retryable or attempt >= max_retries:
                    raise

                delay = base_delay * (2 ** attempt)
                time.sleep(delay)

        if last_error is not None:
            raise last_error
    
    def evaluate_result(self, result: OMRResult) -> OMRResult:
        
        if not self.answer_key:
            raise ValueError("Answer key not set. Call set_answer_key() first.")
        
        correct = 0
        total = len(self.answer_key)
        
        for q_num, correct_answer in self.answer_key.items():
            student_answer = result.answers.get(str(q_num), "").strip().upper()
            is_correct = (student_answer == correct_answer)
            result.question_results[str(q_num)] = is_correct
            
            if is_correct:
                correct += 1
        
        result.correct_count = correct
        result.incorrect_count = total - correct
        result.score_percentage = (correct / total * 100) if total > 0 else 0
        
        return result
    
    def process_sheet(self, image_path: str) -> OMRResult:
        
        result = self.extract_from_image(image_path)
        result = self.evaluate_result(result)
        return result
    
    def process_batch(self, image_dir: str, output_csv: str = "omr_results.csv") -> List[OMRResult]:
        
        results = []
        image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}
        
        # Find all images in directory
        image_paths = [
            p for p in Path(image_dir).iterdir() 
            if p.suffix.lower() in image_extensions
        ]
        
        print(f"Found {len(image_paths)} images to process")
        
        for idx, img_path in enumerate(image_paths, 1):
            print(f"Processing {idx}/{len(image_paths)}: {img_path.name}")
            result = self.process_sheet(str(img_path))
            result.student_name = result.student_name or img_path.stem  # Fallback to filename
            results.append(result)
        
        # Save results to CSV
        self._save_results_to_csv(results, output_csv)
        
        return results
    
    def _get_mime_type(self, file_path: str) -> str:
        """Get MIME type based on file extension"""
        ext = Path(file_path).suffix.lower()
        mime_map = {
            '.png': 'image/png',
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.bmp': 'image/bmp',
            '.tiff': 'image/tiff'
        }
        return mime_map.get(ext, 'image/png')
    
    def _extract_json_from_response(self, response_text: str) -> Optional[str]:
        cleaned = response_text.strip()

        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
            cleaned = re.sub(r"\s*```$", "", cleaned)

        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            return cleaned[start : end + 1]

        return None

    def _normalize_answers(self, answers: Dict[str, str]) -> Dict[str, str]:
        if not isinstance(answers, dict):
            return {}

        normalized: Dict[str, str] = {}
        for key, value in answers.items():
            normalized[str(key).strip()] = str(value).strip().upper() if value is not None else ""
        return normalized
    
    def _save_results_to_csv(self, results: List[OMRResult], output_path: str) -> None:
        """Save all results to a CSV file"""
        if not results:
            print("No results to save")
            return
        
        # Prepare rows
        rows = []
        for r in results:
            # Convert question_results to columns
            q_results = {f"Q{q}_correct": "Yes" if correct else "No" 
                        for q, correct in r.question_results.items()}
            
            row = {
            
                "Correct": r.correct_count,
                "Incorrect": r.incorrect_count,
                "Score (%)": round(r.score_percentage, 2),
                "Confidence": r.confidence,
                "Needs Review": "Yes" if r.needs_review else "No",
                "Raw Answers": json.dumps(r.answers) if r.answers else ""
            }
            row.update(q_results)
            rows.append(row)
        
        # Write to CSV
        if rows:
            with open(output_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
        
        print(f"Results saved to {output_path}")


# ============================================
# Example usage and CLI entry point
# ============================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate OMR sheets using Gemini Vision AI")
    parser.add_argument("-i", "--input", required=True, help="Input image file or directory")
    parser.add_argument("-o", "--output", default="omr_results.csv", help="Output CSV file")
    parser.add_argument("-k", "--key", help="CSV file with answer key (columns: question, correct_answer)")
    parser.add_argument("--key-value", help="Answer key as JSON string, e.g., '{\"1\":\"A\",\"2\":\"B\"}'")
    parser.add_argument("--confidence-threshold", type=float, default=0.7, help="Confidence threshold (0-1)")
    parser.add_argument("--api-key", help="Gemini API key (or set GOOGLE_API_KEY env var)")
    
    args = parser.parse_args()
    
    # Initialize checker
    checker = GeminiOMRChecker(api_key=args.api_key)
    
    # Set confidence threshold
    checker.confidence_threshold = args.confidence_threshold
    
    # Set answer key
    if args.key:
        checker.load_answer_key_from_csv(args.key)
        print(f"Loaded answer key from {args.key}")
    elif args.key_value:
        answer_key = json.loads(args.key_value)
        checker.set_answer_key(answer_key)
        print("Using provided answer key")
    else:
        print("WARNING: No answer key provided. Results will only show extracted answers.")
    
    # Determine if input is file or directory
    input_path = Path(args.input)
    
    if input_path.is_file():
        # Process single file
        print(f"Processing single file: {input_path.name}")
        result = checker.process_sheet(str(input_path))
        
        # Print results
        print(f"\n=== Results for {input_path.name} ===")
        print(f"Extracted Answers: {result.answers}")
        
        if checker.answer_key:
            print(f"Correct: {result.correct_count}/{len(checker.answer_key)}")
            print(f"Score: {result.score_percentage:.1f}%")
        
        print(f"AI Confidence: {result.confidence:.2f}")
        print(f"Needs Review: {'Yes' if result.needs_review else 'No'}")
        
        # Save single result
        checker._save_results_to_csv([result], args.output)
        
    elif input_path.is_dir():
        # Process batch
        print(f"Processing directory: {input_path}")
        results = checker.process_batch(str(input_path), args.output)
        
        # Print summary
        print(f"\n=== Processing Complete ===")
        print(f"Processed: {len(results)} sheets")
        print(f"Results saved to: {args.output}")
        
        needs_review = [r for r in results if r.needs_review]
        if needs_review:
            print(f"⚠️  {len(needs_review)} sheets need manual review (low confidence)")
        
        if checker.answer_key:
            avg_score = sum(r.score_percentage for r in results) / len(results) if results else 0
            print(f"Average Score: {avg_score:.1f}%")
    
    else:
        print(f"Error: Input path not found: {args.input}")


if __name__ == "__main__":
    main()