"""
Example: Process student sheet and compare against teacher answer key.
"""

from pathlib import Path
import cv2
from src.pipeline import OMRPipeline
from src.visualization.annotate import draw_detection_boxes, draw_score_report


def main():
    # Configuration
    MODEL_PATH = "artifacts/yolo/best.pt"
    STUDENT_SHEET = "student_sheet.jpg"
    TEACHER_ANSWERS = {
        "Q1": "B",
        "Q2": "A",
        "Q3": "VII",  # Roman numeral answer
        "Q4": "T",
        "Q5": "C",
    }
    OUTPUT_DIR = Path("outputs")
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # Initialize pipeline
    pipeline = OMRPipeline(MODEL_PATH)
    pipeline.enable_debug()
    
    # Question mapping with flexible configuration
    # Shows different question types and option counts
    question_mapping = {
        "block_0": {"question_id": "Q1", "options": ["A", "B", "C", "D"]},
        "block_1": {"question_id": "Q2", "options": ["A", "B", "C", "D", "E"]},  # 5 options
        "block_2": {"question_id": "Q3", "num_options": 15},  # Roman I-XV
        "block_3": {"question_id": "Q4", "options": ["T", "F", "NG"]},
        "block_4": {"question_id": "Q5", "options": ["A", "B", "C", "D"]},
    }
    
    # Process and compare
    print(f"Processing student sheet: {STUDENT_SHEET}")
    comparison = pipeline.process_sheet_comparison(
        STUDENT_SHEET,
        TEACHER_ANSWERS,
        question_mapping=question_mapping,
    )
    
    # Display results
    print("\n" + "="*50)
    print("COMPARISON RESULTS")
    print("="*50)
    
    metrics = comparison["metrics"]
    print(f"\nOverall Score: {metrics['percentage']:.1f}%")
    print(f"Correct: {metrics['correct']}/{metrics['total_questions']}")
    print(f"Wrong: {metrics['wrong']}")
    print(f"Unanswered: {metrics['unanswered']}")
    
    print("\nDetailed Results:")
    for qid, detail in metrics["details"].items():
        status = "✓" if detail["correct"] else "✗"
        print(f"{status} {qid}: Expected {detail['expected']}, Got {detail['student']}")
    
    # Save annotated images
    image_with_detections = draw_detection_boxes(comparison["image_detections"], comparison["debug"].get("step4_detections_count", []))
    image_with_score = draw_score_report(image_with_detections, metrics)
    
    cv2.imwrite(str(OUTPUT_DIR / "result_aligned.jpg"), comparison["image_aligned"])
    cv2.imwrite(str(OUTPUT_DIR / "result_detections.jpg"), comparison["image_detections"])
    cv2.imwrite(str(OUTPUT_DIR / "result_scored.jpg"), image_with_score)
    
    print(f"\nOutputs saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
