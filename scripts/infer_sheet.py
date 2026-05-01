from __future__ import annotations

import argparse
from pathlib import Path
import sys
import json

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipeline import OMRPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run OCR-based OMR inference on a single sheet image")
    parser.add_argument("--image", required=True, help="Path to input sheet image")
    parser.add_argument("--yolo", default="artifacts/yolo/best.pt", help="Path to YOLO model")
    parser.add_argument("--out", default="outputs/infer_result.json", help="Output path for extracted answers JSON")
    args = parser.parse_args()

    # Initialize pipeline
    pipeline = OMRPipeline(args.yolo)
    
    # Process image
    print(f"Processing: {args.image}")
    result = pipeline.process_image(args.image)
    
    # Print results
    print(f"\nExtracted answers:")
    for qid, answer_data in result.answers.items():
        print(f"  {qid}: {answer_data['answer']} ({answer_data['confidence']:.2f})")

    # Save output
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "answers": result.answers,
        "metrics": result.metrics,
        "debug_info": result.debug_info,
        "errors": result.errors,
        "is_valid": result.is_valid,
    }, indent=2), encoding="utf-8")
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
