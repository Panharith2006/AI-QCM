from __future__ import annotations

import argparse
from pathlib import Path
import sys

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.alignment.perspective import perspective_correct
from src.detection.yolo_layout import YoloLayoutDetector
from src.quality.quality_classifier import simple_quality_check
from src.visualization.annotate import put_status_banner


def main() -> None:
    parser = argparse.ArgumentParser(description="Run OMR pipeline on a single sheet image")
    parser.add_argument("--image", required=True)
    parser.add_argument("--yolo", default="artifacts/yolo/best.pt")
    parser.add_argument("--out", default="outputs/infer_result.jpg")
    args = parser.parse_args()

    image = cv2.imread(args.image)
    if image is None:
        raise FileNotFoundError(f"Could not load: {args.image}")

    quality_label, quality_score = simple_quality_check(image)
    aligned = perspective_correct(image)

    detector = YoloLayoutDetector(args.yolo)
    detections = detector.detect(aligned)

    vis = aligned.copy()
    for det in detections:
        cv2.rectangle(vis, (det.x1, det.y1), (det.x2, det.y2), (0, 255, 0), 2)
        cv2.putText(
            vis,
            f"{det.label}:{det.confidence:.2f}",
            (det.x1, max(20, det.y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    banner = f"Quality={quality_label} ({quality_score:.2f}) | Detections={len(detections)}"
    vis = put_status_banner(vis, banner)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), vis)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
