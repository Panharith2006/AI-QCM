from __future__ import annotations

import numpy as np

from src.OCR.ocr_preprocess import preprocess_for_ocr


def read_easyocr(reader, img, allowlist: str, debug: bool = False):
    try:
        raw_img = preprocess_for_ocr(img, use_threshold=False, debug=debug)
        raw_text = ""
        raw_conf = 0.0

        if raw_img is not None and raw_img.size > 0:
            raw_results = reader.readtext(
                raw_img,
                allowlist=allowlist,
                detail=1,
                paragraph=False,
            )
            if raw_results:
                raw_texts = [r[1].upper().strip() for r in raw_results]
                raw_confs = [r[2] for r in raw_results]
                raw_text = " ".join(raw_texts)
                raw_conf = float(np.mean(raw_confs)) if raw_confs else 0.0
                if debug:
                    print(f"      EasyOCR (raw): '{raw_text}' (conf={raw_conf:.2f})")
            elif debug:
                print("      EasyOCR (raw): No text detected")

        if raw_text and raw_conf >= 0.2:
            return raw_text, raw_conf

        threshold_img = preprocess_for_ocr(img, use_threshold=True, debug=debug)
        if threshold_img is not None and threshold_img.size > 0:
            threshold_results = reader.readtext(
                threshold_img,
                allowlist=allowlist,
                detail=1,
                paragraph=False,
            )
            if threshold_results:
                threshold_texts = [r[1].upper().strip() for r in threshold_results]
                threshold_confs = [r[2] for r in threshold_results]
                threshold_text = " ".join(threshold_texts)
                threshold_conf = float(np.mean(threshold_confs)) if threshold_confs else 0.0
                if debug:
                    print(f"      EasyOCR (threshold): '{threshold_text}' (conf={threshold_conf:.2f})")

                if threshold_text and threshold_conf > raw_conf:
                    if debug:
                        print(f"      EasyOCR selected threshold fallback: '{threshold_text}' (conf={threshold_conf:.2f})")
                    return threshold_text, threshold_conf
            elif debug:
                print("      EasyOCR (threshold): No text detected")

        if raw_text:
            return raw_text, raw_conf

        if debug:
            print("      EasyOCR: No text detected")
        return "", 0.0

    except Exception as e:
        if debug:
            print(f"      EasyOCR error: {e}")
        return "", 0.0