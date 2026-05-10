from __future__ import annotations

import cv2


def ensure_bgr(img):
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.ndim == 3 and img.shape[2] == 1:
        return cv2.cvtColor(img[:, :, 0], cv2.COLOR_GRAY2BGR)
    return img


def crop_to_content(img, debug: bool = False):
    try:
        if img.ndim == 2:
            gray = img
        else:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        mask = cv2.inRange(gray, 0, 245)
        coords = cv2.findNonZero(mask)

        if coords is None:
            return img

        x, y, w, h = cv2.boundingRect(coords)
        margin = max(2, int(0.12 * max(w, h)))

        x1 = max(0, x - margin)
        y1 = max(0, y - margin)
        x2 = min(img.shape[1], x + w + margin)
        y2 = min(img.shape[0], y + h + margin)

        if x1 >= x2 or y1 >= y2:
            return img

        return img[y1:y2, x1:x2]

    except Exception as e:
        if debug:
            print(f"      Content crop error: {e}")
        return img


def resize_preserve_aspect_ratio(img, min_side: int = 128, max_side: int = 320, debug: bool = False):
    if img is None or img.size == 0:
        return img

    h, w = img.shape[:2]
    longest_side = max(h, w)

    if longest_side < min_side:
        scale = min_side / float(longest_side)
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        if debug:
            print(f"        Upscaled from {h}x{w} to {new_h}x{new_w}")
    elif longest_side > max_side:
        scale = max_side / float(longest_side)
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        if debug:
            print(f"        Downscaled from {h}x{w} to {new_h}x{new_w}")

    return img


def apply_light_threshold(img, debug: bool = False):
    try:
        if img.ndim == 2:
            gray = img
        else:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        binary = cv2.adaptiveThreshold(
            enhanced,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=11,
            C=2,
        )

        processed = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        if debug:
            print("Applied light threshold fallback")

        return processed

    except Exception as e:
        if debug:
            print(f"Threshold fallback error: {e}")
        return img


def preprocess_for_ocr(img, use_threshold: bool = False, debug: bool = False):
    if img is None or img.size == 0:
        return img

    img = ensure_bgr(img)
    img = crop_to_content(img, debug=debug)
    img = resize_preserve_aspect_ratio(img, debug=debug)
    img = cv2.copyMakeBorder(
        img,
        12,
        12,
        12,
        12,
        cv2.BORDER_CONSTANT,
        value=(255, 255, 255),
    )

    if use_threshold:
        img = apply_light_threshold(img, debug=debug)

    return img