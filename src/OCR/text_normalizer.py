from __future__ import annotations


def extract_answer_from_text(text, options):
    text = text.upper()
    normalized = text.replace(" ", "")

    for opt in options:
        if normalized == opt or text == opt:
            return opt

    return text if text else None


def post_process_text(text: str) -> str:
    if not text:
        return ""

    text = text.upper()

    allowed = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789IVXTFNG-_.:/() ")
    cleaned = "".join([c for c in text if c in allowed])
    return " ".join(cleaned.split())