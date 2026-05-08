import json
import re


def extract_allowed_labels(raw_response: str, allowed_labels: list[str]) -> list[str]:
    """
    Conservative fallback extraction.
    First try JSON:
    - {"labels": ["31", "32"]}
    - {"label": "31"}
    - {"label": "31, 32"}
    Then, only if JSON parsing fails, scan the raw response for exact allowed-label tokens.
    Return sorted deduped labels in allowed_labels order.
    """
    extracted = []
    try:
        data = json.loads(raw_response)
        if isinstance(data, dict):
            if "labels" in data and isinstance(data["labels"], list):
                for item in data["labels"]:
                    if isinstance(item, str):
                        extracted.append(item)
            elif "label" in data and isinstance(data["label"], str):
                parts = re.split(r"[,;/|\n\s]+", data["label"])
                for p in parts:
                    if p:
                        extracted.append(p)
    except json.JSONDecodeError:
        for label in allowed_labels:
            pattern = rf"\b{label}\b" if label.isdigit() else rf"\b{re.escape(label)}\b"
            if re.search(pattern, raw_response, flags=re.IGNORECASE):
                extracted.append(label)

    # Dedup and sort by allowed_labels order
    final_labels = []
    for lbl in allowed_labels:
        if lbl in extracted and lbl not in final_labels:
            final_labels.append(lbl)
    return final_labels


def normalize_prediction_response(
    raw_response: str,
    allowed_labels: list[str],
) -> tuple[list[str], str | None]:
    """
    Return (labels, error).
    labels is sorted/deduped according to allowed_labels order.
    error is None if at least one allowed label was extracted, else a message.
    """
    labels = extract_allowed_labels(raw_response, allowed_labels)
    error = None

    try:
        data = json.loads(raw_response)
        if isinstance(data, dict):
            extracted = []
            if "labels" in data and isinstance(data["labels"], list):
                for item in data["labels"]:
                    if not isinstance(item, str):
                        error = "JSON list fields must contain only strings"
                        break
                    extracted.append(item)
            elif "label" in data and isinstance(data["label"], str):
                parts = re.split(r"[,;/|\n\s]+", data["label"])
                extracted.extend([p for p in parts if p])

            if not error:
                unknown = {u for u in extracted if u} - set(allowed_labels)
                if unknown:
                    error = f"Unknown labels in JSON: {', '.join(sorted(unknown))}"
    except json.JSONDecodeError:
        pass  # Text fallback used

    if not labels and not error:
        error = "No valid allowed labels found"

    return labels, error
