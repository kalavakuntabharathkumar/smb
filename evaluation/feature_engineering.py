"""
Feature engineering for evaluation records.
"""
import numpy as np
import pandas as pd

HALLUCINATION_MARKERS = ("according to", "studies show", "it is well known", "as everyone knows")

def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) or 1e-9
    return float(np.dot(a, b) / denom)

def _hallucination_flag(text: str) -> int:
    lowered = text.lower()
    return int(any(marker in lowered for marker in HALLUCINATION_MARKERS))

def extract_features(model_output: str, ground_truth: str = "") -> dict:
    return {
        "length": len(model_output.split()),
        "char_length": len(model_output),
        "hallucination_flag": _hallucination_flag(model_output),
        "similarity_to_ground_truth": 0.0 if not ground_truth else 1.0,
    }

def build_feature_dataframe(records: list[dict]) -> pd.DataFrame:
    rows = []
    for record in records:
        row = extract_features(record["model_output"], record.get("ground_truth", ""))
        row["id"] = record.get("id")
        rows.append(row)
    return pd.DataFrame(rows)
