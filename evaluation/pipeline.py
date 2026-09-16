import json
from pathlib import Path
import pandas as pd

DATASET_PATH = Path(__file__).resolve().parent.parent / "datasets" / "eval_dataset.json"
FEATURE_TABLE_PATH = Path(__file__).resolve().parent.parent / "data" / "eval_features.csv"

def _load_dataset():
    with DATASET_PATH.open() as f:
        return json.load(f)

def run_offline_evaluation():
    records = _load_dataset()
    rows = []
    for r in records:
        text = r["model_output"]
        rows.append({"id": r.get("id"), "length": len(text.split()), "label": r.get("label"), "human_score": r.get("human_score")})
    df = pd.DataFrame(rows)
    FEATURE_TABLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(FEATURE_TABLE_PATH, index=False)
    return {"n_examples": len(records), "feature_table_path": str(FEATURE_TABLE_PATH)}

def run_single_report(model_output: str, ground_truth: str = "", human_score=None):
    return {"model_output": model_output, "features": {"length": len(model_output.split()), "ground_truth_provided": bool(ground_truth)}, "human_score": human_score}
