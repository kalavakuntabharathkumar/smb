"""CLI entry point for the offline evaluation pipeline."""
import json
from evaluation.pipeline import run_offline_evaluation

if __name__ == "__main__":
    report = run_offline_evaluation()
    print(json.dumps(report, indent=2))
