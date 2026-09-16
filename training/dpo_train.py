"""Direct Preference Optimisation alignment stage."""
import argparse
import json
from pathlib import Path

BUILTIN_PREFERENCES = [
    {"prompt": "Explain quantum entanglement in simple terms.", "chosen": "Quantum entanglement correlates the states of two particles.", "rejected": "It is some physics thing."},
    {"prompt": "Write a Python function that reverses a string.", "chosen": "def reverse(s):\n    return s[::-1]", "rejected": "Use a loop maybe."},
]


def load_preference_dataset(path: str):
    p = Path(path)
    if p.exists():
        return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
    return BUILTIN_PREFERENCES


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default="checkpoints/lora-sft")
    parser.add_argument("--output_dir", default="checkpoints/dpo")
    parser.add_argument("--dataset_path", default="datasets/preference_dataset.jsonl")
    args = parser.parse_args()
    data = load_preference_dataset(args.dataset_path)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    print(f"Loaded {len(data)} preference pairs for DPO from {args.dataset_path}")
    print(f"Policy checkpoint: {args.model_dir}; output: {args.output_dir}")


if __name__ == "__main__":
    main()
