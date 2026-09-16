"""LoRA supervised fine-tuning entry point."""
import argparse
from pathlib import Path

SUPPORTED_MODELS = {
    "qwen": "Qwen/Qwen2.5-0.5B-Instruct",
    "phi": "microsoft/phi-2",
}


def build_lora_config():
    from peft import LoraConfig
    return LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM", target_modules=["q_proj", "v_proj"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=SUPPORTED_MODELS, default="qwen")
    parser.add_argument("--output_dir", default="checkpoints/lora-sft")
    parser.add_argument("--epochs", type=int, default=1)
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    print(f"LoRA configuration prepared for {SUPPORTED_MODELS[args.model]}")
    print(f"Output directory: {args.output_dir}; epochs: {args.epochs}")


if __name__ == "__main__":
    main()
