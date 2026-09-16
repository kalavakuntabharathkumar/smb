"""Post-training quantisation helpers for causal-LM checkpoints."""
import argparse
from pathlib import Path


def quantize_awq(model_dir: str, output_dir: str, bits: int = 4):
    from awq import AutoAWQForCausalLM
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoAWQForCausalLM.from_pretrained(model_dir)
    model.quantize(tokenizer, quant_config={"zero_point": True, "q_group_size": 128, "w_bit": bits, "version": "GEMM"})
    model.save_quantized(output_dir)
    tokenizer.save_pretrained(output_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--output_dir", default="checkpoints/quantized")
    parser.add_argument("--bits", type=int, default=4)
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    print(f"Quantisation target: {args.model_dir} -> {args.output_dir} ({args.bits}-bit)")


if __name__ == "__main__":
    main()
