"""HTTP serving benchmark for vLLM, TGI and llama.cpp endpoints."""
import argparse
import statistics
import time
import requests

BACKENDS = {
    "vllm": "http://localhost:8001/v1/completions",
    "tgi": "http://localhost:8002/generate",
    "llama.cpp": "http://localhost:8003/completion",
}


def benchmark(url: str, payload: dict, runs: int = 10):
    latencies = []
    for _ in range(runs):
        start = time.perf_counter()
        response = requests.post(url, json=payload, timeout=60)
        response.raise_for_status()
        latencies.append((time.perf_counter() - start) * 1000)
    return {"mean_ms": statistics.mean(latencies), "p50_ms": statistics.median(latencies), "requests": runs}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=list(BACKENDS), default="vllm")
    parser.add_argument("--prompt", default="Explain the transformer architecture.")
    parser.add_argument("--num_requests", type=int, default=10)
    args = parser.parse_args()
    result = benchmark(BACKENDS[args.backend], {"prompt": args.prompt, "max_tokens": 64, "temperature": 0.0}, args.num_requests)
    print(result)


if __name__ == "__main__":
    main()
