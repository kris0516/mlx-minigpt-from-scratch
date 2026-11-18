"""Optional small-scale forward/generation wall-time probe.

No timing occurs at import or publication; execute only by user choice.
Numbers are NOT temperature, power, thermal-throttling or Swap measurements.
"""
import argparse
import json
import statistics
import time
from pathlib import Path

import mlx.core as mx

from minigpt.config import ModelConfig
from minigpt.generate import generate_greedy
from minigpt.model import MiniGPT


def measure(*, context: int, dims: int, heads: int, layers: int,
            warmups: int, repeats: int, new_tokens: int):
    if any(type(v) is not int or v < 1 for v in (warmups, repeats, new_tokens)):
        raise ValueError("warmups/repeats/new_tokens must be positive ints")
    config = ModelConfig(vocab_size=32, dims=dims, num_heads=heads,
                         num_blocks=layers, max_seq_len=context)
    mx.random.seed(23)
    model = MiniGPT(**config.as_model_kwargs())
    model.eval()
    tokens = (mx.arange(context, dtype=mx.int32) % 32)[None, :]
    mx.eval(model.parameters(), tokens)

    for _ in range(warmups):
        mx.eval(model(tokens))
    forward_seconds = []
    for _ in range(repeats):
        start = time.perf_counter()
        out = model(tokens)
        mx.eval(out)  # Must be inside the timing window under MLX lazy evaluation.
        forward_seconds.append(time.perf_counter() - start)

    # This measures the *current* no-cache full-window generation contract.
    for _ in range(warmups):
        warm = generate_greedy(model, [0], new_tokens, block_size=context)
        mx.eval(warm)
    generation_seconds = []
    for _ in range(repeats):
        start = time.perf_counter()
        result = generate_greedy(model, [0], new_tokens, block_size=context)
        mx.eval(result)
        generation_seconds.append(time.perf_counter() - start)

    forward_median = statistics.median(forward_seconds)
    generate_median = statistics.median(generation_seconds)
    return {
        "protocol": "small-uncompiled-eval-synchronized",
        "hardware": "record separately", "mlx_version": "0.32.2",
        "model": config.as_model_kwargs(),
        "batch_size": 1, "warmups": warmups, "repeats": repeats,
        "new_tokens": new_tokens,
        "forward_seconds": forward_seconds,
        "forward_median_seconds": forward_median,
        "forward_positions_per_second": context / forward_median,
        "generate_seconds": generation_seconds,
        "generate_median_seconds": generate_median,
        "generated_tokens_per_second": new_tokens / generate_median,
        "temperature_celsius": None,
        "watts": None,
        "macos_swap_bytes": None,
        "thermal_instrumentation": "not performed",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--context", type=int, default=16)
    parser.add_argument("--dims", type=int, default=32)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--new-tokens", type=int, default=5)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    record = measure(context=args.context, dims=args.dims,
                     heads=args.heads, layers=args.layers,
                     warmups=args.warmups, repeats=args.repeats,
                     new_tokens=args.new_tokens)
    body = json.dumps(record, ensure_ascii=False, indent=2)
    print(body)
    if args.output is not None:
        with args.output.open("x", encoding="utf-8") as f:
            f.write(body + "\n")


if __name__ == "__main__":
    main()
