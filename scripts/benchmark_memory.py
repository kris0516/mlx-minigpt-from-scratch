"""Optional forward-only MLX memory probe; no work at import time.

Run later ONLY as an explicit user action:
PYTHONPATH=src uv run python scripts/benchmark_memory.py --batch 1 --context 16
One configuration per fresh process; NOT a system RAM/Swap measurement.
"""
import argparse
import json
from pathlib import Path

import mlx.core as mx

from minigpt.config import ModelConfig
from minigpt.model import MiniGPT


def snapshot():
    """All counters are allocator bytes, not macOS memory or dedicated VRAM."""
    return {
        "active_bytes": mx.get_active_memory(),
        "cache_bytes": mx.get_cache_memory(),
        "raw_peak_bytes": mx.get_peak_memory(),
    }


def probe(*, batch: int, context: int, dims: int, heads: int, layers: int):
    if type(batch) is not int or batch <= 0:
        raise ValueError("batch must be positive")
    config = ModelConfig(vocab_size=32, dims=dims, num_heads=heads,
                         num_blocks=layers, max_seq_len=context)
    mx.random.seed(17)
    model = MiniGPT(**config.as_model_kwargs())
    ids = mx.zeros((batch, context), dtype=mx.int32)
    # Establish an explicit pre-forward baseline.
    mx.eval(model.parameters(), ids)
    before = snapshot()
    mx.reset_peak_memory()  # v0.32.2 resets raw peak to zero, not current active.

    logits = model(ids)
    mx.eval(logits)
    after = snapshot()
    window_peak = max(before["active_bytes"], after["raw_peak_bytes"])
    return {
        "protocol": "single-forward-after-init",
        "config": {"B": batch, "T": context, "C": dims,
                   "H": heads, "N": layers, "V": 32},
        "logits_shape": list(logits.shape),
        "dense_score_logical_bytes_if_fp32": batch * heads * context * context * 4,
        "baseline": before,
        "after": after,
        "window_peak_active_bytes": window_peak,
        "window_peak_above_baseline_bytes": window_peak - before["active_bytes"],
        "end_active_minus_baseline_bytes": after["active_bytes"] - before["active_bytes"],
        "system_ram_bytes": None,
        "macos_swap_bytes": None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--context", type=int, default=16)
    parser.add_argument("--dims", type=int, default=32)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--output", type=Path, default=None,
                        help="optional new JSON file; never overwrite")
    args = parser.parse_args()
    record = probe(batch=args.batch, context=args.context, dims=args.dims,
                   heads=args.heads, layers=args.layers)
    body = json.dumps(record, ensure_ascii=False, indent=2)
    print(body)
    if args.output is not None:
        with args.output.open("x", encoding="utf-8") as f:
            f.write(body + "\n")


if __name__ == "__main__":
    main()
