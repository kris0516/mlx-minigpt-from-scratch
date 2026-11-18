"""No-cache autoregressive generation with explicit context cropping.

Preserves the untied learned-position MiniGPT baseline and Step 21 semantics.
No work is performed at import time. Supports B=1 prompt only.
"""
import math

import mlx.core as mx


def _check(model, prompt, max_new_tokens, block_size):
    if not isinstance(prompt, (list, tuple)) or not prompt:
        raise ValueError("prompt must contain at least one integer token ID")
    if any(type(v) is not int or not 0 <= v < model.vocab_size for v in prompt):
        raise ValueError("prompt ID out of vocabulary")
    if type(max_new_tokens) is not int or max_new_tokens < 0:
        raise ValueError("max_new_tokens must be a nonnegative int")
    if type(block_size) is not int or not 1 <= block_size <= model.max_seq_len:
        raise ValueError("block_size outside model position capacity")


def _last_logits(model, history, block_size):
    context = history[:, -block_size:]
    logits = model(context)
    if logits.shape != (1, context.shape[1], model.vocab_size):
        raise ValueError("model logits shape violated")
    last = logits[:, -1, :].astype(mx.float32)
    mx.eval(last)
    if not bool(mx.all(mx.isfinite(last)).item()):
        raise FloatingPointError("Nonfinite next-token logits")
    return last


def generate_greedy(model, prompt: list[int], max_new_tokens: int, *, block_size: int):
    """Return integer history (1,L+n); greedy argmax, no KV cache."""
    _check(model, prompt, max_new_tokens, block_size)
    history = mx.array([prompt], dtype=mx.int32)
    was_training = model.training
    model.eval()
    try:
        for _ in range(max_new_tokens):
            last = _last_logits(model, history, block_size)
            token = mx.argmax(last, axis=-1).astype(mx.int32)[:, None]
            history = mx.concatenate([history, token], axis=1)
            mx.eval(history)
        return history
    finally:
        model.train(was_training)


def generate_sample(model, prompt: list[int], max_new_tokens: int, *,
                    block_size: int, temperature: float, key: mx.array):
    """Categorical sampling using caller-supplied MLX key; returns history and next key."""
    _check(model, prompt, max_new_tokens, block_size)
    if not isinstance(temperature, (int, float)) or isinstance(temperature, bool):
        raise ValueError("temperature must be positive finite")
    if not math.isfinite(float(temperature)) or temperature <= 0:
        raise ValueError("temperature must be positive finite")
    history = mx.array([prompt], dtype=mx.int32)
    was_training = model.training
    model.eval()
    try:
        for _ in range(max_new_tokens):
            last = _last_logits(model, history, block_size)
            scaled = (last - mx.max(last, axis=-1, keepdims=True)) / temperature
            mx.eval(scaled)
            if not bool(mx.all(mx.isfinite(scaled)).item()):
                raise FloatingPointError("Nonfinite temperature-scaled logits")
            next_key, draw_key = mx.random.split(key)
            token = mx.random.categorical(scaled, axis=-1, key=draw_key).astype(mx.int32)[:, None]
            history = mx.concatenate([history, token], axis=1)
            key = next_key
            mx.eval(history, key)
        return history, key
    finally:
        model.train(was_training)
