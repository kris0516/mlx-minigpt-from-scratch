"""Explicit-state MiniGPT loss, training and validation helpers (MLX v0.32.2).

No training, model construction, or data access occurs at import time.
Uncompiled by default to keep state/evaluation boundaries transparent.
"""
import math
from random import Random

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_map

from .data import get_batch


def language_model_loss(model: nn.Module, X: mx.array, Y: mx.array) -> mx.array:
    """Raw logits (B,T,V) + shifted targets (B,T) -> scalar loss ()."""
    logits = model(X)
    if Y.shape != X.shape or logits.shape[:2] != X.shape:
        raise ValueError("X/Y/logit token-position shapes disagree")
    B, T, V = logits.shape
    losses = nn.losses.cross_entropy(
        mx.reshape(logits, (B * T, V)),
        mx.reshape(Y, (B * T,)),
        axis=-1, reduction="none",
    )
    return mx.mean(losses)


def global_grad_norm(grads) -> mx.array:
    """Euclidean norm across every trainable gradient leaf; shape ()."""
    leaves = [leaf for _, leaf in tree_flatten(grads)]
    if not leaves:
        raise ValueError("No gradient arrays")
    squared = [mx.sum(mx.square(leaf.astype(mx.float32))) for leaf in leaves]
    return mx.sqrt(sum(squared))


def make_train_step(model: nn.Module, optimizer, *, max_grad_norm: float | None = 1.0):
    """Return one uncompiled step closure bound to one model/optimizer pair."""
    if max_grad_norm is not None and (not math.isfinite(max_grad_norm) or max_grad_norm <= 0):
        raise ValueError("max_grad_norm must be positive finite or None")
    loss_and_grad = nn.value_and_grad(model, language_model_loss)

    def step(X: mx.array, Y: mx.array):
        loss_before_update, grads = loss_and_grad(model, X, Y)
        norm = global_grad_norm(grads)
        mx.eval(loss_before_update, norm)
        if not (math.isfinite(float(loss_before_update.item())) and math.isfinite(float(norm.item()))):
            raise FloatingPointError("Nonfinite loss or global gradient norm")
        if max_grad_norm is not None:
            scale = mx.minimum(
                mx.array(1.0, dtype=mx.float32),
                mx.array(max_grad_norm, dtype=mx.float32) / (norm + 1e-12),
            )
            grads = tree_map(lambda leaf: leaf * scale, grads)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)
        return loss_before_update, norm  # old-parameter loss, preclip norm
    return step


def estimate_validation_loss(model, tokens: mx.array, *, batch_size: int,
                             block_size: int, batches: int, rng: Random) -> float:
    """No optimizer update; consume only this validation RNG's state."""
    if type(batches) is not int or batches <= 0:
        raise ValueError("batches must be positive int")
    was_training = model.training
    model.eval()
    try:
        total = 0.0
        for _ in range(batches):
            X, Y, _ = get_batch(tokens, batch_size, block_size, rng)
            value = language_model_loss(model, X, Y)
            mx.eval(value)
            number = float(value.item())
            if not math.isfinite(number):
                raise FloatingPointError("Nonfinite validation loss")
            total += number
        return total / batches
    finally:
        model.train(was_training)


def run_training(model, train_step, train_tokens: mx.array, *,
                 batch_size: int, block_size: int, updates: int, train_rng: Random,
                 val_tokens: mx.array | None = None, val_rng: Random | None = None,
                 val_every: int = 5, val_batches: int = 2) -> list[dict]:
    """Execute only when explicitly called; returned records are regular Python scalars."""
    if type(updates) is not int or updates < 0:
        raise ValueError("updates must be nonnegative int")
    if type(val_every) is not int or val_every <= 0:
        raise ValueError("val_every must be positive int")
    if val_tokens is not None and val_rng is None:
        raise ValueError("Provide independent validation RNG")
    records = []
    for index in range(1, updates + 1):
        model.train()
        X, Y, _ = get_batch(train_tokens, batch_size, block_size, train_rng)
        pre_update_loss, preclip_norm = train_step(X, Y)
        record = {
            "update": index,
            "train_loss_pre_update": float(pre_update_loss.item()),
            "grad_norm_preclip": float(preclip_norm.item()),
            "validation_loss_current": None,
        }
        if val_tokens is not None and index % val_every == 0:
            record["validation_loss_current"] = estimate_validation_loss(
                model, val_tokens, batch_size=batch_size, block_size=block_size,
                batches=val_batches, rng=val_rng,
            )
        records.append(record)
    return records
