"""Tiny optional overfit diagnostic. Never run at lesson publication time.

Run later ONLY with explicit learner approval:
PYTHONPATH=src uv run python scripts/tiny_overfit.py --updates 50
"""
import argparse
import json
import math

import mlx.core as mx
import mlx.optimizers as optim

from minigpt.model import MiniGPT
from minigpt.train import language_model_loss, make_train_step


def tiny_overfit(updates: int = 50):
    if type(updates) is not int or updates <= 0:
        raise ValueError("updates must be positive")
    mx.random.seed(17)
    model = MiniGPT(vocab_size=3, dims=16, num_heads=4,
                    num_blocks=1, max_seq_len=4)
    optimizer = optim.Adam(learning_rate=1e-3, betas=[0.9, 0.999],
                           eps=1e-8, bias_correction=True)
    optimizer.init(model.trainable_parameters())
    X = mx.array([[0, 1, 2, 0]], dtype=mx.int32)
    Y = mx.array([[1, 2, 0, 1]], dtype=mx.int32)
    start_loss = language_model_loss(model, X, Y)
    mx.eval(start_loss)
    initial = float(start_loss.item())
    train_step = make_train_step(model, optimizer, max_grad_norm=1.0)
    losses = []
    for _ in range(updates):
        loss, _ = train_step(X, Y)
        losses.append(float(loss.item()))  # measured on parameters before this update
    final_loss = language_model_loss(model, X, Y)
    mx.eval(final_loss)
    final = float(final_loss.item())
    if not all(math.isfinite(v) for v in (initial, final, *losses)):
        raise FloatingPointError("Nonfinite training curve")
    return {"updates": updates, "initial_loss": initial,
            "train_loss_pre_update": losses, "final_loss_current": final,
            "loss_decreased": bool(final < initial)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--updates", type=int, default=50)
    args = parser.parse_args()
    result = tiny_overfit(args.updates)
    print(json.dumps(result, indent=2))
    if not result["loss_decreased"]:
        raise AssertionError("Tiny overfit did not reduce loss; investigate before scale-up")


if __name__ == "__main__":
    main()
