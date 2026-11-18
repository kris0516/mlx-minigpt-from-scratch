"""Tiny optional training diagnostic: smoke decrease versus strict memorization.

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
    # Distinguish minimal optimization signal from actual toy memorization.
    final_logits = model(X)  # (1,4,3), after the final Adam update
    final_loss = language_model_loss(model, X, Y)
    predicted = mx.argmax(final_logits, axis=-1).astype(mx.int32)  # (1,4)
    token_accuracy = mx.mean((predicted == Y).astype(mx.float32))  # scalar ()
    mx.eval(final_loss, token_accuracy)
    final = float(final_loss.item())
    accuracy = float(token_accuracy.item())
    if not all(math.isfinite(v) for v in (initial, final, accuracy, *losses)):
        raise FloatingPointError("Nonfinite training curve or token accuracy")

    # A tiny loss drop is only a smoke result, NOT proof of memorization.
    smoke_loss_decreased = bool(final < initial)
    memorized = bool(accuracy == 1.0 and final <= 0.1)
    return {"updates": updates, "initial_loss": initial,
            "train_loss_pre_update": losses, "final_loss_current": final,
            "loss_decreased": smoke_loss_decreased,  # legacy field: smoke only
            "training_token_accuracy": accuracy,
            "memorization_target": {"token_accuracy": 1.0, "final_loss_at_most": 0.1},
            "memorized": memorized}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--updates", type=int, default=50)
    parser.add_argument(
        "--require-memorization", action="store_true",
        help="also require 100%% token accuracy and final loss <= 0.1",
    )
    args = parser.parse_args()
    result = tiny_overfit(args.updates)
    print(json.dumps(result, indent=2))
    if not result["loss_decreased"]:
        raise AssertionError("Smoke check failed: toy loss did not decrease")
    if args.require_memorization and not result["memorized"]:
        raise AssertionError(
            "Toy memorization not reached: require 100% accuracy and loss <= 0.1"
        )
    if not result["memorized"]:
        print("Smoke decrease alone is not toy memorization; strict gate unmet.")


if __name__ == "__main__":
    main()
