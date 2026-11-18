"""Opt-in command-line entrypoints for baseline training and inference.

Importing this module does not access data, initialize MLX objects,
train, sample, benchmark, or write files. Work starts only via main().
Use: PYTHONPATH=src uv run python -m minigpt.cli --help
"""
import argparse
import json
from pathlib import Path
from random import Random

import mlx.core as mx
import mlx.optimizers as optim

from .config import ModelConfig, TrainConfig
from .data import build_vocabulary, decode, encode, load_text, split_train_validation
from .generate import generate_greedy, generate_sample
from .model import MiniGPT
from .train import make_train_step, run_training


def train_command(args):
    output_dir = Path(args.output)
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite: {output_dir}")
    text = load_text(args.corpus)  # explicit path, never download
    stoi, itos = build_vocabulary(text)
    model_cfg = ModelConfig(
        vocab_size=len(itos), dims=args.dims,
        num_heads=args.heads, num_blocks=args.layers,
        max_seq_len=args.context,
    )
    train_cfg = TrainConfig(
        batch_size=args.batch, block_size=args.context, updates=args.updates,
        learning_rate=args.learning_rate, seed=args.seed,
        val_every=args.val_every, val_batches=args.val_batches,
    )
    train_cfg.check_model(model_cfg)
    tokens = encode(text, stoi)
    train_tokens, val_tokens = split_train_validation(tokens)
    if min(train_tokens.shape[0], val_tokens.shape[0]) <= train_cfg.block_size:
        raise ValueError("Train/validation splits must each contain block_size+1 IDs")

    mx.random.seed(train_cfg.seed)  # model initialization seed
    model = MiniGPT(**model_cfg.as_model_kwargs())
    optimizer = optim.Adam(
        learning_rate=train_cfg.learning_rate,
        betas=[0.9, 0.999], eps=1e-8, bias_correction=True,
    )
    optimizer.init(model.trainable_parameters())
    mx.eval(model.parameters(), optimizer.state)

    train_step = make_train_step(model, optimizer, max_grad_norm=train_cfg.max_grad_norm)
    records = run_training(
        model, train_step, train_tokens,
        batch_size=train_cfg.batch_size, block_size=train_cfg.block_size,
        updates=train_cfg.updates, train_rng=Random(train_cfg.seed + 1),
        val_tokens=val_tokens, val_rng=Random(train_cfg.seed + 2),
        val_every=train_cfg.val_every, val_batches=train_cfg.val_batches,
    )

    output_dir.mkdir(parents=True, exist_ok=False)
    model.save_weights(str(output_dir / "model.safetensors"))
    manifest = {
        "format_version": 1,
        "checkpoint_kind": "weights-only-for-inference",
        "model": model_cfg.as_model_kwargs(),
        "train": train_cfg.to_dict(),
        "itos": itos,
        "completed_updates": train_cfg.updates,
        "history": records,
    }
    with (output_dir / "run.json").open("x", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print("Wrote inference weights and metadata:", output_dir)
    print("Optimizer/RNG resume state was NOT saved by this CLI.")
    return output_dir


def generate_command(args):
    checkpoint = Path(args.checkpoint)
    with (checkpoint / "run.json").open("r", encoding="utf-8") as f:
        manifest = json.load(f)
    if manifest.get("format_version") != 1:
        raise ValueError("Unsupported metadata version")
    if manifest.get("checkpoint_kind") != "weights-only-for-inference":
        raise ValueError("Unsupported checkpoint kind")
    itos = manifest["itos"]
    if not isinstance(itos, list) or not itos or any(type(c) is not str or len(c) != 1 for c in itos):
        raise ValueError("Invalid saved vocabulary")
    model_cfg = ModelConfig(**manifest["model"])
    if model_cfg.vocab_size != len(itos) or len(set(itos)) != len(itos):
        raise ValueError("Tokenizer/model metadata mismatch")
    stoi = {ch: i for i, ch in enumerate(itos)}
    model = MiniGPT(**model_cfg.as_model_kwargs())
    model.load_weights(str(checkpoint / "model.safetensors"), strict=True)
    mx.eval(model.parameters())
    prompt = [int(x) for x in encode(args.prompt, stoi).tolist()]
    context = model_cfg.max_seq_len
    if args.temperature is None:
        output = generate_greedy(model, prompt, args.new_tokens, block_size=context)
    else:
        output, _ = generate_sample(
            model, prompt, args.new_tokens, block_size=context,
            temperature=args.temperature, key=mx.random.key(args.seed),
        )
    mx.eval(output)
    print(decode([int(v) for v in output.tolist()[0]], itos))


def build_parser():
    parser = argparse.ArgumentParser(description="MiniGPT educational baseline; MLX 0.32.2")
    commands = parser.add_subparsers(dest="command", required=True)
    tr = commands.add_parser("train", help="explicitly train from local UTF-8 corpus")
    tr.add_argument("--corpus", required=True)
    tr.add_argument("--output", required=True, help="new, nonexisting directory")
    tr.add_argument("--dims", type=int, default=64)
    tr.add_argument("--heads", type=int, default=4)
    tr.add_argument("--layers", type=int, default=2)
    tr.add_argument("--context", type=int, default=64)
    tr.add_argument("--batch", type=int, default=4)
    tr.add_argument("--updates", type=int, default=20)
    tr.add_argument("--learning-rate", type=float, default=3e-4)
    tr.add_argument("--val-every", type=int, default=5)
    tr.add_argument("--val-batches", type=int, default=2)
    tr.add_argument("--seed", type=int, default=2026)
    ge = commands.add_parser("generate", help="explicitly load inference weights and generate")
    ge.add_argument("--checkpoint", required=True)
    ge.add_argument("--prompt", required=True)
    ge.add_argument("--new-tokens", type=int, default=30)
    ge.add_argument("--temperature", type=float, default=None, help="omit for greedy")
    ge.add_argument("--seed", type=int, default=7)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.command == "train":
        return train_command(args)
    return generate_command(args)


if __name__ == "__main__":
    main()
