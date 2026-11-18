"""Character-level corpus and shifted-window batch utilities.

No I/O, downloads, or model execution occurs at import time.
MLX is pinned by this repository to v0.32.2. All sampling uses an
explicit caller-owned Python random.Random state.
"""
from pathlib import Path
from random import Random

import mlx.core as mx


def load_text(path: str | Path) -> str:
    """Read an explicitly selected UTF-8 file; never download."""
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(file_path)
    text = file_path.read_text(encoding="utf-8")
    if not text:
        raise ValueError("Corpus must be nonempty")
    return text


def build_vocabulary(text: str) -> tuple[dict[str, int], list[str]]:
    """Return sorted Unicode character-to-ID and ID-to-character maps."""
    if not isinstance(text, str) or not text:
        raise ValueError("text must be a nonempty string")
    itos = sorted(set(text))
    stoi = {ch: index for index, ch in enumerate(itos)}
    return stoi, itos


def encode(text: str, stoi: dict[str, int]) -> mx.array:
    """Encode using the supplied, fixed vocabulary; unknown IDs are errors."""
    if not text or not isinstance(text, str):
        raise ValueError("text must be nonempty")
    try:
        ids = [stoi[ch] for ch in text]
    except KeyError as exc:
        raise ValueError(f"Character absent from vocabulary: {exc.args[0]!r}") from exc
    return mx.array(ids, dtype=mx.int32)


def decode(ids, itos: list[str]) -> str:
    """Decode an iterable of integer IDs without constructing arrays."""
    if not itos:
        raise ValueError("itos must be nonempty")
    chars = []
    for index in ids:
        if type(index) is not int or not 0 <= index < len(itos):
            raise ValueError(f"Invalid token ID: {index!r}")
        chars.append(itos[index])
    return "".join(chars)


def split_train_validation(tokens: mx.array, fraction: float = 0.9):
    """Contiguous split, with both sides nonempty; shape (N,) -> (Ntr,), (Nv,)."""
    if tokens.ndim != 1 or tokens.shape[0] < 2:
        raise ValueError("Expected 1-D token stream of at least two IDs")
    if not isinstance(fraction, (int, float)) or isinstance(fraction, bool) or not 0 < fraction < 1:
        raise ValueError("fraction must be in (0,1)")
    boundary = int(tokens.shape[0] * fraction)
    if not 0 < boundary < tokens.shape[0]:
        raise ValueError("Split would leave an empty partition")
    return tokens[:boundary], tokens[boundary:]


def get_batch(tokens: mx.array, batch_size: int, block_size: int, rng: Random):
    """Return shifted (X,Y) and sampled start offsets; no implicit RNG.

    Requires N >= block_size+1. Outputs X,Y: (B,T), starts: list[int].
    Sampling is within the one split passed by the caller.
    """
    if tokens.ndim != 1:
        raise ValueError("tokens must have shape (N,)")
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("batch_size must be positive int")
    if type(block_size) is not int or block_size <= 0:
        raise ValueError("block_size must be positive int")
    if not isinstance(rng, Random):
        raise TypeError("Pass an explicit random.Random instance")
    n = tokens.shape[0]
    if n <= block_size:
        raise ValueError("Stream must contain at least block_size+1 tokens")
    starts = [rng.randrange(n - block_size) for _ in range(batch_size)]
    xs = [tokens[i:i + block_size] for i in starts]
    ys = [tokens[i + 1:i + block_size + 1] for i in starts]
    return mx.stack(xs, axis=0), mx.stack(ys, axis=0), starts
