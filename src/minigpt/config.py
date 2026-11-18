"""Validated, serializable configuration for the existing MiniGPT baseline."""
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    dims: int = 64
    num_heads: int = 4
    num_blocks: int = 2
    max_seq_len: int = 64

    def __post_init__(self):
        for name in ("vocab_size", "dims", "num_heads", "num_blocks", "max_seq_len"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive int")
        if self.dims % self.num_heads:
            raise ValueError("dims must be divisible by num_heads")

    def as_model_kwargs(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class TrainConfig:
    batch_size: int = 4
    block_size: int = 64
    learning_rate: float = 3e-4
    updates: int = 20
    val_every: int = 5
    val_batches: int = 2
    seed: int = 2026
    max_grad_norm: float = 1.0

    def __post_init__(self):
        import math
        for name in ("batch_size", "block_size", "updates", "val_every", "val_batches"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive int")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be nonnegative int")
        for name in ("learning_rate", "max_grad_norm"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (float, int)):
                raise ValueError(f"{name} must be finite positive")
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite positive")

    def check_model(self, model: ModelConfig):
        if self.block_size > model.max_seq_len:
            raise ValueError("block_size exceeds learned position capacity")

    def to_dict(self) -> dict:
        return asdict(self)
