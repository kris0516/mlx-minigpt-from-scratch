"""Untied baseline MiniGPT, directly extracted from Step 15.4.
Do not silently change checkpoint structure or enable Step 22 variants.
"""

import math
import mlx.core as mx
import mlx.nn as nn


class CausalMultiHeadAttention(nn.Module):
    def __init__(self, dims: int, num_heads: int):
        super().__init__()

        if dims <= 0:
            raise ValueError('dims must be positive')
        if num_heads <= 0:
            raise ValueError('num_heads must be positive')
        if dims % num_heads != 0:
            raise ValueError('dims must be divisible by num_heads')

        self.dims = dims
        self.num_heads = num_heads
        self.head_dim = dims // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)  # Python float, not trainable

        self.qkv_proj = nn.Linear(
            input_dims=dims,
            output_dims=3 * dims,
            bias=False,
        )
        self.c_proj = nn.Linear(
            input_dims=dims,
            output_dims=dims,
            bias=False,
        )

    def __call__(self, x: mx.array) -> mx.array:
        if len(x.shape) != 3:
            raise ValueError('attention input must have shape (B,T,C)')

        B, T, C = x.shape
        if C != self.dims:
            raise ValueError('attention input width must equal dims')

        H = self.num_heads
        D = self.head_dim

        qkv = self.qkv_proj(x)  # (B,T,3C)
        q, k, v = mx.split(qkv, indices_or_sections=3, axis=-1)

        q = mx.reshape(q, (B, T, H, D))
        k = mx.reshape(k, (B, T, H, D))
        v = mx.reshape(v, (B, T, H, D))

        q = mx.swapaxes(q, 1, 2)  # (B,H,T,D)
        k = mx.swapaxes(k, 1, 2)
        v = mx.swapaxes(v, 1, 2)

        scores = mx.matmul(
            q,
            mx.swapaxes(k, -1, -2),
        ) * self.scale  # (B,H,T,T)

        # IMPORTANT: structural mask is local to forward.
        # It is NOT stored as self.causal_mask.
        future = mx.triu(mx.ones((T, T)), k=1) > 0
        mask = mx.where(
            future,
            mx.array(float('-inf'), dtype=scores.dtype),
            mx.array(0.0, dtype=scores.dtype),
        )

        weights = mx.softmax(scores + mask, axis=-1)
        head_out = mx.matmul(weights, v)  # (B,H,T,D)

        token_major = mx.swapaxes(head_out, 1, 2)  # (B,T,H,D)
        merged = mx.reshape(token_major, (B, T, C))
        return self.c_proj(merged)  # (B,T,C)


class FeedForwardMLP(nn.Module):
    def __init__(self, dims: int):
        super().__init__()
        self.fc1 = nn.Linear(dims, 4 * dims, bias=True)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(4 * dims, dims, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        return self.fc2(self.act(self.fc1(x)))


class TransformerBlock(nn.Module):
    def __init__(self, dims: int, num_heads: int):
        super().__init__()
        self.norm1 = nn.RMSNorm(dims=dims)
        self.attn = CausalMultiHeadAttention(dims=dims, num_heads=num_heads)
        self.norm2 = nn.RMSNorm(dims=dims)
        self.mlp = FeedForwardMLP(dims=dims)

    def __call__(self, x: mx.array) -> mx.array:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


# Definitions only. This master notebook does not execute them.

class MiniGPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        dims: int,
        num_heads: int,
        num_blocks: int,
        max_seq_len: int,
    ):
        super().__init__()

        if vocab_size <= 0:
            raise ValueError('vocab_size must be positive')
        if dims <= 0:
            raise ValueError('dims must be positive')
        if num_heads <= 0:
            raise ValueError('num_heads must be positive')
        if num_blocks <= 0:
            raise ValueError('num_blocks must be positive')
        if max_seq_len <= 0:
            raise ValueError('max_seq_len must be positive')
        if dims % num_heads != 0:
            raise ValueError('dims must be divisible by num_heads')

        # Python metadata: these are not trainable mx.array parameters.
        self.vocab_size = vocab_size
        self.dims = dims
        self.max_seq_len = max_seq_len

        # Learned lookup tables.
        self.token_embedding = nn.Embedding(
            num_embeddings=vocab_size,
            dims=dims,
        )
        self.position_embedding = nn.Embedding(
            num_embeddings=max_seq_len,
            dims=dims,
        )

        # A list of distinct Module instances; MLX recursively tracks them.
        self.blocks = [
            TransformerBlock(
                dims=dims,
                num_heads=num_heads,
            )
            for _ in range(num_blocks)
        ]

        self.final_norm = nn.RMSNorm(dims=dims)
        self.lm_head = nn.Linear(
            input_dims=dims,
            output_dims=vocab_size,
            bias=False,
        )

    def __call__(self, token_ids: mx.array) -> mx.array:
        if len(token_ids.shape) != 2:
            raise ValueError('token_ids must have shape (B,T)')

        B, T = token_ids.shape

        if T < 1:
            raise ValueError('sequence length T must be at least 1')
        if T > self.max_seq_len:
            raise ValueError(
                f'sequence length {T} exceeds max_seq_len {self.max_seq_len}'
            )

        # Forward-local structural indices: not Module parameters.
        position_ids = mx.arange(T)  # (T,)

        token_features = self.token_embedding(token_ids)        # (B,T,C)
        position_features = self.position_embedding(position_ids)  # (T,C)
        x = token_features + position_features                  # (B,T,C)

        for block in self.blocks:
            x = block(x)  # invariant: (B,T,C)

        x = self.final_norm(x)  # (B,T,C)
        logits = self.lm_head(x)  # (B,T,V)

        expected_shape = (B, T, self.vocab_size)
        if logits.shape != expected_shape:
            raise RuntimeError(
                f'logits shape {logits.shape} != expected {expected_shape}'
            )

        return logits


# Definition only. No MiniGPT forward is executed in this master notebook.
