"""Weights-only save/load round trip in a temporary directory."""
import tempfile
import unittest
from pathlib import Path

import mlx.core as mx
from minigpt.model import MiniGPT


class WeightRoundTripTests(unittest.TestCase):
    def test_save_load_preserves_forward(self):
        cfg = dict(vocab_size=7, dims=16, num_heads=4,
                   num_blocks=1, max_seq_len=8)
        mx.random.seed(11)
        first = MiniGPT(**cfg)
        X = mx.array([[1, 2, 3]], dtype=mx.int32)
        expected = first(X)
        mx.eval(expected, first.parameters())
        with tempfile.TemporaryDirectory() as workdir:
            weights = Path(workdir) / "model.safetensors"
            first.save_weights(str(weights))
            mx.random.seed(12)
            restored = MiniGPT(**cfg)
            restored.load_weights(str(weights), strict=True)
            actual = restored(X)
            mx.eval(actual)
            self.assertTrue(bool(mx.allclose(expected, actual, rtol=1e-5, atol=1e-6).item()))


if __name__ == "__main__":
    unittest.main()
