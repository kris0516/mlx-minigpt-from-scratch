"""Causal semantics and shape contracts for baseline MiniGPT."""
import unittest

import mlx.core as mx
from minigpt.model import MiniGPT


class ModelTests(unittest.TestCase):
    def make_model(self):
        mx.random.seed(19)
        return MiniGPT(vocab_size=8, dims=16, num_heads=4,
                       num_blocks=1, max_seq_len=8)

    def test_forward_shape(self):
        model = self.make_model()
        tokens = mx.array([[1, 2, 3], [3, 2, 1]], dtype=mx.int32)
        logits = model(tokens)
        mx.eval(logits)
        self.assertEqual(logits.shape, (2, 3, 8))

    def test_causal_mask_upper_triangle(self):
        T = 3
        scores = mx.zeros((T, T), dtype=mx.float32)
        allowed = mx.arange(T)[:, None] >= mx.arange(T)[None, :]
        weights = mx.softmax(mx.where(allowed, scores, float("-inf")), axis=-1)
        mx.eval(weights)
        for i in range(T):
            for j in range(i + 1, T):
                self.assertEqual(float(weights[i, j].item()), 0.0)

    def test_future_token_does_not_change_prefix_logits(self):
        model = self.make_model()
        a = mx.array([[1, 2, 3, 4]], dtype=mx.int32)
        b = mx.array([[1, 2, 5, 6]], dtype=mx.int32)
        out_a = model(a)
        out_b = model(b)
        mx.eval(out_a, out_b)
        ok = mx.allclose(out_a[:, :2, :], out_b[:, :2, :], rtol=1e-5, atol=1e-5)
        self.assertTrue(bool(ok.item()))

    def test_context_length_guard(self):
        model = self.make_model()
        with self.assertRaises(ValueError):
            model(mx.zeros((1, 9), dtype=mx.int32))


if __name__ == "__main__":
    unittest.main()
