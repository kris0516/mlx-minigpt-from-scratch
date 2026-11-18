"""Generation state/shape tests with a deterministic toy logits callback."""
import unittest

import mlx.core as mx
from minigpt.generate import generate_greedy, generate_sample


class ToyNextToken:
    vocab_size = 3
    max_seq_len = 3

    def __init__(self):
        self.training = True
        self.table = mx.array([[0.0, 3.0, 0.0],
                               [0.0, 0.0, 3.0],
                               [3.0, 0.0, 0.0]], dtype=mx.float32)

    def __call__(self, ids):
        return self.table[ids]

    def eval(self):
        self.training = False

    def train(self, mode=True):
        self.training = mode


class GenerationTests(unittest.TestCase):
    def test_greedy_appends_to_full_history(self):
        toy = ToyNextToken()
        result = generate_greedy(toy, [0], 3, block_size=2)
        mx.eval(result)
        self.assertEqual(result.shape, (1, 4))
        self.assertEqual(result.tolist()[0], [0, 1, 2, 0])
        self.assertTrue(toy.training)

    def test_sampling_key_is_explicit_and_reproducible(self):
        toy = ToyNextToken()
        a, k1 = generate_sample(toy, [0], 3, block_size=2,
                                temperature=1.0, key=mx.random.key(5))
        b, k2 = generate_sample(toy, [0], 3, block_size=2,
                                temperature=1.0, key=mx.random.key(5))
        mx.eval(a, b, k1, k2)
        self.assertEqual(a.tolist(), b.tolist())
        self.assertEqual(k1.tolist(), k2.tolist())


if __name__ == "__main__":
    unittest.main()
