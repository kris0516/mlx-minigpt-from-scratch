"""Data contracts: write tests now; execute only when explicitly authorized."""
import unittest
from random import Random

import mlx.core as mx
from minigpt.data import build_vocabulary, decode, encode, get_batch, split_train_validation


class DataTests(unittest.TestCase):
    def test_character_roundtrip(self):
        stoi, itos = build_vocabulary("baba")
        self.assertEqual(stoi, {"a": 0, "b": 1})
        ids = encode("abba", stoi)
        mx.eval(ids)
        self.assertEqual(ids.shape, (4,))
        self.assertEqual(decode([int(v) for v in ids.tolist()], itos), "abba")

    def test_contiguous_split(self):
        stream = mx.array([0, 1, 2, 3, 4], dtype=mx.int32)
        train, val = split_train_validation(stream, fraction=0.6)
        mx.eval(train, val)
        self.assertEqual(train.tolist(), [0, 1, 2])
        self.assertEqual(val.tolist(), [3, 4])

    def test_shifted_batch(self):
        stream = mx.array([0, 1, 2, 3, 4], dtype=mx.int32)
        X, Y, starts = get_batch(stream, 2, 2, Random(7))
        mx.eval(X, Y)
        self.assertEqual(X.shape, (2, 2))
        self.assertEqual(Y.shape, (2, 2))
        for x, y, start in zip(X.tolist(), Y.tolist(), starts):
            self.assertEqual(x, [start, start + 1])
            self.assertEqual(y, [start + 1, start + 2])

    def test_reject_insufficient_context(self):
        with self.assertRaises(ValueError):
            get_batch(mx.array([0, 1]), 1, 2, Random(0))


if __name__ == "__main__":
    unittest.main()
