import unittest
import numpy as np
import numpy.testing as npt

from vector_quantization.codebooks import random_codebook


class TestRandomCodebook(unittest.TestCase):
    def test_random_codebook(self):
        vectors = np.array([[1, 1], [2, 2], [1, 1], [3, 3], [4, 4], [1, 1]])
        codebook = random_codebook(vectors, 4)
        codebook = np.array(codebook)
        print(codebook)
        self.assertTrue([1, 1] in codebook)
        self.assertTrue([2, 2] in codebook)
        self.assertTrue([3, 3] in codebook)
        self.assertTrue([4, 4] in codebook)

    def test_seed_always_same(self):
        vectors = np.array([[1, 1], [2, 2], [1, 1], [3, 3], [4, 4], [1, 1], [3, 3], [5, 5], [9, 9], [21, 21]])
        codebook_1 = random_codebook(vectors, 4, seed=42)
        codebook_2 = random_codebook(vectors, 4, seed=42)
        npt.assert_almost_equal(codebook_1, codebook_2)
