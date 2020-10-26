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
